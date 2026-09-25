"""Verify synthetic SFace SavedModel forward/input-gradient before TF.js conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite frozen verification")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    rgb_u8 = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    nchw = np.ascontiguousarray(rgb_u8.transpose(2, 0, 1)[None].astype(np.float32))
    if hashlib.sha256(nchw.tobytes()).hexdigest() != reference["nchw_f32_sha256"]:
        raise RuntimeError("Synthetic input differs from frozen reference")
    model = tf.saved_model.load(str(args.saved_model))
    if "serving_default" not in model.signatures:
        raise RuntimeError("SavedModel has no serving_default signature")
    fn = model.signatures["serving_default"]
    _, keyword_specs = fn.structured_input_signature
    if len(keyword_specs) != 1:
        raise RuntimeError(f"Expected one runtime input, got {list(keyword_specs)}")
    input_name, spec = next(iter(keyword_specs.items()))
    shape = spec.shape.as_list()
    if shape == [1, 3, 112, 112]:
        layout = "NCHW"
        input_np = nchw
        gradient_to_nchw = lambda grad: grad
        direction_np = np.asarray(reference["direction_nchw"], dtype=np.float32).reshape(nchw.shape)
    elif shape == [1, 112, 112, 3]:
        layout = "NHWC"
        input_np = np.transpose(nchw, (0, 2, 3, 1)).copy()
        gradient_to_nchw = lambda grad: np.transpose(grad, (0, 3, 1, 2))
        direction_np = np.transpose(
            np.asarray(reference["direction_nchw"], dtype=np.float32).reshape(nchw.shape), (0, 2, 3, 1)).copy()
    else:
        raise RuntimeError(f"Unexpected public input shape: {shape}")
    if spec.dtype != tf.float32:
        raise RuntimeError(f"Unexpected public dtype: {spec.dtype}")
    target = tf.constant(np.asarray(reference["target_unit"], dtype=np.float32))

    def raw_for(array: tf.Tensor) -> tf.Tensor:
        outputs = fn(**{input_name: array})
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one embedding output, got {list(outputs)}")
        value = next(iter(outputs.values()))
        if value.shape.num_elements() != 128:
            raise RuntimeError(f"Wrong embedding shape: {value.shape}")
        return tf.reshape(value, [128])

    x = tf.Variable(input_np)
    with tf.GradientTape() as tape:
        raw_tensor = raw_for(x)
        loss = tf.tensordot(raw_tensor, target, axes=1)
    gradient_tensor = tape.gradient(loss, x)
    if gradient_tensor is None:
        raise RuntimeError("SavedModel input gradient is absent")
    raw = raw_tensor.numpy()
    grad = gradient_to_nchw(gradient_tensor.numpy())
    repeat = raw_for(tf.constant(input_np)).numpy()
    cv_raw = np.asarray(reference["raw_opencv"], dtype=np.float32)
    torch_raw = np.asarray(reference["raw_onnx2torch"], dtype=np.float32)
    torch_grad = np.asarray(reference["gradient_nchw"], dtype=np.float32).reshape(nchw.shape)
    direction = np.asarray(reference["direction_nchw"], dtype=np.float32).reshape(nchw.shape)
    if not all(np.isfinite(a).all() for a in (raw, grad, repeat)) or not np.any(grad):
        raise RuntimeError("Nonfinite or zero converted result")
    analytic = float(np.sum(grad * direction))
    finite_differences = []
    for step in (0.5, 1.0, 2.0):
        plus = raw_for(tf.constant(input_np + step * direction_np))
        minus = raw_for(tf.constant(input_np - step * direction_np))
        estimate = float((tf.tensordot(plus, target, axes=1) - tf.tensordot(minus, target, axes=1)).numpy() / (2 * step))
        error = abs(estimate - analytic)
        passed = error <= 1e-5 or error <= 0.25 * max(abs(estimate), abs(analytic))
        finite_differences.append({"step": step, "estimate": estimate, "error": error, "passed": passed})
    metrics = {
        "raw_max_vs_opencv": float(np.max(np.abs(raw - cv_raw))),
        "raw_max_vs_onnx2torch": float(np.max(np.abs(raw - torch_raw))),
        "raw_cosine_vs_opencv": cosine(raw, cv_raw),
        "gradient_cosine_vs_onnx2torch": cosine(grad, torch_grad),
        "gradient_l2_ratio_vs_onnx2torch": float(np.linalg.norm(grad) / np.linalg.norm(torch_grad)),
        "repeat_raw_max_abs": float(np.max(np.abs(raw - repeat))),
        "directional_derivative": analytic,
    }
    checks = {
        "raw_max": metrics["raw_max_vs_opencv"] <= 1e-3,
        "raw_cosine": metrics["raw_cosine_vs_opencv"] >= .99999,
        "gradient_cosine": metrics["gradient_cosine_vs_onnx2torch"] >= .99,
        "dropout_inference_repeat": metrics["repeat_raw_max_abs"] <= 1e-6,
        "finite_difference": sum(row["passed"] for row in finite_differences) >= 2,
    }
    report = {
        "schema_version": 1,
        "purpose": "native_saved_model_synthetic_parity_before_tfjs",
        "saved_model_pb_sha256": hashlib.sha256((args.saved_model / "saved_model.pb").read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "tensorflow_version": tf.__version__, "signature": input_name, "layout": layout,
        "public_input_shape": shape, "public_output_names": list(fn.structured_outputs),
        "limits": {"raw_max": 1e-3, "raw_cosine_min": .99999, "gradient_cosine_min": .99,
                   "dropout_repeat_max": 1e-6, "fd_pass_min": 2, "fd_abs": 1e-5, "fd_relative": .25},
        "metrics": metrics, "finite_differences": finite_differences,
        "checks": checks, "pass": all(checks.values()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
