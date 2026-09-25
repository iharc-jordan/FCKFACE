"""Check SavedModel cosine-loss gradient against the frozen ONNX reference."""

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
    a, b = a.astype(np.float64).reshape(-1), b.astype(np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite verification")
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    rgb = np.random.default_rng(0).integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    x_np = rgb[None].astype(np.float32)
    if hashlib.sha256(x_np.tobytes()).hexdigest() != reference["nhwc_f32_sha256"]:
        raise RuntimeError("Synthetic input hash mismatch")
    model = tf.saved_model.load(str(args.saved_model))
    fn = model.signatures["serving_default"]
    _, inputs = fn.structured_input_signature
    if len(inputs) != 1:
        raise RuntimeError("Wrong SavedModel input count")
    input_name, spec = next(iter(inputs.items()))
    if spec.shape.as_list() != [1, 112, 112, 3] or spec.dtype != tf.float32:
        raise RuntimeError("Wrong SavedModel input specification")
    target = tf.constant(reference["target_unit"], dtype=tf.float32)

    def output(x: tf.Tensor) -> tf.Tensor:
        result = fn(**{input_name: x})
        if len(result) != 1:
            raise RuntimeError("Wrong SavedModel output count")
        return tf.reshape(next(iter(result.values())), [128])

    def loss(x: tf.Tensor) -> tf.Tensor:
        raw = output(x)
        return 1 - tf.reduce_sum(tf.math.l2_normalize(raw) * target)

    x = tf.Variable(x_np)
    with tf.GradientTape() as tape:
        raw = output(x)
        objective = 1 - tf.reduce_sum(tf.math.l2_normalize(raw) * target)
    grad_tensor = tape.gradient(objective, x)
    if grad_tensor is None:
        raise RuntimeError("Converted input gradient absent")
    gradient = grad_tensor.numpy().reshape(-1)
    expected = np.asarray(reference["gradient_nhwc"], dtype=np.float32)
    raw_np = raw.numpy()
    expected_raw = np.asarray(reference["raw_opencv"], dtype=np.float32)
    if not np.isfinite(gradient).all() or not np.isfinite(raw_np).all() or not np.any(gradient):
        raise RuntimeError("Nonfinite or zero converted result")
    fd = []
    step = reference["finite_difference_step"]
    for check in reference["finite_difference_coordinates"]:
        index = check["index_nhwc"]
        plus, minus = x_np.copy().reshape(-1), x_np.copy().reshape(-1)
        plus[index] += step
        minus[index] -= step
        estimate = float((loss(tf.constant(plus.reshape(x_np.shape))) -
                          loss(tf.constant(minus.reshape(x_np.shape)))).numpy() / (2 * step))
        analytic = float(gradient[index])
        passed = abs(estimate - analytic) <= 1e-5 or abs(estimate - analytic) <= .25 * max(abs(estimate), abs(analytic))
        fd.append({"index_nhwc": index, "estimate": estimate, "analytic": analytic, "passed": passed})
    metrics = {"raw_max_vs_opencv": float(np.max(np.abs(raw_np - expected_raw))),
               "raw_cosine_vs_opencv": cosine(raw_np, expected_raw),
               "gradient_cosine_vs_onnx2torch": cosine(gradient, expected),
               "gradient_l2_ratio": float(np.linalg.norm(gradient) / np.linalg.norm(expected)),
               "loss": float(objective.numpy())}
    checks = {"raw_max": metrics["raw_max_vs_opencv"] <= 1e-3,
              "unit_cosine": metrics["raw_cosine_vs_opencv"] >= .99999,
              "gradient_cosine": metrics["gradient_cosine_vs_onnx2torch"] >= .99,
              "finite_difference": sum(row["passed"] for row in fd) >= 2}
    report = {"schema_version": 1, "purpose": "native_sface_cosine_loss_gradient_parity",
              "saved_model_pb_sha256": hashlib.sha256((args.saved_model / "saved_model.pb").read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
              "tensorflow_version": tf.__version__,
              "limits": {"raw_max": 1e-3, "unit_cosine_min": .99999, "gradient_cosine_min": .99,
                         "fd_abs": 1e-5, "fd_relative": .25, "fd_pass_min": 2},
              "metrics": metrics, "finite_differences": fd, "checks": checks, "pass": all(checks.values())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
