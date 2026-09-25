"""Build a float32 inference clone without training-only regularizers.

The author H5 is read-only. This uses standard legacy-Keras serialization and
the official converter afterwards; it does not modify TF.js JSON by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient(model: tf.keras.Model, x_np: np.ndarray, target: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:
    x = tf.Variable(x_np)
    with tf.GradientTape() as tape:
        raw = tf.cast(model(x, training=False), tf.float32)
        unit = tf.math.l2_normalize(raw, axis=1)
        loss = 1 - tf.reduce_sum(unit * target)
    return raw.numpy().reshape(-1), tape.gradient(loss, x).numpy().reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("Refusing to overwrite existing clone or report")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    author = tf.keras.models.load_model(str(args.weights), compile=False)
    base = json.loads(author.to_json())
    for layer in base["config"]["layers"]:
        layer["config"]["dtype"] = "float32"
    exact = tf.keras.models.model_from_json(json.dumps(base))
    exact.set_weights(author.get_weights())
    stripped = json.loads(json.dumps(base))
    removed = []
    for layer in stripped["config"]["layers"]:
        for key, value in list(layer["config"].items()):
            if key.endswith("_regularizer") and value is not None:
                removed.append({"layer": layer["name"], "field": key, "class_name": value.get("class_name")})
                layer["config"][key] = None
    if not removed:
        raise RuntimeError("No serialized training regularizers found; expected L2")
    clone = tf.keras.models.model_from_json(json.dumps(stripped))
    clone.set_weights(exact.get_weights())
    if any(a.shape != b.shape or not np.array_equal(a, b) for a, b in zip(exact.get_weights(), clone.get_weights())):
        raise RuntimeError("Clone weights differ")
    rgb = (np.arange(112 * 112 * 3, dtype=np.uint32) % 256).astype(np.uint8)
    x_np = ((rgb.astype(np.float32).reshape(1, 112, 112, 3) - 127.5) / 128).astype(np.float32)
    target = tf.constant(np.asarray(reference["target"], dtype=np.float32).reshape(1, 512))
    raw_exact, grad_exact = gradient(exact, x_np, target)
    raw_clone, grad_clone = gradient(clone, x_np, target)
    raw_frozen = np.asarray(reference["float32_raw_output"], dtype=np.float32)
    grad_frozen = np.asarray(reference["float32_input_gradient"], dtype=np.float32)
    comparisons = {
        "raw_exact_vs_no_regularizer_max_abs": float(np.max(np.abs(raw_exact - raw_clone))),
        "gradient_exact_vs_no_regularizer_max_abs": float(np.max(np.abs(grad_exact - grad_clone))),
        "raw_exact_vs_frozen_max_abs": float(np.max(np.abs(raw_exact - raw_frozen))),
        "gradient_exact_vs_frozen_max_abs": float(np.max(np.abs(grad_exact - grad_frozen))),
    }
    if not all(np.isfinite(array).all() for array in (raw_exact, grad_exact, raw_clone, grad_clone)):
        raise RuntimeError("Nonfinite output or gradient")
    if any(value > 1e-6 for value in comparisons.values()):
        raise RuntimeError(f"Inference clone does not meet predeclared 1e-6 parity: {comparisons}")
    clone.save(str(args.output), include_optimizer=False)
    report = {
        "schema_version": 1,
        "purpose": "standard_keras_inference_clone_no_training_regularizers",
        "author_h5_sha256": sha(args.weights),
        "frozen_float32_reference_sha256": sha(args.reference),
        "clone_h5_sha256": sha(args.output),
        "tensorflow_version": tf.__version__,
        "removed_training_only_fields": removed,
        "removed_count": len(removed),
        "comparison_tolerance_max_abs": 1e-6,
        "comparisons": comparisons,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"removed_count": len(removed), "comparisons": comparisons, "clone_h5_sha256": report["clone_h5_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
