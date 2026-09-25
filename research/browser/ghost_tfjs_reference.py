"""Freeze synthetic GhostFaceNet float32 input-gradient parity data.

Use the isolated legacy-Keras converter venv. No photographs are read.
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


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clone-h5", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.clone_h5.exists():
        parser.error("Refusing to overwrite a frozen reference or clone")
    author = tf.keras.models.load_model(str(args.weights), compile=False)
    config = json.loads(author.to_json())
    for layer in config["config"]["layers"]:
        layer["config"]["dtype"] = "float32"
    clone = tf.keras.models.model_from_json(json.dumps(config))
    clone.set_weights(author.get_weights())
    assert all(layer.dtype_policy.name == "float32" for layer in clone.layers)

    rgb = (np.arange(112 * 112 * 3, dtype=np.uint32) % 256).astype(np.uint8)
    x_np = ((rgb.astype(np.float32).reshape(1, 112, 112, 3) - 127.5) / 128).astype(np.float32)
    target_np = (np.sin(np.arange(512) * 0.17) + np.cos(np.arange(512) * 0.11)).astype(np.float32)
    target_np /= np.linalg.norm(target_np)
    target = tf.constant(target_np.reshape(1, 512))
    x = tf.Variable(x_np)
    with tf.GradientTape() as tape:
        raw = tf.cast(clone(x, training=False), tf.float32)
        unit = tf.math.l2_normalize(raw, axis=1)
        loss = 1.0 - tf.reduce_sum(unit * target)
    grad_np = tape.gradient(loss, x).numpy().reshape(-1)
    raw_np = raw.numpy().reshape(-1)
    unit_np = unit.numpy().reshape(-1)
    author_np = tf.cast(author(x_np, training=False), tf.float32).numpy().reshape(-1)
    if not all(np.isfinite(v).all() for v in (grad_np, raw_np, unit_np, author_np)):
        raise RuntimeError("Nonfinite synthetic parity output")
    if float(np.linalg.norm(grad_np)) == 0:
        raise RuntimeError("Zero input gradient")

    # Finite differences at the five strongest coordinates. Criterion is
    # predeclared: at least three must meet abs <= 1e-5 or relative <= 0.25.
    epsilon = 0.01
    indices = np.argsort(-np.abs(grad_np), kind="stable")[:5]
    checks = []
    for index in indices:
        plus = x_np.copy().reshape(-1)
        minus = x_np.copy().reshape(-1)
        plus[index] += epsilon
        minus[index] -= epsilon
        def scalar(flat: np.ndarray) -> float:
            out = clone(flat.reshape(1, 112, 112, 3), training=False)
            return float(1 - tf.reduce_sum(tf.math.l2_normalize(out, axis=1) * target).numpy())
        fd = (scalar(plus) - scalar(minus)) / (2 * epsilon)
        actual = float(grad_np[index])
        checks.append({"index": int(index), "analytic": actual, "finite_difference": fd,
                       "passed": abs(actual - fd) <= 1e-5 or abs(actual - fd) <= 0.25 * max(abs(actual), abs(fd))})

    clone.save(str(args.clone_h5), include_optimizer=False)
    result = {
        "schema_version": 1,
        "purpose": "synthetic_rgb112_exact_weight_float32_clone_gradient",
        "weights_sha256": sha(args.weights),
        "clone_h5_sha256": sha(args.clone_h5),
        "tensorflow_version": tf.__version__,
        "input_shape": [1, 112, 112, 3],
        "normalization": "(RGB_uint8_float32 - 127.5) / 128.0",
        "crop_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "normalized_tensor_sha256": hashlib.sha256(x_np.tobytes()).hexdigest(),
        "target": target_np.tolist(),
        "float32_raw_output": raw_np.tolist(),
        "float32_unit_output": unit_np.tolist(),
        "author_raw_output": author_np.tolist(),
        "author_unit_output": (author_np / np.linalg.norm(author_np)).tolist(),
        "float32_loss": float(loss.numpy()),
        "float32_input_gradient": grad_np.tolist(),
        "finite_difference": {"epsilon": epsilon, "checks": checks,
                              "passed": sum(item["passed"] for item in checks) >= 3},
        "author_vs_clone_unit_cosine": float(np.dot(author_np, raw_np) / (np.linalg.norm(author_np) * np.linalg.norm(raw_np))),
        "author_vs_clone_max_unit_difference": float(np.max(np.abs(author_np / np.linalg.norm(author_np) - unit_np))),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("clone_h5_sha256", "float32_loss", "author_vs_clone_unit_cosine", "author_vs_clone_max_unit_difference", "finite_difference")}, indent=2))


if __name__ == "__main__":
    run()
