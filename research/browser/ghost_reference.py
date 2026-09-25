"""Read-only GhostFaceNet artifact parity probe using the prior Python environment.

Run with that environment's Python and the cached author H5 path. Prints only
synthetic-input measurements; never reads or writes a photograph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# The author H5 was serialized with legacy Keras. Set before TensorFlow import.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from deepface.models.facial_recognition.GhostFaceNet import GhostFaceNetV1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    args = parser.parse_args()
    weights = args.weights.resolve(strict=True)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()

    serialized = tf.keras.models.load_model(str(weights), compile=False)
    rebuilt = GhostFaceNetV1()
    rebuilt.load_weights(str(weights))
    fp32_config = json.loads(serialized.to_json())
    for layer in fp32_config["config"]["layers"]:
        layer["config"]["dtype"] = "float32"
    fp32 = tf.keras.models.model_from_json(json.dumps(fp32_config))
    fp32.set_weights(serialized.get_weights())

    # RGB uint8 values in the author's 112x112 aligned-face input shape.
    rgb = (np.arange(112 * 112 * 3, dtype=np.uint32) % 256).astype(np.float32)
    normalized = ((rgb.reshape(1, 112, 112, 3) - 127.5) / 128.0).astype(np.float32)
    reference = serialized(normalized, training=False).numpy().reshape(-1)
    comparison = rebuilt(normalized, training=False).numpy().reshape(-1)
    fp32_reference = fp32(normalized, training=False).numpy().reshape(-1)
    if not all(np.isfinite(result).all() for result in (reference, comparison, fp32_reference)):
        raise RuntimeError("Non-finite synthetic embedding")

    print(json.dumps({
        "weights_sha256": digest,
        "model_shape": list(serialized.input_shape[1:]),
        "embedding_size": int(reference.size),
        "parameters": int(serialized.count_params()),
        "author_vs_deepface_max_abs_difference": float(np.max(np.abs(reference - comparison))),
        "author_vs_fp32_max_abs_difference": float(np.max(np.abs(reference - fp32_reference))),
        "author_vs_fp32_cosine": float(np.dot(reference, fp32_reference) / (np.linalg.norm(reference) * np.linalg.norm(fp32_reference))),
        "synthetic_embedding_first_eight": reference[:8].tolist(),
        "synthetic_embedding_l2_norm": float(np.linalg.norm(reference)),
        "fp32_embedding_first_eight": fp32_reference[:8].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
