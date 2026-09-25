"""Bounded 0095 random-input finite-difference diagnosis; no photographs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import tensorflow as tf  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(raw: np.ndarray, target: np.ndarray) -> float:
    vec = raw.reshape(-1).astype(np.float64)
    return float(np.dot(vec / np.linalg.norm(vec), target.astype(np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "saved_model", "synthetic", "synthetic_gradients", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    expected = {
        "onnx": "08c79d93c175877bb1f6591b9a72d4971a4c82fa03640589f762393d2d383963",
        "synthetic": "94f95e7603b638b63ed65bdbc6984acfc46a846f67ce06416fbc4112816fa8f2",
        "synthetic_gradients": "d1d59597bd7a0e515be13712bd76f0bff6405d277421c09ac3e29f8f3bda0255",
    }
    for key, digest in expected.items():
        if sha(getattr(args, key)) != digest:
            raise ValueError(f"Frozen {key} hash differs")
    if sha(args.saved_model / "saved_model.pb") != "5f1e86f68f811a0b482058ad204cc9bcb128ef4f569c152d3ee31dfa1e77ad6b":
        raise ValueError("Frozen SavedModel hash differs")

    with np.load(args.synthetic) as syn, np.load(args.synthetic_gradients) as sg:
        source = syn["random_input"].copy()
        torch_gradient = sg["random_gradient"].copy()
        target = sg["target_unit"].copy()

    loaded = tf.saved_model.load(str(args.saved_model))
    function = loaded.signatures["serving_default"]
    name = next(iter(function.structured_input_signature[1]))
    variable = tf.Variable(np.ascontiguousarray(source.transpose(0, 2, 3, 1)))
    with tf.GradientTape() as tape:
        raw = tf.reshape(next(iter(function(**{name: variable}).values())), [256])
        objective = tf.tensordot(tf.math.l2_normalize(raw), tf.constant(target), axes=1)
    analytic = tape.gradient(objective, variable)
    if analytic is None:
        raise ValueError("TF gradient disconnected")
    tf_gradient = np.ascontiguousarray(analytic.numpy().transpose(0, 3, 1, 2))

    # Preselect by gradient discrepancy, once, before evaluating any finite differences.
    difference = np.abs(tf_gradient - torch_gradient)
    difference[:, :, (0, 127), :] = -1
    difference[:, :, :, (0, 127)] = -1
    flat = difference.reshape(-1)
    order = np.argsort(-flat, kind="stable")[:3]
    coordinates = [tuple(int(x) for x in np.unravel_index(int(index), difference.shape))
                   for index in order]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])
    ort_input = session.get_inputs()[0].name
    ort_output = session.get_outputs()[0].name

    def ort_value(data: np.ndarray) -> float:
        return scalar(session.run([ort_output], {ort_input: data})[0], target)

    def tf_value(data: np.ndarray) -> float:
        tensor = tf.convert_to_tensor(np.ascontiguousarray(data.transpose(0, 2, 3, 1)))
        return scalar(next(iter(function(**{name: tensor}).values())).numpy(), target)

    rows = []
    for coord in coordinates:
        row = {"coordinate_nchw": coord,
               "torch_analytic": float(torch_gradient[coord]),
               "tf_analytic": float(tf_gradient[coord]),
               "absolute_analytic_difference": float(difference[coord]),
               "finite_differences": {}}
        for epsilon in (0.01, 0.1):
            plus = source.copy()
            minus = source.copy()
            plus[coord] += np.float32(epsilon)
            minus[coord] -= np.float32(epsilon)
            actual_step = float(plus[coord] - minus[coord])
            ort_fd = (ort_value(plus) - ort_value(minus)) / actual_step
            tf_fd = (tf_value(plus) - tf_value(minus)) / actual_step
            row["finite_differences"][str(epsilon)] = {
                "actual_step": actual_step,
                "ort": ort_fd,
                "tf": tf_fd,
                "ort_minus_torch_analytic": ort_fd - row["torch_analytic"],
                "tf_minus_tf_analytic": tf_fd - row["tf_analytic"],
            }
        rows.append(row)
    result = {"purpose": "fixed top-three random-case interior finite-difference diagnosis",
              "source_sha256": sha(Path(__file__)),
              "source_hashes": expected,
              "saved_model_pb_sha256": sha(args.saved_model / "saved_model.pb"),
              "tensorflow_version": tf.__version__,
              "onnxruntime_version": ort.__version__,
              "backend": "TF_ENABLE_ONEDNN_OPTS=" + os.environ.get("TF_ENABLE_ONEDNN_OPTS", "unset"),
              "objective": "dot(l2_normalize(raw256), fixed seed7095 target_unit)",
              "epsilons": [0.01, 0.1],
              "rows": rows}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
