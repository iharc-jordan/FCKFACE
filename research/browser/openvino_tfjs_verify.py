"""Compare converted OpenVINO 0095 SavedModel with three frozen synthetic outputs.

Forward-only development feasibility check. No photographs or gradients.
The ONNX/OpenVINO public input is float32 BGR [1,3,128,128] in 0..255;
onnx2tf exposes the same channel order in NHWC. No RGB swap or rescaling.
"""
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
import tensorflow as tf  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    if sha(args.reference) != "94f95e7603b638b63ed65bdbc6984acfc46a846f67ce06416fbc4112816fa8f2":
        raise ValueError("Frozen OpenVINO synthetic reference changed")
    loaded = tf.saved_model.load(str(args.saved_model))
    if "serving_default" not in loaded.signatures:
        raise ValueError("Converted model has no serving_default")
    function = loaded.signatures["serving_default"]
    _, specs = function.structured_input_signature
    if len(specs) != 1:
        raise ValueError("Expected one public runtime input")
    input_name, spec = next(iter(specs.items()))
    if spec.dtype != tf.float32 or spec.shape.as_list() != [1,128,128,3]:
        raise ValueError(f"Unexpected input signature: {input_name} {spec}")
    rows = {}
    with np.load(args.reference) as ref:
        for name in ("range", "flat", "random"):
            source = ref[f"{name}_input"]
            expected = ref[f"{name}_reference"]
            if source.shape != (1,3,128,128) or expected.shape != (1,256,1,1):
                raise ValueError(f"Wrong frozen array shapes: {name}")
            if source.dtype != np.float32 or expected.dtype != np.float32:
                raise ValueError(f"Wrong frozen array dtype: {name}")
            bgr_nhwc = np.ascontiguousarray(source.transpose(0,2,3,1))
            outputs = function(**{input_name: tf.constant(bgr_nhwc)})
            if len(outputs) != 1:
                raise ValueError(f"Unexpected converted outputs: {list(outputs)}")
            actual = next(iter(outputs.values())).numpy()
            if actual.size != 256 or not np.isfinite(actual).all():
                raise ValueError(f"Nonfinite/wrong converted output: {name} {actual.shape}")
            raw_max = float(np.max(np.abs(actual.reshape(-1)-expected.reshape(-1))))
            cosine = unit_cosine(actual, expected)
            rows[name] = {"raw_max_abs":raw_max,"unit_cosine":cosine,
                          "pass":raw_max<=1e-3 and cosine>=.99999}
    report={"purpose":"OpenVINO0095 standard SavedModel synthetic forward parity only",
            "saved_model_pb_sha256":sha(args.saved_model/"saved_model.pb"),
            "reference_npz_sha256":sha(args.reference),
            "tensorflow_version":tf.__version__,"input_name":input_name,
            "input_layout":"BGR NHWC from frozen BGR NCHW transpose only",
            "output_names":list(function.structured_outputs),
            "limits":{"raw_max_abs":1e-3,"unit_cosine_min":.99999},
            "cases":rows,"pass":all(row["pass"] for row in rows.values())}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
