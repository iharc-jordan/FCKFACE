"""Five-case native TensorFlow gradient parity for converted OpenVINO 0095.

Component test only: frozen synthetic tensors and two previously reviewed
128-pixel development crops. No detection, gallery, browser or photo export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

EXPECTED = {
    "synthetic": "94f95e7603b638b63ed65bdbc6984acfc46a846f67ce06416fbc4112816fa8f2",
    "synthetic_gradients": "d1d59597bd7a0e515be13712bd76f0bff6405d277421c09ac3e29f8f3bda0255",
    "crops": "b7e899bce9c9151b917674392a805360e9dac769a2e12aa2b12203c037cb6aa8",
    "crop_gradients": "16454b2e6d2e960893614b39892925bc8cebb91669b1a574969bd04ea46348fb",
}
LIMITS = {"raw_max_abs":1e-3,"unit_cosine_min":.99999,
          "gradient_cosine_min":.99999,"gradient_max_abs":1e-5}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    return float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("saved_model","synthetic","synthetic_gradients","crops",
                 "crop_gradients","protocol","output"):
        parser.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    args = parser.parse_args()
    if args.output.exists():parser.error("Output already exists")
    for name,expected in EXPECTED.items():
        if sha(getattr(args,name))!=expected:
            raise ValueError(f"Frozen {name} SHA differs")
    protocol=json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["limits"]!=LIMITS or protocol["source_hashes"]!=EXPECTED:
        raise ValueError("Predeclared gradient protocol differs")
    start=perf_counter()
    loaded=tf.saved_model.load(str(args.saved_model))
    load_seconds=perf_counter()-start
    function=loaded.signatures["serving_default"]
    _,specs=function.structured_input_signature
    if len(specs)!=1:raise ValueError("Expected one public input")
    input_name,spec=next(iter(specs.items()))
    if spec.dtype!=tf.float32 or spec.shape.as_list()!=[1,128,128,3]:
        raise ValueError(f"Unexpected SavedModel public input: {spec}")
    rows={}
    with np.load(args.synthetic) as syn, np.load(args.synthetic_gradients) as sg, \
         np.load(args.crops) as crops, np.load(args.crop_gradients) as cg:
        target=sg["target_unit"]
        if target.shape!=(256,) or not np.array_equal(target,cg["target_unit"]):
            raise ValueError("Seed-7095 unit target differs between frozen references")
        if abs(float(np.linalg.norm(target))-1)>1e-6:
            raise ValueError("Frozen target is not unit length")
        cases=[(name,syn[f"{name}_input"],syn[f"{name}_reference"],sg[f"{name}_gradient"])
               for name in ("range","flat","random")]
        cases += [(name,crops[f"{name}_input_f32"],crops[f"{name}_native_raw"],
                   cg[f"{name}_input_gradient"]) for name in ("frll-024","frll-036")]
        for name,source,expected_raw,expected_grad in cases:
            if (source.shape!=(1,3,128,128) or expected_raw.shape!=(1,256,1,1)
                    or expected_grad.shape!=source.shape or
                    any(x.dtype!=np.float32 for x in (source,expected_raw,expected_grad))):
                raise ValueError(f"Frozen shape/dtype differs: {name}")
            x=tf.Variable(np.ascontiguousarray(source.transpose(0,2,3,1)))
            t0=perf_counter()
            with tf.GradientTape() as tape:
                outputs=function(**{input_name:x})
                if len(outputs)!=1:raise ValueError("Expected one output")
                raw=tf.reshape(next(iter(outputs.values())),[256])
                scalar=tf.tensordot(tf.math.l2_normalize(raw),tf.constant(target),axes=1)
            grad=tape.gradient(scalar,x)
            seconds=perf_counter()-t0
            if grad is None:raise ValueError(f"Disconnected converted gradient: {name}")
            raw_np=raw.numpy()
            grad_np=np.ascontiguousarray(grad.numpy().transpose(0,3,1,2))
            finite=bool(np.isfinite(raw_np).all() and np.isfinite(grad_np).all())
            nonzero=bool(np.any(grad_np))
            raw_error=float(np.max(np.abs(raw_np-expected_raw.reshape(-1))))
            raw_cos=cosine(raw_np,expected_raw)
            grad_error=float(np.max(np.abs(grad_np-expected_grad)))
            grad_cos=cosine(grad_np,expected_grad) if nonzero else None
            reference_unit=expected_raw.reshape(-1).astype(np.float64)
            reference_unit/=np.linalg.norm(reference_unit)
            reference_scalar=float(np.dot(reference_unit,target.astype(np.float64)))
            row={"raw_max_abs":raw_error,"raw_unit_cosine":raw_cos,
                 "gradient_max_abs":grad_error,"gradient_cosine":grad_cos,
                 "gradient_l2":float(np.linalg.norm(grad_np)),
                 "scalar":float(scalar.numpy()),"reference_scalar":reference_scalar,
                 "scalar_abs_error":abs(float(scalar.numpy())-reference_scalar),
                 "seconds":seconds,"finite":finite,"gradient_nonzero":nonzero}
            row["pass"]=(finite and nonzero and raw_error<=LIMITS["raw_max_abs"]
                         and raw_cos>=LIMITS["unit_cosine_min"]
                         and grad_error<=LIMITS["gradient_max_abs"]
                         and grad_cos>=LIMITS["gradient_cosine_min"])
            rows[name]=row
    report={"purpose":"0095 native TF GradientTape vs frozen Torch component parity",
            "model_sha256":sha(args.saved_model/"saved_model.pb"),
            "protocol_sha256":sha(args.protocol),"reference_sha256":EXPECTED,
            "source_sha256":sha(Path(__file__)),"tensorflow_version":tf.__version__,
            "input_name":input_name,"input_layout":"BGR NHWC transpose only from BGR NCHW 0..255",
            "objective":"dot(l2_normalize(raw256), fixed seed7095 target_unit)",
            "limits":LIMITS,"model_load_seconds":load_seconds,"cases":rows,
            "pass":all(row["pass"] for row in rows.values())}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["pass"]:raise SystemExit(2)


if __name__=="__main__":main()
