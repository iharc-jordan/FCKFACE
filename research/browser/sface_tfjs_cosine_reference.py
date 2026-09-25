"""Freeze a synthetic SFace cosine-loss input gradient for the browser probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx2torch import convert


MODEL_SHA = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
COORDINATES_NHWC = (0, (112 * 56 + 56) * 3 + 1, 112 * 112 * 3 - 1)
STEP = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite frozen reference")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA:
        raise RuntimeError("Wrong authoritative SFace model")
    source_bytes = args.source_reference.read_bytes()
    source = json.loads(source_bytes)
    if source["model_sha256"] != MODEL_SHA:
        raise RuntimeError("Wrong source reference")
    torch.set_num_threads(2)
    rgb = np.random.default_rng(0).integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    nchw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32))
    nhwc = nchw.transpose(0, 2, 3, 1).copy()
    if hashlib.sha256(nchw.tobytes()).hexdigest() != source["nchw_f32_sha256"]:
        raise RuntimeError("Synthetic input changed")
    model = convert(onnx.load(str(args.model))).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target = torch.tensor(source["target_unit"], dtype=torch.float32)
    x = torch.from_numpy(nchw.copy()).requires_grad_(True)
    raw = model(x).reshape(-1)
    loss = 1 - torch.dot(torch.nn.functional.normalize(raw, dim=0), target)
    gradient_nchw = torch.autograd.grad(loss, x)[0].detach().numpy()
    gradient_nhwc = gradient_nchw.transpose(0, 2, 3, 1).copy()
    checks = []
    with torch.no_grad():
        for index in COORDINATES_NHWC:
            altered = nhwc.reshape(-1).copy()
            altered[index] += STEP
            plus = torch.from_numpy(altered.reshape(nhwc.shape).transpose(0, 3, 1, 2).copy())
            altered[index] -= 2 * STEP
            minus = torch.from_numpy(altered.reshape(nhwc.shape).transpose(0, 3, 1, 2).copy())
            def value(tensor: torch.Tensor) -> float:
                output = model(tensor).reshape(-1)
                return float(1 - torch.dot(torch.nn.functional.normalize(output, dim=0), target))
            estimate = (value(plus) - value(minus)) / (2 * STEP)
            checks.append({"index_nhwc": index, "estimate": estimate,
                           "analytic": float(gradient_nhwc.reshape(-1)[index])})
    if not all(np.isfinite(a).all() for a in (raw.detach().numpy(), gradient_nhwc)):
        raise RuntimeError("Nonfinite reference")
    report = {
        "schema_version": 1,
        "purpose": "seed0_synthetic_sface_cosine_loss_browser_reference",
        "model_sha256": MODEL_SHA,
        "source_reference_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rgb_u8_sha256": source["rgb_u8_sha256"],
        "nhwc_f32_sha256": hashlib.sha256(nhwc.tobytes()).hexdigest(),
        "raw_onnx2torch": raw.detach().numpy().tolist(),
        "raw_opencv": source["raw_opencv"],
        "target_unit": source["target_unit"],
        "cosine_loss": float(loss.detach()),
        "gradient_nhwc": gradient_nhwc.reshape(-1).tolist(),
        "finite_difference_step": STEP,
        "finite_difference_coordinates": checks,
        "torch_version": torch.__version__,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cosine_loss": report["cosine_loss"], "gradient_l2": float(np.linalg.norm(gradient_nhwc)),
                      "finite_difference_coordinates": checks}, indent=2))


if __name__ == "__main__":
    main()
