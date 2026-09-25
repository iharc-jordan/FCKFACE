"""Freeze exact synthetic SFace OpenCV/onnx2torch forward and gradient arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from onnx2torch import convert


EXPECTED_MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"


def unit(array: np.ndarray) -> np.ndarray:
    return array / np.linalg.norm(array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite frozen reference")
    model_path = args.model.resolve(strict=True)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Unexpected SFace ONNX hash")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    rng = np.random.default_rng(0)
    rgb_u8 = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    nchw = np.ascontiguousarray(rgb_u8.transpose(2, 0, 1)[None].astype(np.float32))
    cv_raw = np.asarray(cv2.FaceRecognizerSF.create(str(model_path), "").feature(
        np.ascontiguousarray(rgb_u8[:, :, ::-1]))).reshape(-1)
    model = convert(onnx.load(str(model_path))).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_tensor = torch.from_numpy(nchw.copy()).requires_grad_(True)
    raw_tensor = model(input_tensor).reshape(-1)
    raw = raw_tensor.detach().numpy()
    target_np = unit(rng.standard_normal(raw.shape).astype(np.float32))
    target = torch.from_numpy(target_np)
    loss = torch.dot(raw_tensor, target)
    gradient = torch.autograd.grad(loss, input_tensor)[0].detach().numpy()
    direction = unit(rng.standard_normal(nchw.shape).astype(np.float32))
    analytic = float(np.sum(gradient * direction))
    fd = []
    with torch.no_grad():
        for step in (0.25, 0.5, 1.0, 2.0, 4.0):
            plus = torch.from_numpy(np.ascontiguousarray(nchw + step * direction))
            minus = torch.from_numpy(np.ascontiguousarray(nchw - step * direction))
            estimate = (float(torch.dot(model(plus).reshape(-1), target)) -
                        float(torch.dot(model(minus).reshape(-1), target))) / (2 * step)
            fd.append({"step": step, "estimate": estimate})
    if not all(np.isfinite(v).all() for v in (cv_raw, raw, gradient, target_np, direction)):
        raise RuntimeError("Nonfinite frozen reference")
    report = {
        "schema_version": 1,
        "purpose": "seed0_synthetic_rgb112_sface_native_forward_gradient",
        "model_sha256": digest,
        "input": "RGB uint8 seed0; NCHW float32 0..255; no external normalization",
        "rgb_u8_sha256": hashlib.sha256(rgb_u8.tobytes()).hexdigest(),
        "nchw_f32_sha256": hashlib.sha256(nchw.tobytes()).hexdigest(),
        "raw_opencv": cv_raw.tolist(),
        "raw_onnx2torch": raw.tolist(),
        "target_unit": target_np.tolist(),
        "gradient_nchw": gradient.reshape(-1).tolist(),
        "direction_nchw": direction.reshape(-1).tolist(),
        "directional_derivative": analytic,
        "finite_differences": fd,
        "raw_open_cv_max_difference": float(np.max(np.abs(raw - cv_raw))),
        "loss": float(loss.detach()),
        "opencv_version": cv2.__version__,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model_sha256", "rgb_u8_sha256", "nchw_f32_sha256", "raw_open_cv_max_difference", "loss", "directional_derivative")}, indent=2))


if __name__ == "__main__":
    main()
