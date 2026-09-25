"""Synthetic-only SFace forward/gradient diagnostic; never runs photo optimization.

Uses the cached official OpenCV SFace ONNX with its own normalization. The
model input is RGB NCHW float32 in [0, 255], without external normalization.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from onnx2torch import convert


DEFAULT_MODEL = Path.home() / "Downloads" / "Face Privacy Filter" / "cache" / ".deepface" / "weights" / "face_recognition_sface_2021dec.onnx"
DEFAULT_OUTPUT = Path.home() / "Downloads" / "FCKFACE-data" / "runs" / "sface-gradient-parity-v1" / "results.json"


def rss_bytes() -> int | None:
    """Return current Windows working set without an additional dependency."""
    if not hasattr(ctypes, "windll"):
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    ok = get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    return int(counters.WorkingSetSize) if ok else None


def unit(array: np.ndarray) -> np.ndarray:
    return array / np.linalg.norm(array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model_path = args.model.resolve(strict=True)
    with model_path.open("rb") as model_file:
        sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    rng = np.random.default_rng(0)
    rgb_u8 = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    rgb_nchw = np.ascontiguousarray(rgb_u8.transpose(2, 0, 1)[None].astype(np.float32))
    bgr_u8 = np.ascontiguousarray(rgb_u8[:, :, ::-1])
    expected = np.asarray(cv2.FaceRecognizerSF.create(str(model_path), "").feature(bgr_u8)).reshape(-1)

    rss_before = rss_bytes()
    started = time.perf_counter()
    model = convert(onnx.load(str(model_path))).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    conversion_seconds = time.perf_counter() - started
    rss_after_conversion = rss_bytes()

    x = torch.from_numpy(rgb_nchw.copy()).requires_grad_(True)
    started = time.perf_counter()
    raw = model(x).reshape(-1)
    forward_seconds = time.perf_counter() - started
    raw_np = raw.detach().numpy()
    target = torch.from_numpy(unit(rng.standard_normal(raw_np.shape).astype(np.float32)))
    loss = torch.dot(raw, target)
    started = time.perf_counter()
    gradient = torch.autograd.grad(loss, x)[0].detach().numpy()
    backward_seconds = time.perf_counter() - started
    direction = unit(rng.standard_normal(rgb_nchw.shape).astype(np.float32))
    analytic = float(np.sum(gradient * direction))
    finite_differences = []
    with torch.no_grad():
        for step in (0.25, 0.5, 1.0, 2.0, 4.0):
            plus = torch.from_numpy(np.ascontiguousarray(rgb_nchw + step * direction))
            minus = torch.from_numpy(np.ascontiguousarray(rgb_nchw - step * direction))
            plus_loss = float(torch.dot(model(plus).reshape(-1), target))
            minus_loss = float(torch.dot(model(minus).reshape(-1), target))
            estimate = (plus_loss - minus_loss) / (2 * step)
            finite_differences.append({"step_pixel_l2": step, "estimate": estimate, "absolute_error": abs(estimate - analytic)})
        repeat_raw = model(torch.from_numpy(rgb_nchw.copy())).reshape(-1).numpy()

    delta = raw_np - expected
    result = {
        "input": "seed=0 synthetic uint8 RGB 112x112; NCHW float32 0..255; OpenCV received BGR uint8 of the same pixels",
        "model_path": str(model_path),
        "model_sha256": sha256,
        "runtime": {"opencv": cv2.__version__, "torch": torch.__version__, "onnx": onnx.__version__, "cpu_threads": 2},
        "forward_parity": {
            "raw_feature_length": int(raw_np.size),
            "raw_max_abs_difference": float(np.max(np.abs(delta))),
            "raw_mean_abs_difference": float(np.mean(np.abs(delta))),
            "l2_feature_max_abs_difference": float(np.max(np.abs(unit(raw_np) - unit(expected)))),
            "cosine": float(np.dot(unit(raw_np.astype(np.float64)), unit(expected.astype(np.float64)))),
        },
        "gradient": {
            "loss": "dot(raw128_feature, deterministic unit random vector)",
            "loss_value": float(loss.detach()),
            "all_finite": bool(np.isfinite(gradient).all()),
            "l2_norm": float(np.linalg.norm(gradient)),
            "directional_derivative": analytic,
            "finite_differences": finite_differences,
            "repeat_forward_max_abs_difference": float(np.max(np.abs(repeat_raw - raw_np))),
        },
        "timing_seconds": {"conversion": conversion_seconds, "forward": forward_seconds, "backward": backward_seconds},
        "rss_bytes": {"before_conversion": rss_before, "after_conversion": rss_after_conversion, "end": rss_bytes()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
