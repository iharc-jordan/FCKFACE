"""One frozen, source-only ArcFace ONNX-to-Torch component parity check.

Run only after review of the private protocol. No detector, gallery, new photo,
optimization, or model redistribution is involved. Crop inputs are previously
reviewed SFace-aligned 112px images, not official ArcFace-aligned crops.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
from onnx2torch import convert
import onnxruntime as ort
from PIL import Image
import torch


MODEL_SHA = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
IDS = ("frll-024", "frll-036")
CASES = ("range", "flat", "random", *IDS)
SEED = 7095
EPS_RAW_PIXEL = 0.1
RAW_LIMIT = 1e-3
COS_LIMIT = 0.99999
REL_LIMIT = 0.1
ABS_LIMIT = 1e-6
NONTRIVIAL = 1e-6


def sha(path: Path) -> str:
    block = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            block.update(part)
    return block.hexdigest()


def peak_bytes() -> int | None:
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    record = Counters()
    record.cb = ctypes.sizeof(Counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(record), record.cb)
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(record.PeakWorkingSetSize)


def samples(protocol: dict, crop_manifest: dict, crops: Path) -> dict[str, np.ndarray]:
    ramp = np.arange(112 * 112 * 3, dtype=np.uint32).reshape(112, 112, 3) % 256
    images = {
        "range": ramp.astype(np.uint8),
        "flat": np.full((112, 112, 3), 128, np.uint8),
        "random": np.random.default_rng(0).integers(0, 256, (112, 112, 3), dtype=np.uint8),
    }
    for row in crop_manifest["cases"]:
        identity = row["identity"]
        if identity not in IDS:
            raise ValueError("Unexpected development identity in crop manifest")
        crop = row["crops"]["sface"]
        path = crops / crop["name"]
        if sha(path) != crop["png_sha256"] or sha(path) != protocol["crop_png_sha256"][identity]:
            raise ValueError(f"Reviewed crop changed: {identity}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (112, 112):
                raise ValueError(f"Wrong reviewed crop shape: {identity}")
            rgb = np.asarray(image.convert("RGB"), np.uint8)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != crop["rgb_sha256"]:
            raise ValueError(f"Reviewed RGB pixels changed: {identity}")
        images[identity] = rgb
    if tuple(images) != CASES:
        raise ValueError("Wrong five component inputs")
    return {name: np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32))
            for name, rgb in images.items()}


def normalized_input(raw_rgb: np.ndarray) -> np.ndarray:
    # Official ArcFaceONNX blobFromImages on BGR crop: swapRB=True, mean/std 127.5.
    return np.ascontiguousarray((raw_rgb - 127.5) / 127.5, dtype=np.float32)


def scalar(raw: np.ndarray, target: np.ndarray) -> float:
    vector = raw.astype(np.float64).reshape(-1)
    return float(np.dot(vector / np.linalg.norm(vector), target))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def postflight(args: argparse.Namespace, protocol: dict, crop_manifest: dict) -> dict:
    observed = {
        "script_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(args.protocol),
        "original_model_sha256": sha(args.original_model),
        "private_model_sha256": sha(args.private_model),
        "crop_manifest_sha256": sha(args.crop_manifest),
        "crop_png_sha256": {row["identity"]: sha(args.crops / row["crops"]["sface"]["name"])
                            for row in crop_manifest["cases"]},
    }
    observed["unchanged"] = (
        observed["script_sha256"] == protocol["script_sha256"]
        and observed["protocol_sha256"] == args.protocol_sha256
        and observed["original_model_sha256"] == MODEL_SHA
        and observed["private_model_sha256"] == MODEL_SHA
        and observed["crop_manifest_sha256"] == protocol["crop_manifest_sha256"]
        and observed["crop_png_sha256"] == protocol["crop_png_sha256"]
    )
    return observed


def run(args: argparse.Namespace) -> None:
    started = perf_counter()
    if args.output.exists() or args.protocol.parent != args.output.parent:
        raise ValueError("Use a fresh output directory and a separately frozen protocol")
    repo = Path(__file__).resolve().parent.parent
    if (args.output.resolve().is_relative_to(repo) or
            args.private_model.resolve().is_relative_to(repo) or
            args.protocol.resolve().is_relative_to(repo)):
        raise ValueError("Models and biometric outputs must stay outside the repository")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (protocol["purpose"] != "frozen_arcface_component_only"
            or protocol["cases"] != list(CASES)
            or protocol["script_sha256"] != sha(Path(__file__))
            or protocol["model_sha256"] != MODEL_SHA
            or protocol["limits"] != {"raw_max_abs": RAW_LIMIT, "unit_cosine_min": COS_LIMIT,
                                      "epsilon_raw_pixel": EPS_RAW_PIXEL,
                                      "relative_error_max": REL_LIMIT, "absolute_error_max": ABS_LIMIT,
                                      "nontrivial_gradient_min": NONTRIVIAL,
                                      "nontrivial_relative_passes_per_case": 2}):
        raise ValueError("Frozen component protocol or source changed")
    if (sha(args.original_model) != MODEL_SHA or sha(args.private_model) != MODEL_SHA
            or args.private_model.is_symlink() or args.private_model.resolve() == args.original_model.resolve()
            or sha(args.crop_manifest) != protocol["crop_manifest_sha256"]):
        raise ValueError("ArcFace model copy or crop record changed")
    crop_manifest = json.loads(args.crop_manifest.read_text(encoding="utf-8"))
    if (crop_manifest["purpose"] != "two_prior_development_identity_crops_only"
            or [row["identity"] for row in crop_manifest["cases"]] != list(IDS)
            or any(row["split"] != "development" or row["view"] != "neutral_front"
                   for row in crop_manifest["cases"])):
        raise ValueError("Wrong two reviewed development crops")
    images = samples(protocol, crop_manifest, args.crops)
    if {name: hashlib.sha256(image.tobytes()).hexdigest() for name, image in images.items()} != protocol["raw_rgb_nchw_sha256"]:
        raise ValueError("Five frozen RGB tensors changed")
    args.protocol_sha256 = sha(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "preflight.json").write_text(json.dumps({
        "protocol_sha256": sha(args.protocol), "script_sha256": sha(Path(__file__)),
        "model_sha256": MODEL_SHA, "crop_manifest_sha256": sha(args.crop_manifest),
        "cases": list(CASES), "peak_working_set_before_bytes": peak_bytes(),
    }, indent=2) + "\n", encoding="utf-8")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    try:
        onnx_model = onnx.load(str(args.private_model), load_external_data=False)
        if len(onnx_model.graph.input) != 1 or len(onnx_model.graph.output) != 1:
            raise ValueError("Wrong ArcFace network signature")
        t0 = perf_counter()
        model = convert(onnx_model).eval()
        conversion_seconds = perf_counter() - t0
        del onnx_model
        gc.collect()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(args.private_model), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ValueError("Unexpected ArcFace ONNX Runtime signature")
    except Exception as exc:
        write_json(args.output / "failure.json", {
            "status": "fail", "stage": "conversion_or_onnx_runtime_load",
            "error_type": type(exc).__name__, "error": str(exc),
            "postflight": postflight(args, protocol, crop_manifest),
            "elapsed_seconds": perf_counter() - started,
            "peak_working_set_bytes": peak_bytes(),
        })
        raise
    target = np.random.default_rng(SEED).standard_normal(512).astype(np.float32)
    target /= np.linalg.norm(target)
    target64 = torch.from_numpy(target.astype(np.float64))
    rows: dict[str, dict] = {}
    arrays = {"target_unit": target}
    failure = None
    for name, image in images.items():
        input_tensor = torch.from_numpy(image.copy()).requires_grad_(True)
        t0 = perf_counter()
        raw_tensor = model((input_tensor - 127.5) / 127.5)
        forward_seconds = perf_counter() - t0
        raw = raw_tensor.detach().cpu().numpy()
        expected = session.run(None, {session.get_inputs()[0].name: normalized_input(image)})[0]
        row = {"forward_seconds": forward_seconds, "stage": "forward"}
        rows[name] = row
        arrays[f"{name}_input_raw_rgb"] = image
        arrays[f"{name}_ort_raw"] = expected
        arrays[f"{name}_torch_raw"] = raw
        if (raw.shape != (1, 512) or expected.shape != raw.shape
                or not np.isfinite(raw).all() or not np.isfinite(expected).all()):
            row.update({"raw_shape": list(raw.shape), "ort_shape": list(expected.shape),
                        "torch_finite": bool(np.isfinite(raw).all()),
                        "ort_finite": bool(np.isfinite(expected).all())})
            failure = f"invalid_forward_output:{name}"
            break
        raw_max = float(np.max(np.abs(raw - expected)))
        a, b = raw.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if not np.isfinite(denominator) or denominator <= 0:
            row.update({"raw_max_abs_vs_ort": raw_max, "unit_cosine_vs_ort": None})
            failure = f"invalid_forward_norm:{name}"
            break
        cosine = float(np.dot(a, b) / denominator)
        row.update({"raw_max_abs_vs_ort": raw_max, "unit_cosine_vs_ort": cosine})
        if not np.isfinite(cosine) or raw_max > RAW_LIMIT or cosine < COS_LIMIT:
            failure = f"forward_parity:{name}"
            break
        vector = raw_tensor.reshape(-1).double()
        objective = torch.dot(vector / torch.linalg.vector_norm(vector), target64)
        t0 = perf_counter()
        grad = torch.autograd.grad(objective, input_tensor)[0].detach().cpu().numpy()
        gradient_seconds = perf_counter() - t0
        row.update({"stage": "gradient", "gradient_seconds": gradient_seconds,
                    "gradient_l2": float(np.linalg.norm(grad)) if np.isfinite(grad).all() else None,
                    "gradient_finite": bool(np.isfinite(grad).all())})
        arrays[f"{name}_input_gradient_raw_rgb"] = grad
        if not np.isfinite(grad).all() or float(np.linalg.norm(grad)) <= 0:
            failure = f"invalid_gradient:{name}"
            break
        admissible = (image >= EPS_RAW_PIXEL) & (image <= 255 - EPS_RAW_PIXEL)
        admissible[:, :, 0, :] = False
        admissible[:, :, -1, :] = False
        admissible[:, :, :, 0] = False
        admissible[:, :, :, -1] = False
        admissible = admissible.reshape(-1)
        indices = np.argsort(np.where(admissible, -np.abs(grad.reshape(-1)), np.inf), kind="stable")[:3]
        if len(indices) != 3 or not np.all(admissible[indices]):
            failure = f"insufficient_interior_fd_pixels:{name}"
            break
        points = []
        for index in indices:
            plus, minus = image.copy(), image.copy()
            plus.flat[int(index)] += EPS_RAW_PIXEL
            minus.flat[int(index)] -= EPS_RAW_PIXEL
            delta = float(plus.flat[int(index)] - minus.flat[int(index)])
            plus_raw = session.run(None, {session.get_inputs()[0].name: normalized_input(plus)})[0]
            minus_raw = session.run(None, {session.get_inputs()[0].name: normalized_input(minus)})[0]
            finite = (scalar(plus_raw, target) - scalar(minus_raw, target)) / delta
            analytic = float(grad.flat[int(index)])
            if not np.isfinite(finite):
                points.append({"nchw": list(map(int, np.unravel_index(int(index), image.shape))),
                               "analytic": analytic, "finite_difference": None,
                               "pass_relative": False, "pass_absolute_or_relative": False})
                break
            absolute = abs(finite - analytic)
            relative = absolute / max(abs(finite), abs(analytic), 1e-12)
            nontrivial = abs(analytic) >= NONTRIVIAL
            points.append({"nchw": list(map(int, np.unravel_index(int(index), image.shape))),
                           "analytic": analytic, "finite_difference": finite,
                           "absolute_error": absolute, "relative_error": relative,
                           "nontrivial": nontrivial,
                           "pass_relative": nontrivial and relative <= REL_LIMIT,
                           "pass_absolute_or_relative": relative <= REL_LIMIT or absolute <= ABS_LIMIT})
        success = sum(point["pass_relative"] for point in points) >= 2 and all(point["pass_absolute_or_relative"] for point in points)
        row.update({"stage": "finite_difference", "coordinates": points,
                    "gradient_pass": success})
        if not success:
            failure = f"finite_difference_parity:{name}"
            break
        print(f"ArcFace component {name} pass", flush=True)
    np.savez(args.output / "component-arrays.npz", **arrays)
    after = postflight(args, protocol, crop_manifest)
    if not after["unchanged"]:
        failure = failure or "postflight_frozen_input_changed"
    report = {"purpose": "arcface_onnx2torch_component_only",
              "status": "pass" if failure is None and len(rows) == len(CASES) else "fail",
              "failure": failure, "completed_cases": [name for name, row in rows.items()
                                                   if row.get("gradient_pass") is True],
              "attempted_cases": list(rows),
              "protocol_sha256": args.protocol_sha256, "postflight": after,
              "conversion_seconds": conversion_seconds, "elapsed_seconds": perf_counter() - started,
              "peak_working_set_bytes": peak_bytes(), "onnx2torch_version": "1.5.15",
              "torch_version": torch.__version__, "onnxruntime_version": ort.__version__,
              "cases": rows, "component_arrays_sha256": sha(args.output / "component-arrays.npz")}
    write_json(args.output / "results.json", report)
    if report["status"] != "pass":
        raise ValueError(f"ArcFace component failed: {failure}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "original-model", "private-model", "crop-manifest", "crops", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
