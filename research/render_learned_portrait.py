"""Eight fixed author Informative Drawings face previews; no gallery scoring.

Uses the authors' pinned Generator and two released checkpoints as prior-art
appearance controls. Images, checkpoints, and outputs remain outside Git.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image

from fckface_lab.datasets import grouped_images
from fckface_lab.imaging import decode_image, export_jpeg
from fckface_lab.patterns import actual_distortion, face_mask
import render_stencil as stencil


IDS = stencil.IDS
MODELS = ("model.pth", "model2.pth")
AUTHOR_GITHUB_COMMIT = "2349aee4daf7cb01d8de645b0bbb4f4392fd1395"
AUTHOR_SPACE_COMMIT = "bd4b4299be505803e036203a39c02024b4cfee11"
AUTHOR_FILES_SHA256 = {
    "source/model.py": "e2ce57dea009804343e3a4744a604052b113721bd4770378d75e63b8263cf313",
    "source/LICENSE": "865d995c72673df6f3e6e1d135b7c412ca9d8793d34d5577e6089f8994971f6d",
    "author-space-source/app.py": "4e989aff7a9f4491e9bf463dff0b7c35f84b52b34a0bf4ca918b525fa7d593d0",
    "author-space-weights/model.pth": "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
    "author-space-weights/model2.pth": "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
}
INPUT_SIDE = 512
CONTEXT_PER_SIDE = .20
BOUNDARY_RGB = (0x15, 0x1C, 0x64)
BOUNDARY_SOURCE_PX = 2.0


def process_rss_bytes() -> tuple[int, int]:
    """Current and peak Windows working set for this process, no dependency."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    if sys.platform != "win32":
        return 0, 0
    info = Counters()
    info.cb = ctypes.sizeof(info)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                     ctypes.byref(info), info.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(info.WorkingSetSize), int(info.PeakWorkingSetSize)


def square_crop(box, width: int, height: int):
    x0, y0, x1, y1 = map(float, box)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        raise ValueError("Invalid source box")
    side = int(np.ceil(max(bw, bh) * (1 + 2 * CONTEXT_PER_SIDE)))
    if side > min(width, height):
        raise ValueError("Requested context crop exceeds source dimensions")
    left = int(np.rint((x0 + x1 - side) / 2))
    top = int(np.rint((y0 + y1 - side) / 2))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    if not (left <= x0 and left + side >= x1 and top <= y0 and top + side >= y1):
        raise ValueError("Context crop lost selected face")
    return left, top, left + side, top + side


def composite(source: np.ndarray, drawing: np.ndarray, box, crop):
    """Place author grayscale output in the face ellipse without moving pixels."""
    if drawing.dtype != np.uint8 or drawing.shape != (INPUT_SIDE, INPUT_SIDE):
        raise ValueError("Unexpected author model output")
    left, top, right, bottom = crop
    side = right - left
    restored = np.asarray(Image.fromarray(drawing, "L").resize(
        (side, side), Image.Resampling.BICUBIC), dtype=np.uint8)
    support = face_mask((side, side, 3),
                        (box[0] - left, box[1] - top,
                         box[2] - box[0], box[3] - box[1])) > 0
    import cv2
    distance = cv2.distanceTransform(support.astype(np.uint8),
                                     cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    boundary = support & (distance <= BOUNDARY_SOURCE_PX)
    edited = source.copy()
    patch = edited[top:bottom, left:right]
    patch[support] = np.repeat(restored[:, :, None], 3, axis=2)[support]
    patch[boundary] = BOUNDARY_RGB
    full_support = np.zeros(source.shape[:2], dtype=bool)
    full_support[top:bottom, left:right] = support
    if np.any(edited[~full_support] != source[~full_support]):
        raise RuntimeError("Pixels outside selected face changed")
    return edited, full_support, {
        "face_support_pixels": int(np.count_nonzero(support)),
        "face_support_over_bbox_area": float(np.count_nonzero(support) /
                                              ((box[2] - box[0]) * (box[3] - box[1]))),
        "grayscale_min": int(drawing.min()), "grayscale_max": int(drawing.max()),
        "grayscale_mean": float(drawing.mean()),
        "grayscale_histogram_16_bins": np.histogram(drawing, bins=np.arange(0, 257, 16))[0].tolist(),
        "raw_distortion": actual_distortion(source, edited, stencil.xyxy_to_xywh(box)),
        "outside_support_changed_pixels_pre_jpeg": 0,
        "edge_density_pre_jpeg": stencil.edge_density(edited, full_support),
        "source_edge_density_same_support": stencil.edge_density(source, full_support),
    }


def preflight(args):
    repo_root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if out == repo_root or out.is_relative_to(repo_root):
        raise ValueError("Private images must remain outside Git")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory not empty: {out}")
    preview_sha256 = stencil.digest(args.source_preview)
    expected = args.expected_source_preview_sha256
    if expected is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Expected source preview SHA-256 must be lowercase hex")
        if preview_sha256 != expected:
            raise ValueError("Source preview SHA-256 differs from requested exact rerun")
    prior = json.loads(args.source_preview.read_text(encoding="utf-8"))
    if (prior.get("status") != "preview_only_not_scored" or
            prior.get("candidate_count") != 32 or
            not isinstance(prior.get("candidates"), list) or
            len(prior["candidates"]) != 32):
        raise ValueError("Unexpected source-only preview record")
    grouped = grouped_images(args.manifest, "development")
    ordered = sorted(grouped)
    if len(ordered) != 60 or (ordered[0], ordered[1], ordered[30], ordered[-1]) != IDS:
        raise ValueError("Development identity selection changed")
    source_rows = {}
    expected_arms = {(sigma, arm) for sigma in stencil.SIGMA_SPACE_112
                     for arm in stencil.ARMS}
    for identity in IDS:
        rows = [r for r in prior["candidates"] if r.get("identity") == identity]
        if len(rows) != 8 or {
                (r.get("preset_sigma_space_112"), r.get("arm")) for r in rows
        } != expected_arms:
            raise ValueError(f"Source-preview arm/preset grid changed: {identity}")
        row = rows[0]
        path = grouped[identity]["neutral_front"]
        source_sha256 = stencil.digest(path)
        if any(r.get("source_sha256") != source_sha256 for r in rows):
            raise ValueError(f"Source image bytes changed: {identity}")
        box = np.asarray(row.get("source_box_xyxy"), dtype=np.float64)
        with Image.open(path) as opened:
            source_width, source_height = opened.size
        if (box.shape != (4,) or not np.isfinite(box).all() or
                box[0] < 0 or box[1] < 0 or box[0] >= box[2] or box[1] >= box[3] or
                box[2] > source_width or box[3] > source_height or
                any(np.asarray(r.get("source_box_xyxy"), dtype=np.float64).shape != (4,)
                    or not np.array_equal(np.asarray(r["source_box_xyxy"], dtype=np.float64), box)
                    for r in rows)):
            raise ValueError(f"Source face box changed or invalid: {identity}")
        source_rows[identity] = (path, tuple(map(float, box)))
    if len({r.get("identity") for r in prior["candidates"]}) != len(IDS):
        raise ValueError("Unexpected source-preview identity")
    for name, expected_sha256 in AUTHOR_FILES_SHA256.items():
        if stencil.digest(args.author / name) != expected_sha256:
            raise ValueError(f"Pinned author file changed: {name}")
    model_path = args.author / "source" / "model.py"
    app_path = args.author / "author-space-source" / "app.py"
    return out, source_rows, preview_sha256, model_path, app_path


def generate(args):
    start = time.perf_counter()
    out, source_rows, preview_sha256, model_path, app_path = preflight(args)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_paths = {
        "render_learned_portrait.py": Path(__file__),
        "render_stencil.py": Path(__file__).parent / "render_stencil.py",
        "author-model.py": model_path,
        "author-app.py": app_path,
        "author-LICENSE": args.author / "source" / "LICENSE",
    }
    snapshot_hashes = {}
    for name, path in snapshot_paths.items():
        snapshot_hashes[name] = stencil.write_new(out / "source-snapshot" / name,
                                                  path.read_bytes())
    protocol = {
        "scope": "Informative Drawings prior-art appearance preview only; no recognition scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identities": IDS, "models": MODELS,
        "source_view": "neutral_front", "source_preview_sha256": preview_sha256,
        "manifest_sha256": stencil.digest(args.manifest),
        "source_images": {identity: {"path": str(path), "sha256": stencil.digest(path),
                                     "box_xyxy": box}
                          for identity, (path, box) in source_rows.items()},
        "crop": {"rule": "square centered on bbox, 20 percent of larger box side context on each side, clipped to image",
                 "input_side": INPUT_SIDE, "resample": "Pillow BICUBIC"},
        "model": "author Generator(3,1,3), torch.load(weights_only=True,map_location=cpu), eval/inference_mode",
        "input_transform": "author Space torchvision.transforms.ToTensor RGB, no normalization",
        "output": "author single-channel sigmoid grayscale, ToPILImage; bicubic crop restore; opaque within full selected face_mask>0; 2px navy inner boundary; JPEG Q95 4:4:4",
        "author_github_commit": AUTHOR_GITHUB_COMMIT,
        "author_space_commit": AUTHOR_SPACE_COMMIT,
        "author_file_sha256": AUTHOR_FILES_SHA256,
        "weights_sha256": {name: AUTHOR_FILES_SHA256[f"author-space-weights/{name}"]
                           for name in MODELS},
        "source_snapshot_sha256": snapshot_hashes,
        "threads": 2,
    }
    stencil.write_new(out / "protocol.json", (json.dumps(protocol, indent=2) + "\n").encode())

    # Import only the pinned author model definition. Importing the Space app
    # itself would immediately execute its Gradio server and load weights.
    sys.path.insert(0, str(model_path.parent))
    from model import Generator  # type: ignore  # author source outside Git
    import torch
    import torchvision.transforms as transforms
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    sources = {}
    sheets = {identity: {"full": [], "face": []} for identity in IDS}
    for identity, (path, box) in source_rows.items():
        pixels = decode_image(path)
        crop = square_crop(box, pixels.shape[1], pixels.shape[0])
        input_image = Image.fromarray(pixels).crop(crop).resize(
            (INPUT_SIDE, INPUT_SIDE), Image.Resampling.BICUBIC)
        input_path = out / identity / "clean" / "input512.png"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        if input_path.exists():
            raise FileExistsError(input_path)
        input_image.save(input_path)
        clean_jpeg = export_jpeg(pixels)
        clean = decode_image(clean_jpeg)
        full = out / identity / "clean" / "export.jpg"
        face = out / identity / "clean" / "face.png"
        stencil.write_new(full, clean_jpeg)
        stencil.write_new(face, stencil.crop_png(clean, box))
        sheets[identity]["full"].append(("clean", full))
        sheets[identity]["face"].append(("clean", face))
        sources[identity] = (pixels, clean, box, crop, input_image,
                             stencil.digest(input_path))

    records = []
    for model_name in MODELS:
        loaded_at = time.perf_counter()
        generator = Generator(3, 1, 3)
        checkpoint = args.author / "author-space-weights" / model_name
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        generator.load_state_dict(state)
        del state
        generator.eval()
        load_seconds = time.perf_counter() - loaded_at
        for identity in IDS:
            pixels, clean, box, crop, input_image, input_hash = sources[identity]
            infer_start = time.perf_counter()
            with torch.inference_mode():
                output = generator(to_tensor(input_image).unsqueeze(0))[0]
            drawing = np.asarray(to_image(output), dtype=np.uint8).copy()
            inference_seconds = time.perf_counter() - infer_start
            if drawing.ndim != 2:
                raise ValueError("Author Generator did not produce grayscale")
            edited, support, metrics = composite(pixels, drawing, box, crop)
            jpeg = export_jpeg(edited)
            decoded = decode_image(jpeg)
            folder = out / identity / model_name.removesuffix(".pth")
            output_path = folder / "export.jpg"
            crop_path = folder / "face.png"
            raw_path = folder / "author-output512.png"
            output_hash = stencil.write_new(output_path, jpeg)
            face_hash = stencil.write_new(crop_path, stencil.crop_png(decoded, box))
            drawing_image = Image.fromarray(drawing, "L")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if raw_path.exists():
                raise FileExistsError(raw_path)
            drawing_image.save(raw_path)
            current_rss, peak_rss = process_rss_bytes()
            metrics["post_jpeg_distortion"] = actual_distortion(
                clean, decoded, stencil.xyxy_to_xywh(box))
            metrics["edge_density_post_jpeg"] = stencil.edge_density(decoded, support)
            records.append({
                "identity": identity, "author_model": model_name,
                "source_crop_xyxy": crop, "source_input_512_sha256": input_hash,
                "author_output_512": str(raw_path),
                "author_output_512_sha256": stencil.digest(raw_path),
                "full_photo": str(output_path), "full_photo_sha256": output_hash,
                "face_crop": str(crop_path), "face_crop_sha256": face_hash,
                "model_load_seconds": round(load_seconds, 4),
                "inference_seconds": round(inference_seconds, 4),
                "process_rss_bytes": current_rss, "process_peak_rss_bytes": peak_rss,
                "metrics": metrics,
            })
            sheets[identity]["full"].append((model_name, output_path))
            sheets[identity]["face"].append((model_name, crop_path))
            print(f"Rendered {identity} {model_name} in {inference_seconds:.2f}s", flush=True)
        del generator
        gc.collect()
    sheet_records = {}
    for identity in IDS:
        full_path = out / f"{identity}-full-sheet.png"
        face_path = out / f"{identity}-face-sheet.png"
        stencil.contact_sheet(sheets[identity]["full"], full_path, True)
        stencil.contact_sheet(sheets[identity]["face"], face_path, False)
        sheet_records[identity] = {"full_photo": str(full_path),
                                   "full_photo_sha256": stencil.digest(full_path),
                                   "face_crop": str(face_path),
                                   "face_crop_sha256": stencil.digest(face_path)}
    report = {"status": "preview_only_not_scored", "candidate_count": len(records),
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.perf_counter() - start, 3),
              "protocol_sha256": stencil.digest(out / "protocol.json"),
              "candidates": records, "sheets": sheet_records}
    stencil.write_new(out / "previews.json", (json.dumps(report, indent=2) + "\n").encode())
    print(f"Informative Drawings previews complete: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-preview", "author", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-source-preview-sha256", type=str,
                        help="Optional digest for an exact rerun of a specific source preview")
    generate(parser.parse_args())
