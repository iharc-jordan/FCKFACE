"""Frozen H3c full-face graphic portrait previews; no recognition scoring.

The four development identities, two filter presets, four arms, and palette are
selected without reference-gallery access. Private images stay outside Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fckface_lab.datasets import grouped_images
from fckface_lab.imaging import decode_image, export_jpeg
from fckface_lab.patterns import actual_distortion, face_mask
from fckface_lab.recognition import SFaceModel
from reconstruct_graphics import similarity


IDS = ("frll-001", "frll-003", "frll-083", "frll-143")
SIGMA_SPACE_112 = (3, 5)
SIGMA_COLOR = 32
BILATERAL_DIAMETER = 9
ARMS = ("filtered_poster", "unfiltered_poster", "filtered_palette_continuous",
        "filtered_grayscale")
PALETTE_HEX = ("#151C64", "#5331A8", "#146FBD", "#71DCF0")
PALETTE = np.array([[0x15, 0x1C, 0x64], [0x53, 0x31, 0xA8],
                    [0x14, 0x6F, 0xBD], [0x71, 0xDC, 0xF0]], dtype=np.uint8)
BINS = (0, 64, 128, 192, 255)
CANONICAL_SIDE = 112
BOUNDARY_WIDTH_112 = 1.25


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def xyxy_to_xywh(box):
    x0, y0, x1, y1 = map(float, box)
    return x0, y0, x1 - x0, y1 - y0


def crop_png(image: np.ndarray, box) -> bytes:
    x0, y0, x1, y1 = box
    h, w = image.shape[:2]
    margin_x, margin_y = .14 * (x1 - x0), .14 * (y1 - y0)
    bounds = (max(0, int(x0 - margin_x)), max(0, int(y0 - margin_y)),
              min(w, int(x1 + margin_x)), min(h, int(y1 + margin_y)))
    stream = BytesIO()
    Image.fromarray(image[bounds[1]:bounds[3], bounds[0]:bounds[2]]).save(stream, "PNG")
    return stream.getvalue()


def edge_density(image: np.ndarray, support: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0
    return float(np.count_nonzero(edges & support) / max(1, np.count_nonzero(support)))


def render(source: np.ndarray, box, landmarks, sigma_space: int, arm: str):
    """Replace the selected face support with canonical 112-pixel artwork."""
    if sigma_space not in SIGMA_SPACE_112 or arm not in ARMS:
        raise ValueError("Unknown frozen H3c arm or preset")
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Expected RGB uint8 source")
    h, w = source.shape[:2]
    x0, y0, x1, y1 = map(float, box)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        raise ValueError("Invalid XYXY box")
    left = max(0, int(np.floor(x0 - .12 * bw)))
    top = max(0, int(np.floor(y0 - .12 * bh)))
    right = min(w, int(np.ceil(x1 + .12 * bw)))
    bottom = min(h, int(np.ceil(y1 + .12 * bh)))
    roi_box = (x0 - left, y0 - top, bw, bh)
    support = face_mask(source[top:bottom, left:right].shape, roi_box) > 0
    matrix, offset = similarity(np.asarray(landmarks, dtype=np.float64))
    forward = np.array([[matrix[0, 0], matrix[1, 0], offset[0]],
                        [matrix[0, 1], matrix[1, 1], offset[1]]], dtype=np.float32)
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    canonical = cv2.warpAffine(gray, forward, (CANONICAL_SIDE, CANONICAL_SIDE),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    filtered = cv2.bilateralFilter(canonical, BILATERAL_DIAMETER, SIGMA_COLOR,
                                    sigma_space)
    tone = canonical if arm == "unfiltered_poster" else filtered
    if arm in ("filtered_poster", "unfiltered_poster"):
        levels = np.clip(tone.astype(np.int16) // 64, 0, 3)
        art = PALETTE[levels]
    elif arm == "filtered_palette_continuous":
        # Continuous RGB interpolation across the same four fixed palette
        # anchors; no source colour or fine residual is mixed back in.
        channel = [np.interp(tone, (0, 85, 170, 255), PALETTE[:, index])
                   for index in range(3)]
        art = np.rint(np.stack(channel, axis=-1)).astype(np.uint8)
    else:
        art = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

    yy, xx = np.mgrid[top:bottom, left:right].astype(np.float32)
    u = xx * matrix[0, 0] + yy * matrix[1, 0] + offset[0]
    v = xx * matrix[0, 1] + yy * matrix[1, 1] + offset[1]
    mapped = cv2.remap(art, u.astype(np.float32), v.astype(np.float32),
                       interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_REPLICATE)
    scale = float(np.linalg.norm(matrix[:, 0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid similarity scale")
    boundary_source_pixels = max(1.0, BOUNDARY_WIDTH_112 / scale)
    inward_distance = cv2.distanceTransform(support.astype(np.uint8),
                                            cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    boundary = support & (inward_distance <= boundary_source_pixels)
    mapped[boundary] = PALETTE[0]
    edited = source.copy()
    patch = edited[top:bottom, left:right]
    patch[support] = mapped[support]
    # Check the full image because source geometry and off-face pixels are
    # part of this experiment's appearance contract.
    full_support = np.zeros((h, w), dtype=bool)
    full_support[top:bottom, left:right] = support
    if np.any(edited[~full_support] != source[~full_support]):
        raise RuntimeError("Changed pixels outside the frozen face support")
    histogram = {colour: int(np.count_nonzero(np.all(mapped[support] == rgb, axis=1)))
                 for colour, rgb in zip(PALETTE_HEX, PALETTE)}
    unique_rgb = int(len(np.unique(mapped[support].reshape(-1, 3), axis=0)))
    face_pixels = int(np.count_nonzero(support))
    return edited, full_support, {
        "face_support_pixels": face_pixels,
        "face_coverage_fraction": float(face_pixels / max(1, face_pixels)),
        "changed_pixels_pre_jpeg": int(np.count_nonzero(np.any(patch != source[top:bottom, left:right], axis=2) & support)),
        "outside_support_changed_pixels_pre_jpeg": 0,
        "canonical_similarity_scale": scale,
        "boundary_width_source_pixels": boundary_source_pixels,
        "palette_exact_rgb_histogram_pre_jpeg": histogram,
        "unique_rgb_colours_in_support_pre_jpeg": unique_rgb,
        "edge_density_pre_jpeg": edge_density(edited, full_support),
        "source_edge_density_same_support": edge_density(source, full_support),
        "raw_distortion": actual_distortion(source, edited, xyxy_to_xywh(box)),
    }


def contact_sheet(entries, destination: Path, full_photo: bool):
    cols = 3
    cell_w, cell_h, title_h = (260, 255, 30) if full_photo else (225, 245, 30)
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + title_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(entries):
        x = index % cols * cell_w
        y = index // cols * (cell_h + title_h)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (cell_w - image.width) // 2, y + title_h))
        draw.text((x + 5, y + 6), label, fill="black")
    sheet.save(destination)


def preflight(args):
    out = args.out.resolve()
    root = Path(__file__).resolve().parents[1]
    if out == root or out.is_relative_to(root):
        raise ValueError("Private previews must remain outside the source repository")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    groups = grouped_images(args.manifest, "development")
    ordered = sorted(groups)
    expected = (ordered[0], ordered[1], ordered[len(ordered) // 2], ordered[-1])
    if len(ordered) != 60 or expected != IDS:
        raise ValueError(f"Development selection changed: {expected}")
    for identity in IDS:
        if "neutral_front" not in groups[identity]:
            raise ValueError(f"Missing source view for {identity}")
    artifact = json.loads(args.calibration.read_text(encoding="utf-8"))
    if artifact.get("status") != "complete":
        raise ValueError("Calibration artifact incomplete")
    pipeline = artifact["pipeline"]
    hashes = {"renderer": digest(Path(__file__)),
              "manifest": digest(args.manifest), "calibration": digest(args.calibration),
              "yunet": digest(args.yunet), "sface": digest(args.sface),
              "recognition": digest(root / "research/fckface_lab/recognition.py"),
              "imaging": digest(root / "research/fckface_lab/imaging.py"),
              "patterns": digest(root / "research/fckface_lab/patterns.py"),
              "datasets": digest(root / "research/fckface_lab/datasets.py"),
              "similarity_source": digest(root / "research/reconstruct_graphics.py")}
    if hashes["manifest"] != artifact["dataset"]["manifest_sha256"]:
        raise ValueError("Manifest differs from calibration")
    for name in ("yunet", "sface"):
        if hashes[name] != pipeline["model_artifacts"][name]["sha256"]:
            raise ValueError(f"{name} model differs from calibration")
    for name in ("recognition", "imaging"):
        if hashes[name] != pipeline["source_sha256"][f"fckface_lab/{name}.py"]:
            raise ValueError(f"{name} preprocessing differs from calibration")
    if pipeline["detector_input"]["maximum_side_px"] != 640:
        raise ValueError("Detector scale differs from calibration")
    return out, groups, hashes


def generate(args):
    start = time.perf_counter()
    out, groups, hashes = preflight(args)
    cv2.setNumThreads(2)
    out.mkdir(parents=True, exist_ok=True)
    source_files = {
        "renderer": Path(__file__),
        "recognition": Path(__file__).parent / "fckface_lab/recognition.py",
        "imaging": Path(__file__).parent / "fckface_lab/imaging.py",
        "patterns": Path(__file__).parent / "fckface_lab/patterns.py",
        "datasets": Path(__file__).parent / "fckface_lab/datasets.py",
        "similarity_source": Path(__file__).parent / "reconstruct_graphics.py",
    }
    for name, path in source_files.items():
        snapshot = out / "source-snapshot" / path.name
        if name in ("recognition", "imaging", "patterns", "datasets"):
            snapshot = out / "source-snapshot" / "fckface_lab" / path.name
        write_new(snapshot, path.read_bytes())
        if digest(snapshot) != hashes[name]:
            raise RuntimeError("Source snapshot hash mismatch")
    source_records = []
    for identity in IDS:
        path = groups[identity]["neutral_front"]
        source_records.append({"identity": identity, "view": "neutral_front",
                               "path": str(path), "sha256": digest(path)})
    protocol = {
        "scope": "H3c preview only; source-only development; no gallery or scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "first two, index len(sorted development IDs)//2, last",
        "identities": IDS, "source_records": source_records,
        "canonical_side": CANONICAL_SIDE,
        "face_support": "patterns.face_mask > 0, full elliptical support",
        "sigma_space_canonical_112": SIGMA_SPACE_112,
        "bilateral_diameter": BILATERAL_DIAMETER, "bilateral_sigma_color": SIGMA_COLOR,
        "arms": ARMS, "luminance_bins": BINS, "palette": PALETTE_HEX,
        "boundary": {"colour": PALETTE_HEX[0], "inward_width_canonical": BOUNDARY_WIDTH_112},
        "input_hashes": hashes,
        "versions": {"python": sys.version, "opencv": cv2.__version__,
                     "numpy": np.__version__, "pillow": Image.__version__},
    }
    write_new(out / "protocol.json", (json.dumps(protocol, indent=2) + "\n").encode())
    model = SFaceModel(args.yunet, args.sface)
    results = []
    for identity in IDS:
        path = groups[identity]["neutral_front"]
        source = decode_image(path)
        detected = model.embed(source)
        if detected.status != "valid" or detected.selected_box is None or detected.landmarks is None:
            raise ValueError(f"Source detection unavailable: {identity} {detected.reason}")
        clean_jpeg = export_jpeg(source)
        clean = decode_image(clean_jpeg)
        clean_photo = out / identity / "clean" / "export.jpg"
        clean_crop = out / identity / "clean" / "face.png"
        write_new(clean_photo, clean_jpeg)
        write_new(clean_crop, crop_png(clean, detected.selected_box))
        full_sheet = [("clean", clean_photo)]
        face_sheet = [("clean", clean_crop)]
        for sigma_space in SIGMA_SPACE_112:
            for arm in ARMS:
                began = time.perf_counter()
                edited, support, metrics = render(source, detected.selected_box,
                                                  detected.landmarks, sigma_space, arm)
                jpeg = export_jpeg(edited)
                decoded = decode_image(jpeg)
                folder = out / identity / f"space{sigma_space}-{arm}"
                photo = folder / "export.jpg"
                crop = folder / "face.png"
                photo_hash = write_new(photo, jpeg)
                crop_hash = write_new(crop, crop_png(decoded, detected.selected_box))
                metrics["post_jpeg_distortion"] = actual_distortion(
                    clean, decoded, xyxy_to_xywh(detected.selected_box))
                metrics["edge_density_post_jpeg"] = edge_density(decoded, support)
                record = {"identity": identity, "preset_sigma_space_112": sigma_space,
                          "arm": arm, "source_path": str(path), "source_sha256": digest(path),
                          "source_box_xyxy": detected.selected_box,
                          "source_landmarks": detected.landmarks.tolist(),
                          "full_photo": str(photo), "full_photo_sha256": photo_hash,
                          "face_crop": str(crop), "face_crop_sha256": crop_hash,
                          "metrics": metrics,
                          "render_export_seconds": round(time.perf_counter() - began, 4)}
                results.append(record)
                label = f"s{sigma_space} {arm}"
                full_sheet.append((label, photo))
                face_sheet.append((label, crop))
        contact_sheet(full_sheet, out / f"{identity}-full-sheet.png", True)
        contact_sheet(face_sheet, out / f"{identity}-face-sheet.png", False)
        print(f"Rendered 8 previews for {identity}", flush=True)
    report = {"status": "preview_only_not_scored", "candidate_count": len(results),
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.perf_counter() - start, 3),
              "protocol_sha256": digest(out / "protocol.json"), "candidates": results,
              "sheets": {identity: {
                  "full_photo": str(out / f"{identity}-full-sheet.png"),
                  "full_photo_sha256": digest(out / f"{identity}-full-sheet.png"),
                  "face_crop": str(out / f"{identity}-face-sheet.png"),
                  "face_crop_sha256": digest(out / f"{identity}-face-sheet.png")}
                  for identity in IDS}}
    write_new(out / "previews.json", (json.dumps(report, indent=2) + "\n").encode())
    print(f"Preview-only output: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "calibration", "yunet", "sface", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    generate(parser.parse_args())
