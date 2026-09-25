"""H3d fixed step-to-ridge portrait linework previews, without scoring.

Four development sources, two canonical stroke widths, and three fixed arms.
Recognition galleries and other identity views are never read here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

from fckface_lab.imaging import decode_image, export_jpeg
from fckface_lab.patterns import actual_distortion, face_mask
from fckface_lab.recognition import SFaceModel
import render_stencil as stencil


IDS = stencil.IDS
CANONICAL_SIDE = 112
STROKE_WIDTHS_112 = (2, 3)
ARMS = ("cyan_paper", "continuous_palette", "four_level_palette")
BLUR_SIGMAS = (.8, 1.6)
CANNY_THRESHOLDS = (40, 100)
MIN_CONTOUR_PIXELS = 6
PALETTE = stencil.PALETTE
PALETTE_HEX = stencil.PALETTE_HEX
PAPER_RGB = PALETTE[3]
INK_RGB = PALETTE[0]


def _regions():
    """Fixed face-feature zones after the five landmarks map to canonical 112."""
    yy, xx = np.mgrid[:CANONICAL_SIDE, :CANONICAL_SIDE]
    ellipse = lambda cx, cy, rx, ry: (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1
    left_eye = ellipse(36, 39, 18, 15)
    right_eye = ellipse(76, 39, 18, 15)
    eyes = left_eye | right_eye
    nose = ellipse(56, 62, 16, 18)
    nostrils = ellipse(56, 69, 12, 7)
    mouth = ellipse(56, 84, 28, 13)
    radial = ((xx - 56) / 48) ** 2 + ((yy - 61) / 54) ** 2
    jaw = (yy >= 73) & (radial >= .77) & (radial <= 1.12)
    return {"eyes_brows": eyes, "nose": nose, "mouth": mouth,
            "jaw": jaw, "nostrils": nostrils}


def _keep_contours(binary: np.ndarray):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    sizes = stats[:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, dtype=bool)
    keep[1:] = sizes[1:] >= MIN_CONTOUR_PIXELS
    return keep[labels], int(count - 1), int(np.count_nonzero(keep))


def _dark_anchors(gray: np.ndarray, regions):
    anchors = np.zeros(gray.shape, dtype=bool)
    thresholds = {}
    for name, zone in (("eyes_brows", regions["eyes_brows"]),
                       ("nostrils", regions["nostrils"]),
                       ("mouth", regions["mouth"])):
        threshold = min(float(np.percentile(gray[zone], 24)), 105.0)
        thresholds[name] = threshold
        anchors |= zone & (gray <= threshold)
    # Restrict specks, preserving source-derived local shadow shapes.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        anchors.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
    return keep[labels], thresholds


def _canonical_art(gray: np.ndarray, width: int, arm: str):
    regions = _regions()
    allowed = regions["eyes_brows"] | regions["nose"] | regions["mouth"] | regions["jaw"]
    fine = cv2.GaussianBlur(gray, (0, 0), sigmaX=BLUR_SIGMAS[0])
    coarse = cv2.GaussianBlur(gray, (0, 0), sigmaX=BLUR_SIGMAS[1])
    edges_fine = cv2.Canny(fine, *CANNY_THRESHOLDS) > 0
    edges_coarse = cv2.Canny(coarse, *CANNY_THRESHOLDS) > 0
    close_to_coarse = cv2.dilate(edges_coarse.astype(np.uint8),
                                 np.ones((3, 3), np.uint8)) > 0
    stable = edges_fine & close_to_coarse & allowed
    retained, component_count, retained_count = _keep_contours(stable)
    stroke = cv2.dilate(retained.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))) > 0
    anchors, anchor_thresholds = _dark_anchors(gray, regions)
    ink = stroke | anchors
    if arm == "cyan_paper":
        art = np.empty((CANONICAL_SIDE, CANONICAL_SIDE, 3), dtype=np.uint8)
        art[:] = PAPER_RGB
    elif arm == "continuous_palette":
        art = np.rint(np.stack([
            np.interp(gray, (0, 85, 170, 255), PALETTE[:, c]) for c in range(3)
        ], axis=-1)).astype(np.uint8)
    elif arm == "four_level_palette":
        art = PALETTE[np.clip(gray.astype(np.int16) // 64, 0, 3)]
    else:
        raise ValueError(f"Unknown fixed H3d arm: {arm}")
    art[ink] = INK_RGB
    return art, {"stable_contour_pixels": int(np.count_nonzero(stable)),
                 "candidate_components": component_count,
                 "retained_components": retained_count,
                 "retained_contour_pixels": int(np.count_nonzero(retained)),
                 "dilated_stroke_pixels": int(np.count_nonzero(stroke)),
                 "anchor_pixels": int(np.count_nonzero(anchors)),
                 "total_ink_pixels": int(np.count_nonzero(ink)),
                 "anchor_luminance_thresholds": anchor_thresholds,
                 "allowed_region_pixels": int(np.count_nonzero(allowed))}


def render(source: np.ndarray, box, landmarks, width: int, arm: str):
    if width not in STROKE_WIDTHS_112 or arm not in ARMS:
        raise ValueError("Unknown frozen H3d arm or stroke width")
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Expected RGB uint8 source")
    h, w = source.shape[:2]
    x0, y0, x1, y1 = map(float, box)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        raise ValueError("Invalid XYXY source box")
    left = max(0, int(np.floor(x0 - .12 * bw)))
    top = max(0, int(np.floor(y0 - .12 * bh)))
    right = min(w, int(np.ceil(x1 + .12 * bw)))
    bottom = min(h, int(np.ceil(y1 + .12 * bh)))
    support = face_mask(source[top:bottom, left:right].shape,
                        (x0 - left, y0 - top, bw, bh)) > 0
    matrix, offset = stencil.similarity(np.asarray(landmarks, dtype=np.float64))
    forward = np.array([[matrix[0, 0], matrix[1, 0], offset[0]],
                        [matrix[0, 1], matrix[1, 1], offset[1]]], dtype=np.float32)
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    canonical = cv2.warpAffine(gray, forward, (CANONICAL_SIDE, CANONICAL_SIDE),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    art, line_metrics = _canonical_art(canonical, width, arm)
    yy, xx = np.mgrid[top:bottom, left:right].astype(np.float32)
    u = xx * matrix[0, 0] + yy * matrix[1, 0] + offset[0]
    v = xx * matrix[0, 1] + yy * matrix[1, 1] + offset[1]
    mapped = cv2.remap(art, u.astype(np.float32), v.astype(np.float32),
                       interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_REPLICATE)
    scale = float(np.linalg.norm(matrix[:, 0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid canonical similarity scale")
    boundary_pixels = max(1.0, stencil.BOUNDARY_WIDTH_112 / scale)
    distance = cv2.distanceTransform(support.astype(np.uint8),
                                     cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    mapped[support & (distance <= boundary_pixels)] = INK_RGB
    edited = source.copy()
    patch = edited[top:bottom, left:right]
    patch[support] = mapped[support]
    full_support = np.zeros((h, w), dtype=bool)
    full_support[top:bottom, left:right] = support
    if np.any(edited[~full_support] != source[~full_support]):
        raise RuntimeError("Changed pixels outside face support")
    palette_hist = {hex_rgb: int(np.count_nonzero(np.all(mapped[support] == rgb, axis=1)))
                    for hex_rgb, rgb in zip(PALETTE_HEX, PALETTE)}
    metrics = {
        "canonical_similarity_scale": scale,
        "face_support_pixels": int(np.count_nonzero(support)),
        "face_support_over_bbox_area": float(np.count_nonzero(support) / (bw * bh)),
        "changed_pixels_pre_jpeg": int(np.count_nonzero(
            np.any(patch != source[top:bottom, left:right], axis=2) & support)),
        "outside_support_changed_pixels_pre_jpeg": 0,
        "boundary_width_source_pixels": boundary_pixels,
        "palette_exact_rgb_histogram_pre_jpeg": palette_hist,
        "unique_rgb_colours_in_support_pre_jpeg": int(len(np.unique(mapped[support], axis=0))),
        "edge_density_pre_jpeg": stencil.edge_density(edited, full_support),
        "source_edge_density_same_support": stencil.edge_density(source, full_support),
        "raw_distortion": actual_distortion(source, edited, stencil.xyxy_to_xywh(box)),
        **line_metrics,
    }
    return edited, full_support, metrics


def generate(args):
    started = time.perf_counter()
    out, groups, source_hashes = stencil.preflight(args)
    cv2.setNumThreads(2)
    out.mkdir(parents=True, exist_ok=True)
    source_files = {
        "render_contours.py": Path(__file__),
        "render_stencil.py": Path(__file__).parent / "render_stencil.py",
        "reconstruct_graphics.py": Path(__file__).parent / "reconstruct_graphics.py",
        "fckface_lab/datasets.py": Path(__file__).parent / "fckface_lab/datasets.py",
        "fckface_lab/imaging.py": Path(__file__).parent / "fckface_lab/imaging.py",
        "fckface_lab/patterns.py": Path(__file__).parent / "fckface_lab/patterns.py",
        "fckface_lab/recognition.py": Path(__file__).parent / "fckface_lab/recognition.py",
    }
    snapshots = {}
    for name, path in source_files.items():
        snapshots[name] = stencil.write_new(out / "source-snapshot" / name, path.read_bytes())
        if snapshots[name] != stencil.digest(path):
            raise RuntimeError("Code snapshot hash mismatch")
    source_records = [{"identity": identity, "view": "neutral_front",
                       "path": str(groups[identity]["neutral_front"]),
                       "sha256": stencil.digest(groups[identity]["neutral_front"])}
                      for identity in IDS]
    protocol = {
        "scope": "H3d development preview only; no gallery, no recognition scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "first two, middle index len(sorted development IDs)//2, last",
        "identities": IDS, "source_records": source_records,
        "canonical_side": CANONICAL_SIDE,
        "face_support": "patterns.face_mask > 0, full elliptical face support",
        "stroke_widths_canonical_112": STROKE_WIDTHS_112,
        "arms": ARMS, "gaussian_sigmas_canonical_112": BLUR_SIGMAS,
        "canny_thresholds": CANNY_THRESHOLDS,
        "cross_scale_consistency": "fine Canny edge within 1 canonical pixel of coarse Canny edge",
        "min_connected_contour_pixels": MIN_CONTOUR_PIXELS,
        "region_geometry": "canonical eyes/brows, nose, mouth, lower outer jaw; no forehead/cheek texture",
        "anchor_rule": "source luminance <= per-zone 24th percentile capped at 105; connected area >= 3",
        "paper_rgb_hex": PALETTE_HEX[3], "ink_rgb_hex": PALETTE_HEX[0],
        "control_background_palette": PALETTE_HEX,
        "input_hashes": source_hashes, "code_snapshot_sha256": snapshots,
        "versions": {"python": sys.version, "opencv": cv2.__version__,
                     "numpy": np.__version__, "pillow": Image.__version__},
    }
    stencil.write_new(out / "protocol.json", (json.dumps(protocol, indent=2) + "\n").encode())
    model = SFaceModel(args.yunet, args.sface)
    records = []
    for identity in IDS:
        path = groups[identity]["neutral_front"]
        source = decode_image(path)
        detection = model.embed(source)
        if detection.status != "valid" or detection.selected_box is None or detection.landmarks is None:
            raise ValueError(f"Source detection unavailable: {identity} {detection.reason}")
        clean_jpeg = export_jpeg(source)
        clean = decode_image(clean_jpeg)
        clean_full = out / identity / "clean" / "export.jpg"
        clean_face = out / identity / "clean" / "face.png"
        stencil.write_new(clean_full, clean_jpeg)
        stencil.write_new(clean_face, stencil.crop_png(clean, detection.selected_box))
        full_sheet = [("clean", clean_full)]
        face_sheet = [("clean", clean_face)]
        for width in STROKE_WIDTHS_112:
            for arm in ARMS:
                began = time.perf_counter()
                edited, support, metrics = render(source, detection.selected_box,
                                                  detection.landmarks, width, arm)
                jpeg = export_jpeg(edited)
                decoded = decode_image(jpeg)
                folder = out / identity / f"stroke{width}-{arm}"
                photo = folder / "export.jpg"
                face = folder / "face.png"
                image_hash = stencil.write_new(photo, jpeg)
                face_hash = stencil.write_new(face, stencil.crop_png(decoded,
                                                                     detection.selected_box))
                metrics["post_jpeg_distortion"] = actual_distortion(
                    clean, decoded, stencil.xyxy_to_xywh(detection.selected_box))
                metrics["edge_density_post_jpeg"] = stencil.edge_density(decoded, support)
                records.append({"identity": identity, "stroke_width_canonical_112": width,
                                "arm": arm, "source_path": str(path),
                                "source_sha256": stencil.digest(path),
                                "source_box_xyxy": detection.selected_box,
                                "source_landmarks": detection.landmarks.tolist(),
                                "full_photo": str(photo), "full_photo_sha256": image_hash,
                                "face_crop": str(face), "face_crop_sha256": face_hash,
                                "metrics": metrics,
                                "render_export_seconds": round(time.perf_counter() - began, 4)})
                label = f"w{width} {arm}"
                full_sheet.append((label, photo))
                face_sheet.append((label, face))
        stencil.contact_sheet(full_sheet, out / f"{identity}-full-sheet.png", True)
        stencil.contact_sheet(face_sheet, out / f"{identity}-face-sheet.png", False)
        print(f"Generated six H3d portraits for {identity}", flush=True)
    report = {
        "status": "preview_only_not_scored", "candidate_count": len(records),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "protocol_sha256": stencil.digest(out / "protocol.json"),
        "candidates": records,
        "sheets": {identity: {
            "full_photo": str(out / f"{identity}-full-sheet.png"),
            "full_photo_sha256": stencil.digest(out / f"{identity}-full-sheet.png"),
            "face_crop": str(out / f"{identity}-face-sheet.png"),
            "face_crop_sha256": stencil.digest(out / f"{identity}-face-sheet.png")}
            for identity in IDS},
    }
    stencil.write_new(out / "previews.json", (json.dumps(report, indent=2) + "\n").encode())
    print(f"H3d preview-only run completed: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "calibration", "yunet", "sface", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    generate(parser.parse_args())
