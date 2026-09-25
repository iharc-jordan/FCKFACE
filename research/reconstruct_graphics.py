"""Two-ID development test of an opaque graphic information bottleneck.

The render geometry, palette, arms, and sigmas are fixed in this file before
native SFace scores. Generate previews first; score only after appearance review.
No image or result is written inside the source repository.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import grouped_images
from fckface_lab.evaluation import Reference, evaluate_gallery
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.patterns import actual_distortion, face_mask
from fckface_lab.recognition import SFaceModel


IDS = ("frll-001", "frll-003")
VIEWS = ("neutral_front", "smiling_front", "neutral_left_3quarter", "neutral_right_3quarter")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960", "half_restore", "crop90", "blur")
SIGMAS_112 = (3, 5)
ARMS = ("bottleneck", "unblurred_input", "fixed_tiles")
PALETTE = np.array([[0x15, 0x1C, 0x64], [0x53, 0x31, 0xA8],
                    [0x14, 0x6F, 0xBD], [0x71, 0xDC, 0xF0]], dtype=np.uint8)
PALETTE_HEX = ("#151C64", "#5331A8", "#146FBD", "#71DCF0")
# Canonical XYXY rectangles at face width 112. No search or score feedback.
PANELS = (
    (34, 22, 47, 34), (65, 22, 78, 34),
    (22, 44, 33, 56), (79, 44, 90, 56),
    (26, 62, 42, 76), (70, 62, 86, 76),
)
GUARD_RADIUS_112 = 5.0
TILE_SIDE_112 = 8.0
CANONICAL_LANDMARKS = np.array([
    [.32 * 112, .35 * 112], [.68 * 112, .35 * 112],
    [.50 * 112, .56 * 112], [.37 * 112, .76 * 112],
    [.63 * 112, .76 * 112],
], dtype=np.float64)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def xyxy_to_xywh(box):
    left, top, right, bottom = map(float, box)
    return left, top, right - left, bottom - top


def face_crop_png(image: np.ndarray, box) -> bytes:
    from io import BytesIO
    left, top, right, bottom = box
    h, w = image.shape[:2]
    mx, my = .14 * (right - left), .14 * (bottom - top)
    bounds = (max(0, int(left - mx)), max(0, int(top - my)),
              min(w, int(right + mx)), min(h, int(bottom + my)))
    stream = BytesIO()
    Image.fromarray(image[bounds[1]:bounds[3], bounds[0]:bounds[2]]).save(stream, format="PNG")
    return stream.getvalue()


def similarity(landmarks: np.ndarray):
    """Map selected source landmarks to a canonical 112-pixel face."""
    a = landmarks.astype(np.float64) - landmarks.mean(axis=0)
    b = CANONICAL_LANDMARKS - CANONICAL_LANDMARKS.mean(axis=0)
    u, singular, vt = np.linalg.svd(a.T @ b)
    diagonal = np.diag([1.0, np.linalg.det(u @ vt)])
    matrix = (u @ diagonal @ vt) * (np.sum(singular * np.diag(diagonal)) / np.sum(a * a))
    offset = CANONICAL_LANDMARKS.mean(axis=0) - landmarks.mean(axis=0) @ matrix
    return matrix, offset


def geometry(shape, bbox_xyxy, landmarks):
    h, w = shape[:2]
    left, top, right, bottom = bbox_xyxy
    x0, y0 = max(0, int(left - .2 * (right - left))), max(0, int(top - .2 * (bottom - top)))
    x1, y1 = min(w, int(right + .2 * (right - left))), min(h, int(bottom + .2 * (bottom - top)))
    matrix, offset = similarity(landmarks)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    u = xx * matrix[0, 0] + yy * matrix[1, 0] + offset[0]
    v = xx * matrix[0, 1] + yy * matrix[1, 1] + offset[1]
    support = np.zeros((h, w), dtype=bool)
    patch_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for left_c, top_c, right_c, bottom_c in PANELS:
        patch_mask |= (u >= left_c) & (u < right_c) & (v >= top_c) & (v < bottom_c)
    for cx, cy in CANONICAL_LANDMARKS:
        patch_mask &= (u - cx) ** 2 + (v - cy) ** 2 > GUARD_RADIUS_112 ** 2
    face = face_mask(shape, xyxy_to_xywh(bbox_xyxy)) > 0
    patch_mask &= face[y0:y1, x0:x1]
    support[y0:y1, x0:x1] = patch_mask
    face_count = int(face.sum())
    coverage = int(support.sum()) / max(1, face_count)
    if not (0 < coverage <= .25):
        raise ValueError(f"Six-panel face coverage {coverage:.3f} outside (0, .25]")
    scale = float(np.linalg.norm(matrix[:, 0]))
    return support, (y0, y1, x0, x1), u, v, scale, coverage


def edge_density(image: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0
    return float((edges & mask).sum() / max(1, mask.sum()))


def render(source: np.ndarray, bbox_xyxy, landmarks, sigma_112: int, arm: str):
    if sigma_112 not in SIGMAS_112 or arm not in ARMS:
        raise ValueError("Unfrozen H3b preset")
    mask, (y0, y1, x0, x1), u, v, scale, coverage = geometry(
        source.shape, bbox_xyxy, landmarks)
    patch_mask = mask[y0:y1, x0:x1]
    tx = np.floor(u / TILE_SIDE_112).astype(np.int32)
    ty = np.floor(v / TILE_SIDE_112).astype(np.int32)
    tile_ids = tx.astype(np.int64) + ty.astype(np.int64) * 1000
    ids, inverse = np.unique(tile_ids[patch_mask], return_inverse=True)
    if arm == "fixed_tiles":
        tile_x = ids % 1000
        tile_y = ids // 1000
        levels = ((tile_x + 2 * tile_y) % 4).astype(np.int32)
    else:
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if arm == "bottleneck":
            # sigma is expressed at 112-pixel canonical face scale; the
            # selected source is blurred before 4-level tile reconstruction.
            gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_112 / scale)
        patch_gray = gray[y0:y1, x0:x1][patch_mask]
        sums = np.bincount(inverse, weights=patch_gray, minlength=len(ids))
        counts = np.bincount(inverse, minlength=len(ids))
        means = sums / np.maximum(counts, 1)
        levels = np.clip(np.floor(means / 64), 0, 3).astype(np.int32)
    palette_indices = levels[inverse]
    # A crisp indigo/purple grid makes every eight-pixel tile read as an
    # applied graphic, even where neighbouring coarse tone bins agree.
    border = ((u / TILE_SIDE_112) % 1 < .08) | ((v / TILE_SIDE_112) % 1 < .08)
    border_pixels = border[patch_mask]
    palette_indices[border_pixels] = ((tx + ty) % 2)[patch_mask][border_pixels]
    edited = source.copy()
    edited[y0:y1, x0:x1][patch_mask] = PALETTE[palette_indices]
    if not np.array_equal(edited[~mask], source[~mask]):
        raise RuntimeError("Source changed outside six opaque panels")
    histogram = {colour: int(np.sum(palette_indices == i)) for i, colour in enumerate(PALETTE_HEX)}
    return edited, mask, {"actual_panel_pixels": int(mask.sum()),
                          "face_support_pixels": int((face_mask(source.shape, xyxy_to_xywh(bbox_xyxy)) > 0).sum()),
                          "face_coverage_fraction": coverage,
                          "canonical_similarity_scale": scale,
                          "gaussian_sigma_source_pixels": sigma_112 / scale if arm == "bottleneck" else 0,
                          "palette_histogram_pre_jpeg": histogram,
                          "edge_density_pre_jpeg": edge_density(edited, mask),
                          "source_edge_density_same_panels": edge_density(source, mask),
                          "outside_panel_changed_pixels_pre_jpeg": 0,
                          "raw_distortion": actual_distortion(source, edited,
                                                               xyxy_to_xywh(bbox_xyxy))}


def contact_sheet(image_paths, destination):
    cell_w, cell_h, label_h = 210, 240, 32
    cols = 4
    sheet = Image.new("RGB", (cols * cell_w, 2 * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(image_paths):
        x, y = index % cols * cell_w, index // cols * (cell_h + label_h)
        with Image.open(path) as opened:
            face = opened.convert("RGB")
            face.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(face, (x + (cell_w - face.width) // 2, y + label_h))
        draw.text((x + 5, y + 6), label, fill="black")
    sheet.save(destination)


def setup(args):
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Private outputs must be outside source repository")
    selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
    groups = grouped_images(args.manifest, "development")
    if selection["identities"] != sorted(groups)[:8] or tuple(selection["identities"][:2]) != IDS:
        raise ValueError("Frozen first two development identities differ")
    artifact = json.loads(args.calibration.read_text(encoding="utf-8"))
    if artifact.get("status") != "complete" or artifact.get("purpose") != "development_native_sface_calibration_only":
        raise ValueError("Wrong or unfinished native SFace calibration")
    pipeline = artifact["pipeline"]
    for name, path in (("yunet", args.yunet), ("sface", args.sface)):
        if digest(path) != pipeline["model_artifacts"][name]["sha256"]:
            raise ValueError(f"Calibration {name} model hash differs")
    source = Path(__file__).parent / "fckface_lab"
    for name in ("recognition.py", "imaging.py", "evaluation.py"):
        if digest(source / name) != pipeline["source_sha256"][f"fckface_lab/{name}"]:
            raise ValueError(f"Calibration preprocessing code changed: {name}")
    if digest(args.manifest) != artifact["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from calibration")
    if pipeline["detector_input"]["maximum_side_px"] != 640 or pipeline["detector_input"]["score_threshold"] != .9:
        raise ValueError("Calibration detector settings changed")
    cv2.setNumThreads(2)
    return out, groups, Calibration(**artifact["calibration"]), SFaceModel(args.yunet, args.sface)


def generate(args):
    out, groups, calibration, model = setup(args)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "scope": "H3b two-ID development appearance generation before scores",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identities": IDS, "views": VIEWS, "conditions_for_later_scoring": CONDITIONS,
        "sigmas_canonical_112": SIGMAS_112, "arms": ARMS,
        "panels_xyxy_canonical_112": PANELS, "guard_radius_canonical_112": GUARD_RADIUS_112,
        "tile_side_canonical_112": TILE_SIDE_112, "opaque_palette": PALETTE_HEX,
        "level_rule": "floor(mean tile luminance / 64), clipped 0..3; bottleneck blurs before tile mean; unblurred control omits Gaussian; fixed control ignores source luminance",
        "threshold_for_later_scoring": calibration.threshold,
        "source_sha256": {"renderer": digest(Path(__file__)),
                          "recognition": digest(Path(__file__).parent / "fckface_lab" / "recognition.py"),
                          "patterns_metrics": digest(Path(__file__).parent / "fckface_lab" / "patterns.py"),
                          "imaging": digest(Path(__file__).parent / "fckface_lab" / "imaging.py"),
                          "evaluation": digest(Path(__file__).parent / "fckface_lab" / "evaluation.py"),
                          "manifest": digest(args.manifest), "selection": digest(args.selection),
                          "calibration": digest(args.calibration),
                          "yunet": digest(args.yunet), "sface": digest(args.sface)},
        "versions": {"python": sys.version, "opencv": cv2.__version__,
                     "numpy": np.__version__, "pillow": Image.__version__},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    previews = []
    for identity in IDS:
        source_path = groups[identity]["neutral_front"]
        source = decode_image(source_path)
        detected = model.embed(source)
        if detected.status != "valid" or detected.selected_box is None or detected.landmarks is None:
            raise ValueError(f"Source face unavailable for {identity}: {detected.reason}")
        clean_jpeg = export_jpeg(source)
        clean = decode_image(clean_jpeg)
        clean_dir = out / identity / "clean"
        clean_hash = save(clean_dir / "export.jpg", clean_jpeg)
        clean_crop_hash = save(clean_dir / "export-face.png",
                               face_crop_png(clean, detected.selected_box))
        sheet = [("clean", clean_dir / "export-face.png")]
        for sigma in SIGMAS_112:
            for arm in ARMS:
                began = time.perf_counter()
                edited, panel_mask, metrics = render(source, detected.selected_box,
                                                     detected.landmarks, sigma, arm)
                jpeg = export_jpeg(edited)
                decoded = decode_image(jpeg)
                folder = out / identity / f"sigma{sigma}-{arm}"
                image_hash = save(folder / "export.jpg", jpeg)
                crop_hash = save(folder / "export-face.png",
                                 face_crop_png(decoded, detected.selected_box))
                metrics["post_jpeg_distortion"] = actual_distortion(
                    clean, decoded, xyxy_to_xywh(detected.selected_box))
                metrics["edge_density_post_jpeg_same_panels"] = edge_density(decoded, panel_mask)
                previews.append({"identity": identity, "source_path": str(source_path),
                                 "source_sha256": digest(source_path),
                                 "source_box_xyxy": detected.selected_box,
                                 "sigma_canonical_112": sigma, "arm": arm,
                                 "metrics": metrics, "jpeg_sha256": image_hash,
                                 "face_crop_sha256": crop_hash,
                                 "render_export_seconds": round(time.perf_counter() - began, 4),
                                 "full_photo": str(folder / "export.jpg"),
                                 "face_crop": str(folder / "export-face.png")})
                sheet.append((f"s{sigma} {arm}", folder / "export-face.png"))
        contact_sheet(sheet, out / f"{identity}-contact-sheet.png")
        print(f"Generated six fixed graphics for {identity}", flush=True)
    report = {"status": "preview_only_not_scored", "candidate_count": len(previews),
              "clean_jpeg_sha256": {identity: digest(out / identity / "clean" / "export.jpg") for identity in IDS},
              "contact_sheet_sha256": {identity: digest(out / f"{identity}-contact-sheet.png") for identity in IDS},
              "candidates": previews}
    (out / "previews.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Preview complete: {out}; no SFace comparison scores run", flush=True)


def score(args):
    out, groups, calibration, model = setup(args)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    preview = json.loads((out / "previews.json").read_text(encoding="utf-8"))
    if preview["status"] != "preview_only_not_scored" or len(preview["candidates"]) != 12:
        raise ValueError("Expected exactly twelve unscored, frozen preview candidates")
    expected_hashes = protocol["source_sha256"]
    current = {"renderer": digest(Path(__file__)), "patterns_metrics": digest(Path(__file__).parent / "fckface_lab" / "patterns.py"),
               "manifest": digest(args.manifest), "selection": digest(args.selection),
               "calibration": digest(args.calibration), "yunet": digest(args.yunet), "sface": digest(args.sface)}
    if any(current[name] != expected_hashes[name] for name in current):
        raise ValueError("Preview and scoring inputs or renderer code differ")
    score_path = out / "scores.json"
    if score_path.exists():
        raise FileExistsError(score_path)
    results = []
    for identity in IDS:
        references = []
        reference_status = {}
        for view in VIEWS:
            pixels = decode_image(groups[identity][view])
            embedded = model.embed(pixels)
            reference_status[view] = {"status": embedded.status,
                                      "reason": embedded.reason,
                                      "sha256": digest(groups[identity][view])}
            if embedded.status == "valid":
                references.append(Reference(f"{identity}:{view}", identity, embedded.feature))
        candidate_rows = [row for row in preview["candidates"] if row["identity"] == identity]
        source_box = candidate_rows[0]["source_box_xyxy"]
        clean_base = (out / identity / "clean" / "export.jpg").read_bytes()
        clean_variants = make_variants(clean_base, source_box)
        rows = []
        for label, base, folder in [("clean", clean_base, out / identity / "clean")] + [
            (f"sigma{row['sigma_canonical_112']}-{row['arm']}",
             Path(row["full_photo"]).read_bytes(),
             out / identity / f"sigma{row['sigma_canonical_112']}-{row['arm']}")
            for row in candidate_rows]:
            if label != "clean":
                expected = next(row["jpeg_sha256"] for row in candidate_rows
                                if label == f"sigma{row['sigma_canonical_112']}-{row['arm']}")
                if hashlib.sha256(base).hexdigest() != expected:
                    raise ValueError(f"Preview JPEG changed: {label}")
            variants = make_variants(base, source_box)
            conditions = {}
            for name in CONDITIONS:
                started = time.perf_counter()
                variant = variants[name]
                item = {"status": variant.status, "reason": variant.reason,
                        "target_box_xyxy": variant.target_box}
                if variant.status == "valid":
                    path = folder / f"{name}.jpg"
                    if name == "export":
                        if path.read_bytes() != variant.jpeg:
                            raise ValueError(f"Base export changed: {path}")
                        item["jpeg_sha256"] = digest(path)
                    else:
                        item["jpeg_sha256"] = save(path, variant.jpeg)
                        item["face_crop_sha256"] = save(
                            folder / f"{name}-face.png",
                            face_crop_png(variant.image, variant.target_box))
                    control = clean_variants[name]
                    if control.status == "valid" and control.image.shape == variant.image.shape:
                        item["post_jpeg_distortion"] = actual_distortion(
                            control.image, variant.image, xyxy_to_xywh(variant.target_box))
                    probe = model.embed(variant.image, expected_box=variant.target_box)
                    item["detection"] = {"status": probe.status, "reason": probe.reason,
                                         "count": probe.detection_count,
                                         "selected_box_xyxy": probe.selected_box}
                    if len(references) == len(VIEWS):
                        evaluation = evaluate_gallery(
                            probe, references, calibration, own_identity=identity,
                            own_source_name=f"{identity}:neutral_front",
                            expected_reference_names=[f"{identity}:{v}" for v in VIEWS])
                        item["evaluation"] = {
                            "status": evaluation.status, "reason": evaluation.reason,
                            "per_reference": dict(evaluation.per_reference),
                            "matching_references": list(evaluation.matching_references),
                            "own_source_score": evaluation.own_source_score,
                            "own_identity_matched": evaluation.own_identity_matched}
                    else:
                        item["evaluation"] = {"status": "inconclusive",
                                              "reason": "missing_or_invalid_reference"}
                else:
                    item["evaluation"] = {"status": "inconclusive", "reason": variant.reason}
                item["total_seconds"] = round(time.perf_counter() - started, 4)
                conditions[name] = item
            rows.append({"label": label, "conditions": conditions})
        control = rows[0]
        eligible = (len(references) == len(VIEWS) and all(
            control["conditions"][name]["evaluation"]["status"] == "valid" and
            control["conditions"][name]["evaluation"]["own_identity_matched"] is True
            for name in CONDITIONS))
        results.append({"identity": identity, "reference_status": reference_status,
                        "eligible_clean_control": eligible,
                        "ineligible_reason": None if eligible else "missing_or_unmatched_clean_control",
                        "cases": rows})
        print(f"Scored {identity}, clean eligible={eligible}", flush=True)
    report = {"scope": "Two-ID native SFace development H3b screen; no privacy or novelty claim",
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "calibration_threshold": calibration.threshold,
              "protocol_sha256": digest(out / "protocol.json"),
              "previews_sha256": digest(out / "previews.json"),
              "identities": results}
    score_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Scored exact twelve candidates: {score_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "score"), required=True)
    for name in ("manifest", "calibration", "yunet", "sface", "selection", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    (generate if args.phase == "generate" else score)(args)
