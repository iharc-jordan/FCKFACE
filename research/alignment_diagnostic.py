"""H9 development-only alignment diagnostic on four frozen H7 JPEG exports.

This does not optimize images or change the official ArcFace evaluation. Only
canonical SCRFD queries count as independent-model evidence. YuNet-aligned
queries are counterfactual mechanism probes against the same fixed gallery.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np

from arcface_development import (ARC, ARMS, GALLERY_VIEWS, H7, IDS, MANIFEST,
                                 OUT as ARC_OUT, ArcFaceModel, verify_inputs)
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.imaging import decode_image
from fckface_lab.recognition import (SFaceModel, detector_frame,
    restore_face_coordinates, select_face)

DATA = Path.home() / "Downloads" / "FCKFACE-data"
OUT = DATA / "runs" / "alignment-transfer-diagnostic-v1"
WEIGHTS = Path.home() / "Downloads" / "Face Privacy Filter" / "cache" / ".deepface" / "weights"
ASSETS = ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")
EXPECTED_ASSETS = ("8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                   "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_file(path: Path, expected: str) -> str:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}: {actual} != {expected}")
    return actual


def selected_yunet(model: SFaceModel, rgb: np.ndarray, expected_box=None) -> dict:
    h, w = rgb.shape[:2]
    frame = detector_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    fh, fw = frame.shape[:2]
    model.detector.setInputSize((fw, fh))
    _, faces = model.detector.detect(frame)
    faces = restore_face_coordinates(faces, (w, h), (fw, fh))
    selected, reason = select_face(faces, expected_box)
    if selected is None:
        return {"status": "inconclusive", "reason": reason,
                "detection_count": 0 if faces is None else len(faces)}
    x, y, bw, bh = map(float, selected[:4])
    return {"status": "valid", "box": (x, y, x + bw, y + bh),
            "points": np.asarray(selected[4:14].reshape(5, 2), np.float32),
            "detection_count": len(faces)}


def geometry(points_a: np.ndarray, points_b: np.ndarray, face_align) -> dict:
    """Displacement of aligned-112 coordinates a -> source -> b."""
    a = np.vstack((face_align.estimate_norm(points_a, 112), [0, 0, 1]))
    b = np.vstack((face_align.estimate_norm(points_b, 112), [0, 0, 1]))
    transform = b @ np.linalg.inv(a)
    yy, xx = np.mgrid[0:112, 0:112]
    grid = np.vstack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
    displacement = np.linalg.norm((transform @ grid - grid)[:2], axis=0)
    return {"a_affine_to_112": a[:2].tolist(), "b_affine_to_112": b[:2].tolist(),
            "point_delta_px": (points_b - points_a).tolist(),
            "point_delta_norm_px": np.linalg.norm(points_b - points_a, axis=1).tolist(),
            "frame_displacement_px": {"median": float(np.median(displacement)),
                 "p90": float(np.percentile(displacement, 90)),
                 "p95": float(np.percentile(displacement, 95)),
                 "max": float(np.max(displacement))},
            "relative_rotation_deg": float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))),
            "relative_scale": float(np.linalg.norm(transform[:2, 0]))}


def crop_feature(model: ArcFaceModel, rgb: np.ndarray, points: np.ndarray,
                 png_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    crop = model.face_align.norm_crop(bgr, landmark=points, image_size=112)
    if not cv2.imwrite(str(png_path), crop):
        raise OSError(f"Cannot save diagnostic crop {png_path}")
    raw = np.asarray(model.recognizer.get_feat(crop), np.float32).reshape(-1)
    norm = float(np.linalg.norm(raw))
    if raw.size != 512 or not np.isfinite(raw).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("ArcFace counterfactual feature invalid")
    return raw / norm, crop


def gallery_scores(feature: np.ndarray, gallery: dict, threshold: float) -> dict:
    scores = {view: float(np.dot(feature, ref)) for view, ref in gallery.items()}
    maximum = max(scores.values())
    return {"per_reference": scores, "worst_gallery_cosine": maximum,
            "matches_gallery": maximum >= threshold}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Diagnostic output already exists: {OUT}")
    calibration_path = ARC_OUT / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    provenance = verify_inputs()
    if (calibration["status"] != "complete"
            or calibration["pipeline"]["provenance"] != provenance
            or calibration["pipeline"]["adapter_sha256"] != digest(Path(__file__).with_name("arcface_development.py"))
            or calibration["dataset"]["manifest_sha256"] != digest(MANIFEST)):
        raise ValueError("ArcFace calibration, adapter, or provenance changed")
    threshold = float(calibration["calibration"]["threshold"])
    frozen = json.loads((H7 / "frozen.json").read_text(encoding="utf-8"))
    audit = json.loads((H7 / "artifact-audit.json").read_text(encoding="utf-8"))
    if (not frozen["all_outputs_frozen_before_gallery"] or len(frozen["cases"]) != 2
            or {c["identity"] for c in frozen["cases"]} != set(IDS)
            or set(audit["identities"]) != set(IDS)
            or audit["runner_sha256"] != digest(H7 / "executed-source.py")):
        raise ValueError("H7 frozen set or executed source differs")
    groups = grouped_images(MANIFEST, "development")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {(r["identity"], r["view"]): r["sha256"] for r in manifest["images"]}
    inputs = {}
    for identity in IDS:
        entry = audit["identities"][identity]
        gallery = {}
        for view in GALLERY_VIEWS:
            path = groups[identity][view]
            expected = entry["gallery"][view]["sha256"]
            if expected != known[(identity, view)]:
                raise ValueError(f"Gallery manifest mismatch: {identity}/{view}")
            gallery[view] = {"path": str(path), "sha256": check_file(path, expected)}
        edits = {}
        case = next(c for c in frozen["cases"] if c["identity"] == identity)
        for arm in ARMS:
            path = H7 / identity / arm / "selected.jpg"
            expected = case["arms"][arm]["selected_jpeg_sha256"]
            if expected != entry["arms"][arm]["export"]["sha256"]:
                raise ValueError(f"Export hash mismatch: {identity}/{arm}")
            edits[arm] = {"path": str(path), "sha256": check_file(path, expected)}
        inputs[identity] = {"gallery": gallery, "edits": edits}
    for name, expected in zip(ASSETS, EXPECTED_ASSETS):
        check_file(WEIGHTS / name, expected)
    OUT.mkdir(parents=True)
    (OUT / "model-inputs").mkdir()
    (OUT / "crops").mkdir()
    for name, expected in zip(ASSETS, EXPECTED_ASSETS):
        dst = OUT / "model-inputs" / name
        shutil.copyfile(WEIGHTS / name, dst)
        check_file(dst, expected)
    shutil.copyfile(Path(__file__), OUT / "executed-source.py")
    protocol = {"schema_version": 1, "status": "frozen_before_inference", "at_utc": utc(),
        "purpose": "H9 development alignment mechanism diagnostic only",
        "scope": {"identities": IDS, "arms": ARMS, "edits": 4, "gallery_views_per_identity": GALLERY_VIEWS},
        "primary_comparison": "canonical official SCRFD ArcFace edit query versus fresh-YuNet-landmark ArcFace edit query; same unchanged canonical SCRFD gallery and ArcFace network",
        "secondary_comparison": "source-clean YuNet landmarks on each edit; explanatory only",
        "decision_gate": "All four cases valid; median(canonical worst-gallery cosine minus fresh YuNet worst-gallery cosine) >= 0.05 across four edits AND each identity median across its two arms > 0; otherwise no H9 jitter optimization",
        "limitations": "Counterfactual YuNet query paths are not official ArcFace scoring and cannot replace release/independent evaluation",
        "threshold": threshold, "calibration_sha256": digest(calibration_path),
        "arcface_provenance": provenance,
        "arcface_adapter_sha256": digest(Path(__file__).with_name("arcface_development.py")),
        "h7_frozen_sha256": digest(H7 / "frozen.json"),
        "h7_audit_sha256": digest(H7 / "artifact-audit.json"),
        "h7_executed_source_sha256": digest(H7 / "executed-source.py"),
        "source_sha256": digest(OUT / "executed-source.py"),
        "yunet_sface_assets": {name: {"original": str(WEIGHTS / name),
            "copy": str(OUT / "model-inputs" / name), "sha256": expected,
            "copy_sha256": digest(OUT / "model-inputs" / name)}
            for name, expected in zip(ASSETS, EXPECTED_ASSETS)},
        "inputs": inputs}
    save_json(OUT / "protocol.json", protocol)
    print("Frozen H9 diagnostic protocol; starting model calls", flush=True)
    cv2.setNumThreads(2)
    yunet = SFaceModel(OUT / "model-inputs" / ASSETS[0], OUT / "model-inputs" / ASSETS[1])
    arc = ArcFaceModel()
    started = perf_counter()
    rows = []
    try:
        for identity in IDS:
            clean_rgb = decode_image(Path(inputs[identity]["gallery"]["neutral_front"]["path"]))
            clean_arc = arc.embed(clean_rgb)
            clean_yu = selected_yunet(yunet, clean_rgb)
            if clean_arc.status != "valid" or clean_yu["status"] != "valid":
                raise RuntimeError(f"Invalid clean source {identity}: ArcFace={clean_arc.reason} YuNet={clean_yu.get('reason')}")
            refs = {}
            for view in GALLERY_VIEWS:
                ref = clean_arc if view == "neutral_front" else arc.embed(
                    decode_image(Path(inputs[identity]["gallery"][view]["path"])))
                if ref.status != "valid":
                    raise RuntimeError(f"Invalid canonical gallery {identity}/{view}: {ref.reason}")
                refs[view] = ref.feature
            _, clean_crop_scrfd = crop_feature(arc, clean_rgb, clean_arc.landmarks,
                OUT / "crops" / f"{identity}-clean-scrfd.png")
            _, clean_crop_yunet = crop_feature(arc, clean_rgb, clean_yu["points"],
                OUT / "crops" / f"{identity}-clean-yunet.png")
            clean_row = {"arcface_box": clean_arc.selected_box, "yunet_box": clean_yu["box"],
                "arcface_points": clean_arc.landmarks.tolist(), "yunet_points": clean_yu["points"].tolist(),
                "yunet_vs_scrfd": geometry(clean_arc.landmarks, clean_yu["points"], arc.face_align),
                "crop_mae_yunet_vs_scrfd": float(np.mean(np.abs(clean_crop_scrfd.astype(np.float32) - clean_crop_yunet.astype(np.float32))))}
            for arm in ARMS:
                edit_rgb = decode_image(Path(inputs[identity]["edits"][arm]["path"]))
                if edit_rgb.shape != clean_rgb.shape:
                    raise ValueError(f"Edit dimensions changed: {identity}/{arm}")
                canonical = arc.embed(edit_rgb, expected_box=clean_arc.selected_box)
                fresh = selected_yunet(yunet, edit_rgb, expected_box=clean_yu["box"])
                row = {"identity": identity, "arm": arm, "clean": clean_row,
                       "canonical_detection": {"status": canonical.status, "reason": canonical.reason,
                           "count": canonical.detection_count, "box": canonical.selected_box},
                       "fresh_yunet_detection": {"status": fresh["status"], "reason": fresh.get("reason"),
                           "count": fresh["detection_count"], "box": fresh.get("box")}}
                if canonical.status != "valid" or fresh["status"] != "valid":
                    row.update({"status": "inconclusive", "reason": "query_detection_or_selection_failure"})
                    rows.append(row)
                    continue
                canonical_feature, canonical_crop = crop_feature(arc, edit_rgb, canonical.landmarks,
                    OUT / "crops" / f"{identity}-{arm}-canonical-scrfd.png")
                fresh_feature, fresh_crop = crop_feature(arc, edit_rgb, fresh["points"],
                    OUT / "crops" / f"{identity}-{arm}-fresh-yunet.png")
                fixed_feature, fixed_crop = crop_feature(arc, edit_rgb, clean_yu["points"],
                    OUT / "crops" / f"{identity}-{arm}-clean-yunet.png")
                if np.max(np.abs(canonical_feature - canonical.feature)) > 1e-5:
                    raise ValueError("Canonical ArcFace feature not reproducible from official norm_crop")
                c = gallery_scores(canonical.feature, refs, threshold)
                f = gallery_scores(fresh_feature, refs, threshold)
                q = gallery_scores(fixed_feature, refs, threshold)
                row.update({"status": "valid", "arcface_points": canonical.landmarks.tolist(),
                    "fresh_yunet_points": fresh["points"].tolist(),
                    "clean_yunet_points": clean_yu["points"].tolist(),
                    "geometry": {
                        "fresh_yunet_vs_scrfd_edit": geometry(canonical.landmarks, fresh["points"], arc.face_align),
                        "scrfd_clean_to_edit": geometry(clean_arc.landmarks, canonical.landmarks, arc.face_align),
                        "yunet_clean_to_edit": geometry(clean_yu["points"], fresh["points"], arc.face_align)},
                    "crop_mae": {"fresh_yunet_vs_scrfd": float(np.mean(np.abs(fresh_crop.astype(np.float32)-canonical_crop.astype(np.float32)))),
                        "clean_yunet_vs_scrfd": float(np.mean(np.abs(fixed_crop.astype(np.float32)-canonical_crop.astype(np.float32)))),
                        "fresh_vs_clean_yunet": float(np.mean(np.abs(fresh_crop.astype(np.float32)-fixed_crop.astype(np.float32))))},
                    "canonical_official": c, "fresh_yunet_counterfactual": f,
                    "clean_yunet_counterfactual": q,
                    "canonical_minus_fresh_worst_cosine": c["worst_gallery_cosine"] - f["worst_gallery_cosine"]})
                rows.append(row)
                print(identity, arm, "delta", row["canonical_minus_fresh_worst_cosine"], flush=True)
        valid = len(rows) == 4 and all(r["status"] == "valid" for r in rows)
        deltas = [r["canonical_minus_fresh_worst_cosine"] for r in rows] if valid else []
        medians = {identity: float(np.median([r["canonical_minus_fresh_worst_cosine"]
                   for r in rows if r["identity"] == identity])) for identity in IDS} if valid else {}
        gate = bool(valid and np.median(deltas) >= .05 and all(v > 0 for v in medians.values()))
        save_json(OUT / "result.json", {"schema_version": 1, "status": "complete" if valid else "inconclusive",
            "completed_utc": utc(), "elapsed_seconds": perf_counter()-started,
            "protocol_sha256": digest(OUT / "protocol.json"), "threshold": threshold,
            "cases": rows, "decision": {"all_four_valid": valid,
                "median_canonical_minus_fresh": float(np.median(deltas)) if valid else None,
                "per_identity_median": medians, "gate_passed": gate,
                "interpretation": "alignment mechanism merits future optimization test" if gate else "H9 jitter optimization not supported by this diagnostic"}})
    except Exception as exc:
        save_json(OUT / "failure.json", {"at_utc": utc(), "type": type(exc).__name__,
            "message": str(exc), "completed_cases": rows})
        raise


if __name__ == "__main__":
    main()
