"""Score exact, pre-frozen development JPEGs with calibrated GhostFaceNet.

Example: python research/score_frozen_ghostface.py --candidates frozen.json
  --manifest FRLL/manifest.json --calibration runs/calibration-ghostface-v1/calibration.json
  --yunet cache/.deepface/weights/face_detection_yunet_2023mar.onnx
  --ghostface cache/.deepface/weights/ghostfacenet_v1.h5 --output runs/frozen-ghost-v1

frozen.json: {"schema_version":1,"items":[{"identity":"frll-001",
 "path":"/absolute/frozen.jpg","sha256":"64 lowercase hex digits",
 "method":"gradient","arm":"candidate-a","source_view":"neutral_front"}]}
Paths and IDs are examples; every identity must be in the development split.
No optimization, threshold selection, or held-out evaluation occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from time import perf_counter

import cv2
from PIL import Image

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.evaluation import Reference
from fckface_lab.ghostface import GhostFaceModel
from fckface_lab.imaging import decode_image, export_jpeg
from optimize_regions import CONDITIONS, VIEWS, evaluate_conditions


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checked_inputs(args):
    repo = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.is_relative_to(repo.parent) or output.exists():
        raise ValueError("Output must be a new directory outside Git")
    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    if (cal.get("status") != "complete" or
            cal.get("purpose") != "development_author_h5_ghostfacenet_calibration_only" or
            cal["dataset"]["split"] != "calibration" or
            cal["calibration"]["target_fmr"] != .001):
        raise ValueError("Wrong or incomplete frozen GhostFaceNet calibration")
    for name, hash_value in cal["pipeline"]["source_sha256"].items():
        source = (repo / name).resolve()
        if not source.is_relative_to(repo) or digest(source) != hash_value:
            raise ValueError(f"Calibrated pipeline source changed: {name}")
    for name, path in (("yunet", args.yunet), ("ghostface", args.ghostface)):
        record = cal["pipeline"]["model_artifacts"][name]
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Calibrated {name} artifact changed")
    if digest(args.manifest) != cal["dataset"]["manifest_sha256"]:
        raise ValueError("FRLL manifest differs from frozen calibration")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("split_version") != cal["dataset"]["split_version"]:
        raise ValueError("FRLL identity split differs from calibration")
    groups = grouped_images(args.manifest, "development")
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("items"), list) or not payload["items"]:
        raise ValueError("Expected nonempty frozen candidate items")
    items, keys, paths = [], set(), set()
    for row in payload["items"]:
        if set(row) != {"identity", "path", "sha256", "method", "arm", "source_view"}:
            raise ValueError("Candidate requires exactly identity,path,sha256,method,arm,source_view")
        identity, method, arm = row["identity"], row["method"], row["arm"]
        if (identity not in groups or manifest["identity_splits"].get(identity) != "development"
                or not set(VIEWS).issubset(groups[identity])
                or row["source_view"] != "neutral_front"):
            raise ValueError("Candidate has missing development source/gallery views")
        if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value)
                   for value in (method, arm)):
            raise ValueError("Method and arm must be short path-safe labels")
        path = Path(row["path"])
        if not path.is_absolute() or path.suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError("Candidate must be an absolute JPEG path")
        path = path.resolve(strict=True)
        if path.is_relative_to(repo.parent) or path.is_relative_to(output):
            raise ValueError("Candidate JPEG must be outside Git and output")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid candidate SHA-256")
        if digest(path) != row["sha256"]:
            raise ValueError(f"Candidate hash changed: {path}")
        with Image.open(path) as image:
            if image.format != "JPEG" or image.getexif() or image.info.get("icc_profile"):
                raise ValueError("Frozen candidate must be JPEG without EXIF/ICC")
        key = (identity, method, arm)
        if key in keys or path in paths:
            raise ValueError("Duplicate candidate key or file")
        keys.add(key); paths.add(path)
        items.append({**row, "path": str(path)})
    # The manifest hash alone does not prove the local gallery files stayed put.
    selected_ids = {row["identity"] for row in items}
    expected_gallery = {(row["identity"], row["view"]): row["sha256"]
                        for row in manifest["images"]
                        if row["identity"] in selected_ids and row["view"] in VIEWS}
    if len(expected_gallery) != len(selected_ids) * len(VIEWS):
        raise ValueError("Missing frozen gallery image records")
    for identity in selected_ids:
        for view in VIEWS:
            if digest(groups[identity][view]) != expected_gallery[(identity, view)]:
                raise ValueError(f"Gallery image hash changed: {identity}:{view}")
    return cal, groups, sorted(items, key=lambda r: (r["identity"], r["method"], r["arm"]))


def summary(rows: dict) -> dict:
    valid = len(rows) == len(CONDITIONS) and all(r["status"] == "valid" for r in rows.values())
    return {"valid_all": valid,
            "nonmatch_all": valid and all(r["own_identity_matched"] is False for r in rows.values()),
            "worst_cosine": max((r["maximum_cosine"] for r in rows.values()
                                 if r.get("maximum_cosine") is not None), default=None),
            "invalid_or_inconclusive": [name for name, r in rows.items() if r["status"] != "valid"]}


def run(args) -> None:
    cal, groups, items = checked_inputs(args)
    output = args.output.resolve()
    # Freeze exact candidate list and bytes before reading any clean gallery.
    output.mkdir(parents=True, exist_ok=False)
    preflight = {"created_utc": utc(), "candidate_manifest_sha256": digest(args.candidates),
                 "calibration_sha256": digest(args.calibration),
                 "dataset_manifest_sha256": digest(args.manifest),
                 "scorer_sha256": digest(Path(__file__)), "items": items}
    write_json(output / "preflight.json", preflight)
    cv2.setNumThreads(2)
    model = GhostFaceModel(args.yunet, args.ghostface)
    parity = model.selftest()
    if parity != cal["pipeline"]["synthetic_parity"]:
        raise ValueError("Author graph synthetic parity differs from calibration")
    threshold = Calibration(**cal["calibration"])
    results = []
    for row in items:
        start = perf_counter()
        identity = row["identity"]
        references, source_image, source_box = [], None, None
        reference_evidence = {}
        for view in VIEWS:
            path = groups[identity][view]
            image = decode_image(path)
            probe = model.embed(image)
            reference_evidence[view] = {"status": probe.status, "reason": probe.reason,
                                        "detection_count": probe.detection_count,
                                        "sha256": digest(path)}
            if probe.status == "valid":
                references.append(Reference(f"{identity}:{view}", identity, probe.feature))
            if view == "neutral_front":
                source_image, source_box = image, probe.selected_box
        controls = {}
        if source_box is not None:
            source_jpeg = export_jpeg(source_image)
            controls = evaluate_conditions(source_jpeg, source_box, model, references,
                                           threshold, identity, CONDITIONS,
                                           save_dir=output / identity / row["method"] / row["arm"] / "control")
        eligible = (len(references) == len(VIEWS) and len(controls) == len(CONDITIONS)
                    and all(r["status"] == "valid" and r["own_identity_matched"] is True
                            for r in controls.values()))
        candidate = {}
        if eligible:
            if digest(Path(row["path"])) != row["sha256"]:
                raise ValueError("Frozen JPEG changed after preflight")
            image = decode_image(Path(row["path"]))
            if image.shape != source_image.shape:
                raise ValueError("Frozen candidate dimensions differ from source")
            candidate = evaluate_conditions(Path(row["path"]).read_bytes(), source_box,
                                            model, references, threshold, identity,
                                            CONDITIONS, save_dir=output / identity /
                                            row["method"] / row["arm"] / "candidate")
        result = {"identity": identity, "method": row["method"], "arm": row["arm"],
                  "candidate_path": row["path"], "candidate_sha256": row["sha256"],
                  "reference_evidence": reference_evidence, "eligible": eligible,
                  "controls": controls, "control_summary": summary(controls),
                  "candidate": candidate, "candidate_summary": summary(candidate),
                  "seconds": round(perf_counter() - start, 6)}
        results.append(result)
        write_json(output / "progress.json", {"items": results})
    write_json(output / "results.json", {"status": "complete", "purpose": "development_ghostface_frozen_jpeg_scoring_only",
                                         "preflight_sha256": digest(output / "preflight.json"),
                                         "threshold": threshold.threshold, "conditions": list(CONDITIONS),
                                         "gallery_views": list(VIEWS), "items": results})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "manifest", "calibration", "yunet", "ghostface", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
