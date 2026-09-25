"""Score frozen development JPEGs with the calibrated native SFace pipeline.

This performs no optimization or candidate selection. Inputs use the same
schema_version=1 frozen candidate manifest as score_frozen_arcface.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from time import perf_counter

import cv2
from PIL import Image

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.evaluation import Reference
from fckface_lab.imaging import decode_image, export_jpeg
from fckface_lab.recognition import SFaceModel
from optimize_regions import CONDITIONS, VIEWS, evaluate_conditions


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checked_inputs(args):
    repo = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(repo.parent):
        raise ValueError("Output must be new and outside Git")
    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    if (cal.get("status") != "complete" or
            cal.get("purpose") != "development_native_sface_calibration_only" or
            cal["calibration"]["target_fmr"] != .001 or
            cal["dataset"]["split"] != "calibration"):
        raise ValueError("Wrong or incomplete SFace calibration")
    for name, expected in cal["pipeline"]["source_sha256"].items():
        path = args.models_registry.resolve(strict=True) if name == "models.json" else (repo / name).resolve()
        if name == "models.json":
            if digest(path) != expected:
                raise ValueError("Exact calibrated model registry required")
            frozen = json.loads(path.read_text(encoding="utf-8"))
            current = json.loads((repo / "models.json").read_text(encoding="utf-8"))
            if frozen["development"] != current["development"]:
                raise ValueError("Current development model registry differs from calibration")
            continue
        if not path.is_relative_to(repo):
            raise ValueError(f"Calibrated source escaped repository: {name}")
        if digest(path) != expected:
            raise ValueError(f"Calibrated source changed: {name}")
    for name, path in (("yunet", args.yunet), ("sface", args.sface)):
        artifact = cal["pipeline"]["model_artifacts"][name]
        if digest(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Calibrated model changed: {name}")
    if digest(args.manifest) != cal["dataset"]["manifest_sha256"]:
        raise ValueError("Manifest differs from calibration")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["split_version"] != cal["dataset"]["split_version"]:
        raise ValueError("Frozen split differs from calibration")
    groups = grouped_images(args.manifest, "development")
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("items"):
        raise ValueError("Expected nonempty schema_version=1 candidate manifest")
    items, keys, paths = [], set(), set()
    for row in payload["items"]:
        if set(row) != {"identity", "path", "sha256", "method", "arm", "source_view"}:
            raise ValueError("Wrong candidate keys")
        identity = row["identity"]
        if (identity not in groups or manifest["identity_splits"].get(identity) != "development"
                or row["source_view"] != "neutral_front" or
                not set(VIEWS).issubset(groups[identity])):
            raise ValueError("Candidate requires development identity and four clean views")
        if not all(isinstance(row[key], str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", row[key])
                   for key in ("method", "arm")):
            raise ValueError("Unsafe candidate label")
        candidate = Path(row["path"])
        if not candidate.is_absolute() or candidate.suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError("Candidate must be absolute JPEG")
        candidate = candidate.resolve(strict=True)
        if (candidate.is_relative_to(repo.parent) or candidate.is_relative_to(output)
                or digest(candidate) != row["sha256"]):
            raise ValueError("Candidate location or SHA-256 invalid")
        with Image.open(candidate) as image:
            if image.format != "JPEG" or image.getexif() or image.info.get("icc_profile"):
                raise ValueError("Candidate must be metadata-free JPEG")
            image.verify()
        key = (identity, row["method"], row["arm"])
        if key in keys or candidate in paths:
            raise ValueError("Duplicate candidate")
        keys.add(key); paths.add(candidate)
        items.append({**row, "path": str(candidate)})
    selected = {row["identity"] for row in items}
    expected = {(row["identity"], row["view"]): row["sha256"]
                for row in manifest["images"]
                if row["identity"] in selected and row["view"] in VIEWS}
    if len(expected) != len(selected) * len(VIEWS):
        raise ValueError("Missing clean gallery records")
    for identity in selected:
        for view in VIEWS:
            if digest(groups[identity][view]) != expected[(identity, view)]:
                raise ValueError("Clean gallery changed from manifest")
    return cal, groups, sorted(items, key=lambda row: (row["identity"], row["arm"]))


def summarize(rows: dict) -> dict:
    valid = len(rows) == len(CONDITIONS) and all(row["status"] == "valid" for row in rows.values())
    return {"valid_all": valid,
            "nonmatch_all": valid and all(row.get("own_identity_matched") is False
                                           for row in rows.values()),
            "valid_conditions": sum(row["status"] == "valid" for row in rows.values()),
            "nonmatch_conditions": sum(row.get("own_identity_matched") is False
                                       and row["status"] == "valid" for row in rows.values()),
            "invalid_or_inconclusive": [name for name, row in rows.items()
                                        if row["status"] != "valid"],
            "worst_gallery_cosine": max((row["maximum_cosine"] for row in rows.values()
                                         if row.get("maximum_cosine") is not None), default=None)}


def run(args) -> None:
    cal, groups, items = checked_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "preflight.json", {
        "candidate_manifest_sha256": digest(args.candidates),
        "calibration_sha256": digest(args.calibration),
        "dataset_manifest_sha256": digest(args.manifest),
        "scorer_sha256": digest(Path(__file__)),
        "models_registry_current_sha256": digest(Path(__file__).with_name("models.json")),
        "models_registry_calibration_sha256": cal["pipeline"]["source_sha256"]["models.json"],
        "models_registry_calibration_path": str(args.models_registry.resolve(strict=True)),
        "models_registry_development_subtree_equal": True,
        "items": items})
    cv2.setNumThreads(2)
    model = SFaceModel(args.yunet, args.sface)
    threshold = Calibration(**cal["calibration"])
    results = []
    for row in items:
        started = perf_counter()
        identity = row["identity"]
        references, source_image, source_box = [], None, None
        evidence = {}
        for view in VIEWS:
            path = groups[identity][view]
            image = decode_image(path)
            probe = model.embed(image)
            evidence[view] = {"status": probe.status, "reason": probe.reason,
                              "detection_count": probe.detection_count,
                              "sha256": digest(path)}
            if probe.status == "valid":
                references.append(Reference(f"{identity}:{view}", identity, probe.feature))
            if view == "neutral_front":
                source_image, source_box = image, probe.selected_box
        controls = {}
        if source_box is not None and len(references) == len(VIEWS):
            controls = evaluate_conditions(export_jpeg(source_image), source_box, model,
                                           references, threshold, identity, CONDITIONS,
                                           save_dir=args.output / identity / row["arm"] / "control")
        eligible = (len(references) == len(VIEWS) and len(controls) == len(CONDITIONS)
                    and all(value["status"] == "valid" and value.get("own_identity_matched") is True
                            for value in controls.values()))
        candidate = {}
        if eligible:
            path = Path(row["path"])
            if digest(path) != row["sha256"]:
                raise ValueError("Candidate changed after preflight")
            image = decode_image(path)
            if image.shape != source_image.shape:
                raise ValueError("Candidate dimensions differ from clean source")
            candidate = evaluate_conditions(path.read_bytes(), source_box, model,
                                            references, threshold, identity, CONDITIONS,
                                            save_dir=args.output / identity / row["arm"] / "candidate")
        results.append({"identity": identity, "method": row["method"], "arm": row["arm"],
                        "candidate_path": row["path"], "candidate_sha256": row["sha256"],
                        "reference_evidence": evidence, "eligible": eligible,
                        "controls": controls, "control_summary": summarize(controls),
                        "candidate": candidate, "candidate_summary": summarize(candidate),
                        "seconds": perf_counter() - started})
        write_json(args.output / "progress.json", {"items": results})
        print(f"SFace frozen score {len(results)}/{len(items)} {identity}/{row['arm']}", flush=True)
    write_json(args.output / "results.json", {
        "status": "complete", "purpose": "development_sface_frozen_jpeg_scoring_only",
        "preflight_sha256": digest(args.output / "preflight.json"),
        "threshold": threshold.threshold, "conditions": list(CONDITIONS),
        "gallery_views": list(VIEWS), "items": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "manifest", "calibration", "models-registry", "yunet", "sface", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
