"""Score pre-frozen development JPEGs using frozen InsightFace ArcFace calibration.

Example: python research/score_frozen_arcface.py --candidates frozen.json
  --manifest Downloads/FCKFACE-data/frll/manifest.json
  --calibration Downloads/FCKFACE-data/runs/arcface-development-v1/calibration.json
  --output Downloads/FCKFACE-data/runs/frozen-arcface-h8

frozen.json: {"schema_version":1,"items":[{"identity":"frll-001",
 "path":"/absolute/frozen.jpg","sha256":"64 lowercase hex digits",
 "method":"dots","arm":"candidate-a","source_view":"neutral_front"}]}
Only development identities are accepted. This script performs no optimization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from time import perf_counter

import cv2
from PIL import Image

from arcface_development import ArcFaceModel, checked_result, verify_inputs, GALLERY_VIEWS, CONDITIONS
from fckface_lab.calibration import Calibration
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.evaluation import Reference
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.recognition import EmbeddingResult


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checked_inputs(args):
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.is_relative_to(repo) or output.exists():
        raise ValueError("Output must be a new directory outside Git")
    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    if (cal.get("status") != "complete" or cal.get("purpose") != "development_arcface_calibration_only"
            or cal["calibration"]["target_fmr"] != .001
            or cal["dataset"]["identity_count"] != 20):
        raise ValueError("Wrong or incomplete frozen ArcFace calibration")
    if cal["pipeline"]["provenance"] != verify_inputs():
        raise ValueError("Official model or wrapper differs from calibration")
    adapter = Path(__file__).with_name("arcface_development.py")
    if digest(adapter) != cal["pipeline"]["adapter_sha256"]:
        raise ValueError("ArcFace adapter changed after calibration")
    if digest(args.manifest) != cal["dataset"]["manifest_sha256"]:
        raise ValueError("FRLL manifest differs from calibration")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("split_version") != cal["dataset"]["split_version"]:
        raise ValueError("Frozen identity split differs")
    groups = grouped_images(args.manifest, "development")
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("items"), list) or not payload["items"]:
        raise ValueError("Expected nonempty schema_version=1 candidate items")
    keys, paths, items = set(), set(), []
    for row in payload["items"]:
        if not isinstance(row, dict) or set(row) != {
                "identity", "path", "sha256", "method", "arm", "source_view"}:
            raise ValueError("Candidate requires exactly identity,path,sha256,method,arm,source_view")
        identity, method, arm = row["identity"], row["method"], row["arm"]
        if (not isinstance(identity, str) or identity not in groups
                or manifest["identity_splits"].get(identity) != "development"
                or not set(GALLERY_VIEWS).issubset(groups[identity])
                or row["source_view"] != "neutral_front"):
            raise ValueError("Candidate lacks a development source and four reference views")
        if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value)
                   for value in (method, arm)):
            raise ValueError("Method and arm must be short path-safe labels")
        if not isinstance(row["path"], str):
            raise ValueError("Candidate path must be an absolute JPEG path")
        path = Path(row["path"])
        if not path.is_absolute() or path.suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError("Candidate must be an absolute JPEG path")
        path = path.resolve(strict=True)
        if path.is_relative_to(repo) or path.is_relative_to(output):
            raise ValueError("Candidate JPEG must be outside Git and the new output")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid candidate SHA-256")
        if digest(path) != row["sha256"]:
            raise ValueError(f"Candidate hash changed: {path}")
        with Image.open(path) as image:
            if image.format != "JPEG" or image.getexif() or image.info.get("icc_profile"):
                raise ValueError("Frozen candidate must be JPEG without EXIF/ICC")
            image.verify()
        key = (identity, method, arm)
        if key in keys or path in paths:
            raise ValueError("Duplicate candidate key or file")
        keys.add(key); paths.add(path)
        items.append({**row, "path": str(path)})
    selected_ids = {row["identity"] for row in items}
    expected = {(row["identity"], row["view"]): row["sha256"]
                for row in manifest["images"]
                if row["identity"] in selected_ids and row["view"] in GALLERY_VIEWS}
    if len(expected) != len(selected_ids) * len(GALLERY_VIEWS):
        raise ValueError("Missing frozen gallery image records")
    sources = {}
    for identity in selected_ids:
        sources[identity] = {}
        for view in GALLERY_VIEWS:
            path = groups[identity][view]
            sha = digest(path)
            if sha != expected[(identity, view)]:
                raise ValueError(f"Source/gallery image hash changed: {identity}:{view}")
            sources[identity][view] = {"path": str(path), "sha256": sha}
    return cal, groups, sources, sorted(items, key=lambda row: (
        row["identity"], row["method"], row["arm"]))


def summarize(rows: dict) -> dict:
    valid = len(rows) == len(CONDITIONS) and all(row["status"] == "valid" for row in rows.values())
    return {"valid_all": valid,
            "nonmatch_all": valid and all(row["nonmatch"] for row in rows.values()),
            "valid_conditions": sum(row["status"] == "valid" for row in rows.values()),
            "nonmatch_conditions": sum(row["nonmatch"] for row in rows.values()),
            "inconclusive_conditions": [key for key, row in rows.items() if row["status"] != "valid"],
            "worst_gallery_cosine": max((row["maximum_cosine"] for row in rows.values()
                                         if row.get("maximum_cosine") is not None), default=None)}


def safe_embed(model, image, expected_box):
    try:
        return model.embed(image, expected_box=expected_box)
    except (cv2.error, OSError, RuntimeError, ValueError) as exc:
        return EmbeddingResult("inconclusive", reason=f"processing_error:{type(exc).__name__}")


def score_conditions(jpeg: bytes, box, model, refs, threshold, identity: str,
                     directory: Path, *, variants=None, edited: bool = False) -> dict:
    variants = make_variants(jpeg, box) if variants is None else variants
    directory.mkdir(parents=True, exist_ok=False)
    rows = {}
    for condition in CONDITIONS:
        variant = variants[condition]
        row = {"condition": condition, "status": variant.status, "reason": variant.reason,
               "jpeg_path": None, "jpeg_sha256": None}
        if variant.jpeg is not None:
            saved = directory / f"{condition}.jpg"
            saved.write_bytes(variant.jpeg)
            row["jpeg_path"] = str(saved)
            row["jpeg_sha256"] = digest(saved)
        probe = (safe_embed(model, variant.image, variant.target_box)
                 if variant.status == "valid" else EmbeddingResult("inconclusive", reason=variant.reason))
        if edited and probe.status != "valid":
            probe = EmbeddingResult("inconclusive", reason=probe.reason or probe.status,
                                    detection_count=probe.detection_count)
        row.update(checked_result(probe, refs, threshold, identity))
        rows[condition] = row
    return rows


def run(args) -> None:
    cal, groups, sources, items = checked_inputs(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen_dir = output / "frozen-inputs"
    frozen_dir.mkdir()
    frozen = []
    for index, row in enumerate(items):
        copy = frozen_dir / f"{index:03d}.jpg"
        shutil.copyfile(row["path"], copy)
        if digest(copy) != row["sha256"]:
            raise ValueError("Candidate changed while freezing input bytes")
        frozen.append({**row, "frozen_copy": str(copy)})
    preflight = {"created_utc": utc(), "candidate_manifest_sha256": digest(args.candidates),
                 "calibration_sha256": digest(args.calibration),
                 "dataset_manifest_sha256": digest(args.manifest),
                 "scorer_sha256": digest(Path(__file__)),
                 "adapter_sha256": digest(Path(__file__).with_name("arcface_development.py")),
                 "model_and_upstream": cal["pipeline"]["provenance"],
                 "source_gallery": sources, "items": frozen}
    write_json(output / "preflight.json", preflight)
    cv2.setNumThreads(2)
    model = ArcFaceModel()
    threshold = Calibration(**cal["calibration"])
    results = []
    started = perf_counter()
    for index, row in enumerate(frozen):
        t0 = perf_counter()
        identity = row["identity"]
        refs, evidence, source_image, source_box = [], {}, None, None
        for view in GALLERY_VIEWS:
            path = groups[identity][view]
            if digest(path) != sources[identity][view]["sha256"]:
                raise ValueError(f"Gallery changed after preflight: {identity}:{view}")
            image = decode_image(path)
            embedded = safe_embed(model, image, None)
            evidence[view] = {"status": embedded.status, "reason": embedded.reason,
                              "detection_count": embedded.detection_count,
                              "sha256": sources[identity][view]["sha256"]}
            if embedded.status == "valid":
                refs.append(Reference(f"{identity}:{view}", identity, embedded.feature))
            if view == "neutral_front":
                source_image, source_box = image, embedded.selected_box
        base = output / identity / row["method"] / row["arm"]
        controls = {}
        if source_box is not None and len(refs) == len(GALLERY_VIEWS):
            controls = score_conditions(export_jpeg(source_image), source_box, model, refs,
                                        threshold, identity, base / "control")
        eligible = (len(refs) == len(GALLERY_VIEWS) and len(controls) == len(CONDITIONS)
                    and all(value["status"] == "valid" and value["own_identity_matched"] is True
                            for value in controls.values()))
        candidate = {}
        if eligible:
            selected = Path(row["frozen_copy"])
            if digest(selected) != row["sha256"]:
                raise ValueError("Frozen candidate copy changed")
            try:
                image = decode_image(selected)
                if image.shape != source_image.shape:
                    raise ValueError("candidate_dimensions_changed")
                blob = selected.read_bytes()
                variants = make_variants(blob, source_box)
            except (OSError, ValueError, cv2.error) as exc:
                candidate = {condition: {"status": "inconclusive",
                                         "reason": (str(exc) if str(exc) == "candidate_dimensions_changed"
                                                    else f"candidate_decode_or_variant_error:{type(exc).__name__}"),
                                         "nonmatch": False, "maximum_cosine": None,
                                         "jpeg_path": None, "jpeg_sha256": None}
                             for condition in CONDITIONS}
            else:
                candidate = score_conditions(blob, source_box, model, refs,
                                             threshold, identity, base / "candidate",
                                             variants=variants, edited=True)
        result = {"model": "ArcFace buffalo_l", "identity": identity,
                  "method": row["method"], "arm": row["arm"],
                  "candidate_path": row["path"], "candidate_sha256": row["sha256"],
                  "frozen_copy": row["frozen_copy"], "reference_evidence": evidence,
                  "eligible": eligible, "controls": controls, "control_summary": summarize(controls),
                  "candidate": candidate, "candidate_summary": summarize(candidate),
                  "seconds": perf_counter() - t0}
        results.append(result)
        write_json(output / "progress.json", {"items": results})
        print(f"ArcFace frozen score {index + 1}/{len(frozen)} {identity}/{row['method']}/{row['arm']}", flush=True)
    write_json(output / "results.json", {"status": "complete",
                                         "purpose": "development_arcface_frozen_jpeg_scoring_only",
                                         "model": "ArcFace buffalo_l",
                                         "preflight_sha256": digest(output / "preflight.json"),
                                         "threshold": threshold.threshold,
                                         "conditions": list(CONDITIONS),
                                         "gallery_views": list(GALLERY_VIEWS),
                                         "seconds": perf_counter() - started,
                                         "items": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "manifest", "calibration", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
