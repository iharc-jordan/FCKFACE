"""Development SFace score of eight pre-frozen author drawing JPEGs.

Reuses the shared seven-condition evaluator and exact native SFace calibration.
This is not an appearance acceptance or an independent transfer test.
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

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import grouped_images
from fckface_lab.evaluation import Reference
from fckface_lab.imaging import decode_image, export_jpeg
from fckface_lab.recognition import SFaceModel
from optimize_regions import CONDITIONS, VIEWS, evaluate_conditions
import render_stencil as stencil


IDS = stencil.IDS
ARMS = ("model", "model2")


def save_new_json(path: Path, obj) -> None:
    stencil.write_new(path, (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode())


def write_ghost_manifest(output: Path, items: list[dict]) -> Path:
    path = output / "frozen-ghostface-inputs.json"
    save_new_json(path, {"schema_version": 1, "items": items})
    return path


def verify_clean_export(source: Path, clean_jpeg: Path) -> None:
    if not clean_jpeg.is_file() or clean_jpeg.read_bytes() != export_jpeg(decode_image(source)):
        raise ValueError(f"Clean preview export differs from frozen source: {clean_jpeg}")


def summarize(conditions):
    valid = len(conditions) == len(CONDITIONS) and all(
        conditions[name]["status"] == "valid" for name in CONDITIONS)
    return {
        "valid_all_conditions": valid,
        "nonmatch_all_conditions": valid and all(
            conditions[name]["own_identity_matched"] is False for name in CONDITIONS),
        "nonmatching_condition_count": sum(
            conditions[name]["status"] == "valid" and
            conditions[name]["own_identity_matched"] is False for name in CONDITIONS),
        "worst_gallery_cosine": max((conditions[name]["maximum_cosine"]
                                     for name in CONDITIONS
                                     if conditions[name].get("maximum_cosine") is not None),
                                    default=None),
        "invalid_or_inconclusive": [name for name in CONDITIONS
                                     if conditions[name]["status"] != "valid"],
    }


def preflight(args):
    output = args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output == repo or output.is_relative_to(repo) or output.exists():
        raise ValueError("Scoring output must be a new directory outside Git")
    preview = json.loads(args.preview.read_text(encoding="utf-8"))
    if preview.get("status") != "preview_only_not_scored" or preview.get("candidate_count") != 8:
        raise ValueError("Expected exactly eight frozen unscored author outputs")
    rows = preview["candidates"]
    if {(r["identity"], r["author_model"]) for r in rows} != {
            (identity, name + ".pth") for identity in IDS for name in ARMS}:
        raise ValueError("Eight candidate identities/models differ from frozen protocol")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = grouped_images(args.manifest, "development")
    if not all(identity in groups and all(view in groups[identity] for view in VIEWS)
               and manifest["identity_splits"].get(identity) == "development"
               for identity in IDS):
        raise ValueError("Development source/gallery split changed")
    artifact = json.loads(args.calibration.read_text(encoding="utf-8"))
    if (artifact.get("status") != "complete" or
            artifact.get("purpose") != "development_native_sface_calibration_only" or
            stencil.digest(args.manifest) != artifact["dataset"]["manifest_sha256"]):
        raise ValueError("Wrong SFace calibration or manifest")
    pipeline = artifact["pipeline"]
    hashes = {
        "scorer": stencil.digest(Path(__file__)),
        "preview": stencil.digest(args.preview),
        "preview_protocol": stencil.digest(args.preview.parent / "protocol.json"),
        "manifest": stencil.digest(args.manifest),
        "calibration": stencil.digest(args.calibration),
        "yunet": stencil.digest(args.yunet),
        "sface": stencil.digest(args.sface),
        "recognition": stencil.digest(repo / "research/fckface_lab/recognition.py"),
        "imaging": stencil.digest(repo / "research/fckface_lab/imaging.py"),
        "evaluation": stencil.digest(repo / "research/fckface_lab/evaluation.py"),
        "conditions_helper": stencil.digest(repo / "research/optimize_regions.py"),
    }
    if hashes["preview_protocol"] != preview["protocol_sha256"]:
        raise ValueError("Frozen preview protocol changed")
    for name in ("yunet", "sface"):
        if hashes[name] != pipeline["model_artifacts"][name]["sha256"]:
            raise ValueError(f"Calibrated model artifact changed: {name}")
    for name in ("recognition", "imaging", "evaluation"):
        if hashes[name] != pipeline["source_sha256"][f"fckface_lab/{name}.py"]:
            raise ValueError(f"Calibrated preprocessing changed: {name}")
    if pipeline["detector_input"]["maximum_side_px"] != 640 or \
            pipeline["detector_input"]["score_threshold"] != .9:
        raise ValueError("Native detector settings changed")
    for row in rows:
        path = Path(row["full_photo"]).resolve()
        if not path.is_file() or not path.is_relative_to(args.preview.parent.resolve()):
            raise ValueError("Candidate JPEG not in frozen preview directory")
        if stencil.digest(path) != row["full_photo_sha256"]:
            raise ValueError(f"Frozen JPEG changed: {path}")
        source = groups[row["identity"]]["neutral_front"]
        if stencil.digest(source) != next(r["sha256"] for r in json.loads(
                (args.preview.parent / "protocol.json").read_text(encoding="utf-8"))["source_images"].values()
                if r["path"] == str(source)):
            raise ValueError("Frozen source image changed")
    for identity in IDS:
        verify_clean_export(groups[identity]["neutral_front"],
                            args.preview.parent / identity / "clean" / "export.jpg")
    return output, preview, groups, Calibration(**artifact["calibration"]), hashes


def run(args):
    started = time.perf_counter()
    output, preview, groups, calibration, hashes = preflight(args)
    cv2.setNumThreads(2)
    output.mkdir(parents=True, exist_ok=False)
    header = {"purpose": "development_native_sface_author_prior_art_screen_only",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "identities": IDS, "arms": ARMS, "conditions": CONDITIONS, "gallery_views": VIEWS,
              "threshold": calibration.threshold, "threshold_rule": "cosine >= threshold is match",
              "sha256": hashes,
              "versions": {"python": sys.version, "opencv": cv2.__version__,
                           "numpy": np.__version__}}
    save_new_json(output / "header.json", header)
    # One exact frozen candidate manifest for the separately owned GhostFaceNet
    # scorer. Both author models enter; SFace outcomes cannot select an arm.
    items = [{"identity": row["identity"], "path": str(Path(row["full_photo"]).resolve()),
              "sha256": row["full_photo_sha256"], "method": "informative_drawings",
              "arm": row["author_model"].removesuffix(".pth"),
              "source_view": "neutral_front"}
             for row in sorted(preview["candidates"],
                               key=lambda r: (r["identity"], r["author_model"]))]
    ghost_manifest = write_ghost_manifest(output, items)

    model = SFaceModel(args.yunet, args.sface)
    results = []
    for identity in IDS:
        gallery = []
        reference_status = {}
        for view in VIEWS:
            path = groups[identity][view]
            begin = time.perf_counter()
            probe = model.embed(decode_image(path))
            reference_status[view] = {"status": probe.status, "reason": probe.reason,
                                      "source_sha256": stencil.digest(path),
                                      "embedding_seconds": round(time.perf_counter() - begin, 4)}
            if probe.status == "valid":
                gallery.append(Reference(f"{identity}:{view}", identity, probe.feature))
        source_row = next(row for row in preview["candidates"] if row["identity"] == identity)
        box = tuple(source_row["source_box_xyxy"]) if "source_box_xyxy" in source_row else tuple(
            json.loads((args.preview.parent / "protocol.json").read_text(encoding="utf-8"))["source_images"][identity]["box_xyxy"])
        clean_path = args.preview.parent / identity / "clean" / "export.jpg"
        begin = time.perf_counter()
        clean = evaluate_conditions(clean_path.read_bytes(), box, model, gallery,
                                    calibration, identity, CONDITIONS,
                                    save_dir=output / identity / "clean")
        clean_seconds = time.perf_counter() - begin
        clean_summary = summarize(clean)
        eligible = len(gallery) == len(VIEWS) and clean_summary["valid_all_conditions"] and \
            all(clean[name]["own_identity_matched"] is True for name in CONDITIONS)
        candidate_rows = []
        for arm in ARMS:
            row = next(row for row in preview["candidates"]
                       if row["identity"] == identity and row["author_model"] == arm + ".pth")
            candidate_path = Path(row["full_photo"])
            begin = time.perf_counter()
            conditions = evaluate_conditions(candidate_path.read_bytes(), box, model,
                                             gallery, calibration, identity, CONDITIONS,
                                             save_dir=output / identity / arm)
            seconds = time.perf_counter() - begin
            if conditions["export"]["jpeg_sha256"] != row["full_photo_sha256"]:
                raise RuntimeError("Processed base export differs from frozen candidate")
            candidate_rows.append({"arm": arm, "candidate_jpeg": str(candidate_path),
                                   "candidate_sha256": row["full_photo_sha256"],
                                   "evaluate_seconds": round(seconds, 4),
                                   "summary": summarize(conditions),
                                   "conditions": conditions})
        results.append({"identity": identity, "clean_reference_status": reference_status,
                        "clean_eligible": eligible,
                        "clean_ineligible_reason": None if eligible else "clean_control_missing_invalid_or_unmatched",
                        "clean_evaluate_seconds": round(clean_seconds, 4),
                        "clean_summary": clean_summary, "clean_conditions": clean,
                        "candidates": candidate_rows})
        print(f"Scored {identity}; clean eligible={eligible}", flush=True)
    report = {"status": "complete", "scope": header["purpose"],
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.perf_counter() - started, 3),
              "threshold": calibration.threshold,
              "header_sha256": stencil.digest(output / "header.json"),
              "ghostface_candidate_manifest": str(ghost_manifest),
              "ghostface_candidate_manifest_sha256": stencil.digest(ghost_manifest),
              "identities": results}
    save_new_json(output / "results.json", report)
    print(f"Frozen author prior-art screen saved: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "preview", "calibration", "yunet", "sface", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
