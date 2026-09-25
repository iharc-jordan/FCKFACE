"""Bounded H4 source-only search using a frozen SFace identity subspace.

Two prespecified development photos, four subspace arms, RMS 16, 64 proposals
per arm. Only neutral_front may be decoded before selected JPEGs and coefficients
are frozen. Full native SFace gallery scoring follows the freeze. No held-out
identity, recognizer, or claim of privacy effectiveness enters this experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.evaluation import Reference
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.recognition import SFaceModel
from identity_subspace import EXCLUDED_SCREEN_IDS
from optimize_regions import (Artwork, CONDITIONS, OBJECTIVE_CONDITIONS, SEED,
                              VIEWS, append_json, contact_sheet,
                              evaluate_conditions, write_json)
from optimize_single_photo import freeze_output


IDENTITIES = ("frll-001", "frll-003")
TARGET_RMS = 16.0
MAX_PROPOSALS = 64
ARMS = (("learned8", "learned", 8), ("random8", "random", 8),
        ("learned16", "learned", 16), ("random16", "random", 16))


def verify_input_images(manifest: dict, groups: dict) -> None:
    expected = {(row["identity"], row["view"]): row["sha256"]
                for row in manifest["images"]
                if row["identity"] in IDENTITIES and row["view"] in VIEWS}
    for identity in IDENTITIES:
        for view in VIEWS:
            if ((identity, view) not in expected or view not in groups[identity]
                    or digest(groups[identity][view]) != expected[(identity, view)]):
                raise ValueError(f"FRLL image differs from manifest: {identity}:{view}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_project(feature: np.ndarray, mean: np.ndarray,
                   directions: np.ndarray) -> np.ndarray:
    """No epsilon denominator: zero, nonfinite, or malformed projections fail."""
    feature = np.asarray(feature, dtype=np.float64).reshape(-1)
    if (feature.shape != (128,) or mean.shape != (128,)
            or directions.ndim != 2 or directions.shape[0] != 128
            or not np.isfinite(feature).all() or not np.isfinite(mean).all()
            or not np.isfinite(directions).all()):
        raise ValueError("Invalid 128D SFace feature or projection")
    projection = (feature - mean) @ directions
    norm = float(np.linalg.norm(projection))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Zero or nonfinite projected embedding")
    return projection / norm


def preflight(args):
    source_root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if out == source_root or out.is_relative_to(source_root):
        raise ValueError("Private experiment output must stay outside repository")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty H4 output directory: {out}")
    groups = grouped_images(args.manifest, "development")
    if tuple(sorted(groups)[:2]) != IDENTITIES:
        raise ValueError("First two frozen screen development identities changed")
    calibration_artifact = json.loads(args.calibration.read_text(encoding="utf-8"))
    if (calibration_artifact.get("status") != "complete" or
            calibration_artifact.get("purpose") != "development_native_sface_calibration_only"):
        raise ValueError("Invalid native SFace calibration")
    pipeline = calibration_artifact["pipeline"]
    if digest(args.manifest) != calibration_artifact["dataset"]["manifest_sha256"]:
        raise ValueError("Calibration dataset manifest differs")
    verify_input_images(json.loads(args.manifest.read_text(encoding="utf-8")), groups)
    for name, path in (("yunet", args.yunet), ("sface", args.sface)):
        if digest(path) != pipeline["model_artifacts"][name]["sha256"]:
            raise ValueError(f"Calibration {name} model hash differs")
    for name in ("recognition.py", "imaging.py", "evaluation.py"):
        path = source_root / "research" / "fckface_lab" / name
        if digest(path) != pipeline["source_sha256"][f"fckface_lab/{name}"]:
            raise ValueError(f"Calibration preprocessing source changed: {name}")
    detector = pipeline["detector_input"]
    if detector["maximum_side_px"] != 640 or detector["score_threshold"] != .9:
        raise ValueError("Native detector settings differ from calibration")
    sub_report = json.loads(args.subspace_report.read_text(encoding="utf-8"))
    if digest(args.subspace) != sub_report["subspace_sha256"]:
        raise ValueError("Subspace matrix file changed")
    if (sub_report["source_sha256"]["manifest"] != digest(args.manifest)
            or sub_report["source_sha256"]["calibration"] != digest(args.calibration)
            or sub_report["source_sha256"]["yunet"] != digest(args.yunet)
            or sub_report["source_sha256"]["sface"] != digest(args.sface)
            or set(EXCLUDED_SCREEN_IDS) & set(sub_report["included_identity_ids"])
            or digest(source_root / "research" / "identity_subspace.py") !=
            sub_report["source_sha256"]["identity_subspace.py"]):
        raise ValueError("Subspace training or model provenance conflicts with H4")
    if not set(EXCLUDED_SCREEN_IDS).issubset(set(sub_report["excluded_screen_ids"])):
        raise ValueError("Subspace did not exclude all eight screened identities")
    with np.load(args.subspace, allow_pickle=False) as stored:
        matrices = {name: stored[name].astype(np.float64).copy()
                    for name in ("mean", "learned_8", "random_8", "learned_16", "random_16")}
    if (matrices["mean"].shape != (128,) or
            any(matrices[f"{kind}_{rank}"].shape != (128, rank)
                for kind in ("learned", "random") for rank in (8, 16)) or
            not all(np.isfinite(value).all() for value in matrices.values())):
        raise ValueError("Invalid saved subspace arrays")
    cv2.setNumThreads(2)
    return out, groups, Calibration(**calibration_artifact["calibration"]), matrices


def source_objective(art, coeff, *, model, source_feature, projected_source,
                     mean, directions, cache):
    image, distortion = art.render(coeff, TARGET_RMS, spatial=False)
    jpeg = export_jpeg(image)
    key = hashlib.sha256(jpeg).hexdigest()
    if not distortion["rms_matched"]:
        return 1.0, {"reason": "rms_unattainable", "distortion": distortion}, jpeg, False
    cached = key in cache
    if not cached:
        variants = make_variants(jpeg, art.box_xyxy)
        conditions = {}
        for name in OBJECTIVE_CONDITIONS:
            variant = variants[name]
            if variant.status != "valid":
                conditions[name] = {"status": "inconclusive", "reason": variant.reason}
                continue
            probe = model.embed(variant.image, expected_box=variant.target_box)
            conditions[name] = {"status": probe.status, "reason": probe.reason,
                                "detection_count": probe.detection_count,
                                "feature": probe.feature}
        cache[key] = conditions
    conditions = cache[key]
    evidence_conditions = {}
    values = []
    for name in OBJECTIVE_CONDITIONS:
        row = conditions[name]
        view = {"status": row["status"], "reason": row["reason"],
                "detection_count": row.get("detection_count")}
        if row["status"] == "valid" and row["feature"] is not None:
            try:
                full_cosine = float(np.dot(source_feature, row["feature"]))
                projected = strict_project(row["feature"], mean, directions)
                projected_cosine = float(np.dot(projected_source, projected))
                combined = .5 * full_cosine + .5 * projected_cosine
                if not all(np.isfinite(value) for value in
                           (full_cosine, projected_cosine, combined)):
                    raise ValueError("Nonfinite cosine objective")
                view.update({"full_source_cosine": full_cosine,
                             "projected_source_cosine": projected_cosine,
                             "half_full_plus_half_projected": combined})
                values.append(combined)
            except ValueError as exc:
                view.update({"status": "inconclusive", "reason": str(exc)})
        evidence_conditions[name] = view
    if len(values) != len(OBJECTIVE_CONDITIONS):
        score = 1.0
        reason = "invalid_or_inconclusive_condition_or_projection"
    else:
        score = max(values)
        reason = None
    evidence = {"reason": reason, "distortion": distortion,
                "conditions": evidence_conditions}
    return score, evidence, jpeg, cached


def search(art, *, arm, model, source_feature, projected_source,
           mean, directions, cache, query_log, identity):
    # Same 16 signs, random restarts, adaptive scales, and seed as
    # optimize_single_photo.py. Each arm receives all 64 proposal slots.
    rng = np.random.default_rng(SEED)
    best = None
    unique = 0
    for index in range(MAX_PROPOSALS):
        if index < 16:
            coeff = np.array([1 if index & (1 << bit) else -1
                              for bit in range(4)], dtype=np.float32)
        elif index % 8 == 0:
            coeff = rng.uniform(-1, 1, size=4).astype(np.float32)
        else:
            current = np.asarray(best["coefficients"] if best else [1, 1, 1, 1])
            scale = (.7, .4, .2, .1)[((index - 16) // 12) % 4]
            coeff = np.clip(current + rng.normal(0, scale, size=4), -1, 1).astype(np.float32)
        began = perf_counter()
        score, evidence, jpeg, cached = source_objective(
            art, coeff, model=model, source_feature=source_feature,
            projected_source=projected_source, mean=mean,
            directions=directions, cache=cache)
        row = {"identity": identity, "arm": arm, "query_index": index,
               "coefficients": [float(value) for value in coeff],
               "objective_worst_half_full_half_projected": score,
               "evidence": evidence,
               "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
               "native_feature_cache_hit": cached,
               "seconds": round(perf_counter() - began, 4)}
        append_json(query_log, row)
        unique += not cached
        if best is None or score < best["objective_worst_half_full_half_projected"]:
            best = row
    return best, {"proposals": MAX_PROPOSALS, "new_native_feature_sets": unique,
                  "maximum_budget": MAX_PROPOSALS}


def run(args):
    started = perf_counter()
    out, groups, calibration, matrices = preflight(args)
    out.mkdir(parents=True, exist_ok=True)
    model = SFaceModel(args.yunet, args.sface)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "research/identity_subspace.py",
               root / "research/optimize_single_photo.py",
               root / "research/optimize_regions.py",
               *sorted((root / "research/fckface_lab").glob("*.py"))]
    header = {
        "scope": "H4 open two-ID source-only development optimization; prior full-only aggregate results already known; no transfer claim",
        "started_utc": now(), "identities": IDENTITIES,
        "stage1_allowed_view": "neutral_front only",
        "stage2_views": VIEWS,
        "stage_barrier": "selection-freeze.json before new same-person gallery images/features are read; prior full-only aggregate results were already known",
        "arms": [arm for arm, _, _ in ARMS],
        "target_face_rms": TARGET_RMS,
        "objective": "max over export/jpeg75_420/blur of half normalized full source cosine plus half strictly normalized projected source cosine; invalid=1",
        "objective_conditions": OBJECTIVE_CONDITIONS, "final_conditions": CONDITIONS,
        "proposal_schedule": "16 exhaustive sign seeds, seed0 random restart every eighth later proposal, adaptive Gaussian scales .7/.4/.2/.1; exactly 64 slots per arm",
        "maximum_proposals_per_arm": MAX_PROPOSALS, "seed": SEED,
        "subspace_training_identities_exclude_screen": True,
        "calibration_threshold": calibration.threshold,
        "versions": {"python": sys.version, "opencv": cv2.__version__,
                     "numpy": np.__version__, "pillow": Image.__version__,
                     "cv2_threads": cv2.getNumThreads()},
        "sha256": {"manifest": digest(args.manifest),
                   "calibration": digest(args.calibration),
                   "yunet": digest(args.yunet), "sface": digest(args.sface),
                   "subspace": digest(args.subspace),
                   "subspace_report": digest(args.subspace_report),
                   **{str(path.relative_to(root)): digest(path) for path in sources}},
    }
    write_json(out / "header.json", header)
    query_log = out / "queries.jsonl"
    run_log = out / "run-log.jsonl"
    selected = {}
    source_state = {}
    try:
        # STAGE 1: only neutral_front images and their own SFace features.
        for identity in IDENTITIES:
            began = perf_counter()
            source_path = groups[identity]["neutral_front"]
            source = decode_image(source_path)
            detection = model.embed(source)
            if detection.status != "valid":
                raise ValueError(f"Source embedding inconclusive: {identity}: {detection.reason}")
            feature = detection.feature.astype(np.float64)
            art = Artwork(source, detection.selected_box, detection.landmarks)
            source_state[identity] = {"source": source, "detection": detection,
                                      "source_path": source_path}
            image_cache = {}
            choices = {}
            budgets = {}
            for arm, kind, rank in ARMS:
                matrix = matrices[f"{kind}_{rank}"]
                projected_source = strict_project(feature, matrices["mean"], matrix)
                winner, budget = search(
                    art, arm=arm, model=model, source_feature=feature,
                    projected_source=projected_source, mean=matrices["mean"],
                    directions=matrix, cache=image_cache, query_log=query_log,
                    identity=identity)
                frozen = freeze_output(
                    art, winner["coefficients"], spatial=False,
                    path=out / "stage1" / identity / f"{arm}.jpg")
                if frozen["jpeg_sha256"] != winner["jpeg_sha256"]:
                    raise ValueError("Winner JPEG differed on deterministic replay")
                frozen["selected_query_index"] = winner["query_index"]
                frozen["source_only_objective"] = winner["objective_worst_half_full_half_projected"]
                frozen["objective_evidence"] = winner["evidence"]
                choices[arm] = frozen
                budgets[arm] = budget
            selected[identity] = {
                "source_view": "neutral_front", "source_image_sha256": digest(source_path),
                "source_box_xyxy": detection.selected_box,
                "source_detection_count": detection.detection_count,
                "selected": choices, "query_budgets": budgets,
                "stage1_seconds": round(perf_counter() - began, 3),
            }
            append_json(run_log, {"event": "stage1_identity_complete",
                                  "identity": identity,
                                  "seconds": selected[identity]["stage1_seconds"],
                                  "at_utc": now()})
            print(identity, "four H4 selections frozen in memory",
                  selected[identity]["stage1_seconds"], "seconds", flush=True)

        freeze = {"stage": "source_only_selection_frozen", "frozen_utc": now(),
                  "identities": IDENTITIES,
                  "gallery_images_decoded_or_embedded_before_freeze": [],
                  "source_views_read": {identity: ["neutral_front"] for identity in IDENTITIES},
                  "query_log_sha256": digest(query_log), "selected": selected,
                  "input_sha256": header["sha256"]}
        freeze_path = out / "selection-freeze.json"
        write_json(freeze_path, freeze)
        freeze_sha = digest(freeze_path)
        append_json(run_log, {"event": "selection_frozen", "sha256": freeze_sha,
                              "at_utc": now()})
        print("H4 selections frozen before gallery access:", freeze_sha, flush=True)

        # STAGE 2: native full-space score against four clean references.
        baseline = json.loads(args.full_only_results.read_text(encoding="utf-8"))
        if baseline.get("status") != "complete":
            raise ValueError("Frozen full-only source baseline is incomplete")
        baseline_cases = {case["identity"]: case for case in baseline["cases"]}
        if set(IDENTITIES) - set(baseline_cases):
            raise ValueError("Full-only baseline lacks H4 identities")
        cases = []
        for identity in IDENTITIES:
            state = source_state[identity]
            detection = state["detection"]
            refs = [Reference(f"{identity}:neutral_front", identity,
                              detection.feature)]
            gallery_inputs = {"neutral_front": {"sha256": digest(state["source_path"]),
                                                "read_before_freeze": True}}
            for view in VIEWS[1:]:
                path = groups[identity][view]
                embedded = model.embed(decode_image(path))
                gallery_inputs[view] = {"sha256": digest(path),
                                        "read_before_freeze": False,
                                        "status": embedded.status,
                                        "reason": embedded.reason}
                if embedded.status == "valid":
                    refs.append(Reference(f"{identity}:{view}", identity,
                                          embedded.feature))
            if len(refs) != 4:
                cases.append({"identity": identity, "status": "inconclusive_gallery",
                              "gallery_inputs": gallery_inputs})
                continue
            source_box = detection.selected_box
            clean = evaluate_conditions(
                export_jpeg(state["source"]), source_box, model, refs,
                calibration, identity, CONDITIONS,
                save_dir=out / "stage2" / identity / "clean")
            control_eligible = all(row["status"] == "valid" and
                                   row["own_identity_matched"] is True
                                   for row in clean.values())
            winners = {}
            for arm, _, _ in ARMS:
                frozen = selected[identity]["selected"][arm]
                jpeg_path = Path(frozen["jpeg_path"])
                if digest(jpeg_path) != frozen["jpeg_sha256"]:
                    raise ValueError("Selected exact JPEG changed after freeze")
                rows = evaluate_conditions(
                    jpeg_path.read_bytes(), source_box, model, refs,
                    calibration, identity, CONDITIONS,
                    save_dir=out / "stage2" / identity / arm)
                valid = all(row["status"] == "valid" for row in rows.values())
                winners[arm] = {
                    "coefficients": frozen["coefficients"],
                    "coefficients_sha256": frozen["coefficients_sha256"],
                    "export_jpeg_sha256": frozen["jpeg_sha256"],
                    "selected_query_index": frozen["selected_query_index"],
                    "source_only_objective": frozen["source_only_objective"],
                    "all_conditions_valid": valid,
                    "all_conditions_nonmatching": valid and
                    all(not row["matching_references"] for row in rows.values()),
                    "worst_condition_cosine":
                    max(row["maximum_cosine"] for row in rows.values()) if valid else None,
                    "conditions": rows, "distortion": frozen["distortion"],
                }
            baseline_case = baseline_cases[identity]
            old = baseline_case["source_only_selected"]["joint_regions"]
            if digest(args.full_only_results.parent / "stage2" / identity /
                      "joint_regions" / "export.jpg") != old["export_jpeg_sha256"]:
                raise ValueError("Frozen full-only baseline JPEG changed")
            full_only = {"coefficients": old["coefficients"],
                         "export_jpeg_sha256": old["export_jpeg_sha256"],
                         "worst_condition_cosine": old["worst_condition_cosine"],
                         "all_conditions_valid": old["all_conditions_valid"],
                         "all_conditions_nonmatching": old["all_conditions_nonmatching"],
                         "conditions": old["conditions"]}
            sheet_rows = [("clean", out / "stage2" / identity / "clean" / "export-face.png"),
                          ("full-only", args.full_only_results.parent / "stage2" /
                           identity / "joint_regions" / "export-face.png")]
            sheet_rows += [(arm, out / "stage2" / identity / arm / "export-face.png")
                           for arm, _, _ in ARMS]
            sheet = out / "sheets" / f"{identity}-rms16.png"
            contact_sheet(sheet_rows, sheet)
            cases.append({"identity": identity, "status": "evaluated",
                          "gallery_inputs": gallery_inputs,
                          "clean_control_eligible": control_eligible,
                          "clean_control": clean, "subspace_selected": winners,
                          "frozen_full_only_joint_baseline": full_only,
                          "contact_sheet": str(sheet.relative_to(out)),
                          "contact_sheet_sha256": digest(sheet)})
            print(identity, "H4 final",
                  [(arm, row["worst_condition_cosine"],
                    row["all_conditions_nonmatching"]) for arm, row in winners.items()],
                  flush=True)
        result = {"status": "complete", "completed_utc": now(),
                  "selection_freeze_sha256": freeze_sha,
                  "selection_freeze_path": str(freeze_path),
                  "full_only_baseline_sha256": digest(args.full_only_results),
                  "source_access_stage": "neutral_front only before freeze; other gallery views opened after freeze; prior full-only aggregate results were already known",
                  "cases": cases, "wall_seconds": round(perf_counter() - started, 3),
                  "query_proposals": sum(1 for _ in query_log.open(encoding="utf-8")),
                  "query_log_sha256": digest(query_log),
                  "run_log_sha256": digest(run_log)}
        write_json(out / "results.json", result)
        print("H4 bounded screen complete:", out, result["wall_seconds"], "seconds",
              flush=True)
    except Exception as exc:
        append_json(run_log, {"event": "error", "type": type(exc).__name__,
                              "message": str(exc), "at_utc": now()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "calibration", "yunet", "sface", "subspace",
                 "subspace-report", "full-only-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
