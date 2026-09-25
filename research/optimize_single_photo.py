"""Source-only H2b ablation on fixed pilot or confirmation development faces.

Stage 1 may decode/embed only each neutral-front query. It freezes coefficients
and exact exported JPEG hashes before stage 2 opens any other clean view for
gallery evaluation. This is SFace development evidence, not transfer validation.
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
from fckface_lab.evaluation import Reference, evaluate_gallery
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.recognition import SFaceModel
from optimize_regions import (Artwork, CONDITIONS, OBJECTIVE_CONDITIONS, SEED,
                              VIEWS, append_json, contact_sheet,
                              evaluate_conditions, write_json)

TARGET_RMS = 16.0
ARMS = (("joint_regions", False), ("whole_face_2x2", True))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def single_source_objective(art, coeff, *, spatial, model, source_reference,
                            calibration, identity, cache):
    image, distortion = art.render(coeff, TARGET_RMS, spatial=spatial)
    jpeg = export_jpeg(image)
    sha = hashlib.sha256(jpeg).hexdigest()
    if sha in cache:
        score, evidence = cache[sha]
        return score, evidence, jpeg, True
    if not distortion["rms_matched"]:
        score, evidence = 1.0, {"reason": "rms_unattainable", "distortion": distortion}
    else:
        variants = make_variants(jpeg, art.box_xyxy)
        condition_rows = {}
        for name in OBJECTIVE_CONDITIONS:
            variant = variants[name]
            row = {"status": variant.status, "reason": variant.reason}
            if variant.status == "valid":
                probe = model.embed(variant.image, expected_box=variant.target_box)
                gallery = evaluate_gallery(
                    probe, (source_reference,), calibration, own_identity=identity,
                    own_source_name=source_reference.name,
                    expected_reference_names=(source_reference.name,),
                )
                row.update({"status": gallery.status, "reason": gallery.reason,
                            "source_cosine": gallery.own_source_score,
                            "detection_count": probe.detection_count})
            condition_rows[name] = row
        values = [condition_rows[name].get("source_cosine") for name in OBJECTIVE_CONDITIONS]
        if any(value is None for value in values):
            score = 1.0
            reason = "invalid_or_inconclusive_condition"
        else:
            score = max(values)
            reason = None
        evidence = {"reason": reason, "distortion": distortion,
                    "conditions": condition_rows}
    cache[sha] = (score, evidence)
    return score, evidence, jpeg, False


def search(art, *, spatial, arm, model, source_reference,
           calibration, identity, cache, log):
    """Same 16 sign seeds, adaptive proposals, and exact-JPEG cache as H2b."""
    rng = np.random.default_rng(SEED)
    best = None
    unique = 0
    reused_in_row = 0
    proposals = 0
    for index in range(64):
        if index < 16:
            coeff = np.array([1 if index & (1 << bit) else -1
                              for bit in range(4)], dtype=np.float32)
        elif index % 8 == 0:
            coeff = rng.uniform(-1, 1, size=4).astype(np.float32)
        else:
            current = np.asarray(best["coefficients"] if best else [1, 1, 1, 1])
            scale = (.7, .4, .2, .1)[((index-16)//12) % 4]
            coeff = np.clip(current + rng.normal(0, scale, size=4), -1, 1).astype(np.float32)
        start = perf_counter()
        score, evidence, jpeg, cached = single_source_objective(
            art, coeff, spatial=spatial, model=model,
            source_reference=source_reference, calibration=calibration,
            identity=identity, cache=cache)
        row = {"identity": identity, "arm": arm, "query_index": index,
               "coefficients": [float(value) for value in coeff],
               "source_only_objective": score, "evidence": evidence,
               "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
               "cache_hit": cached, "seconds": round(perf_counter()-start, 4)}
        append_json(log, row)
        unique += not cached
        proposals += 1
        reused_in_row = reused_in_row + 1 if cached else 0
        if best is None or score < best["source_only_objective"]:
            best = row
        if reused_in_row >= 8:
            break
    return best, {"proposals": proposals, "unique_objective_calls": unique,
                  "maximum_budget": 64}


def freeze_output(art, coeff, *, spatial, path):
    image, distortion = art.render(np.asarray(coeff, dtype=np.float32),
                                   TARGET_RMS, spatial=spatial)
    payload = export_jpeg(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(payload)
    coefficients = list(map(float, coeff))
    return {"jpeg_path": str(path), "jpeg_sha256": digest(path),
            "coefficients": coefficients,
            "coefficients_sha256": hashlib.sha256(
                json.dumps(coefficients, separators=(",", ":")).encode()).hexdigest(),
            "distortion": distortion}


def run(args):
    start = perf_counter()
    source_root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if out == source_root or out.is_relative_to(source_root):
        raise ValueError("Output must stay outside the repository")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory")
    groups = grouped_images(args.manifest, "development")
    selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
    first_eight = sorted(groups)[:8]
    if (selection.get("identities") != first_eight
            or selection.get("source_view") != "neutral_front"
            or selection.get("reference_views") != list(VIEWS)
            or selection.get("processing_conditions") != list(CONDITIONS)):
        raise ValueError("Frozen first-eight selection differs from the manifest or pipeline")
    expected_confirmation = ("frll-004", "frll-007", "frll-011",
                             "frll-013", "frll-018", "frll-019")
    if tuple(selection["identities"][2:8]) != expected_confirmation:
        raise ValueError("Frozen confirmation IDs differ from the prespecified six")
    identities = selection["identities"][:2] if args.subset == "pilot" else list(expected_confirmation)
    if args.subset == "pilot" and args.oracle_results is None:
        raise ValueError("Pilot comparison requires --oracle-results")
    artifact = json.loads(args.calibration.read_text(encoding="utf-8"))
    if artifact["status"] != "complete" or artifact["dataset"]["split"] != "calibration":
        raise ValueError("Invalid calibration artifact")
    calibration = Calibration(**artifact["calibration"])
    if digest(args.manifest) != artifact["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from frozen calibration")
    for name in ("recognition.py", "imaging.py", "evaluation.py", "datasets.py"):
        local = source_root / "research/fckface_lab" / name
        if digest(local) != artifact["pipeline"]["source_sha256"][f"fckface_lab/{name}"]:
            raise ValueError(f"{name} differs from frozen calibration pipeline")
    for name, path in (("yunet", args.yunet), ("sface", args.sface)):
        if digest(path) != artifact["pipeline"]["model_artifacts"][name]["sha256"]:
            raise ValueError(f"{name} model differs from calibration")
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    model = SFaceModel(args.yunet, args.sface)
    source_files = [Path(__file__), source_root / "research/optimize_regions.py",
                    *sorted((source_root / "research/fckface_lab").glob("*.py"))]
    header = {"schema_version": 1, "scope": "source-only SFace development confirmation; method and IDs were development-influenced, not blinded",
              "started_utc": now(), "subset": args.subset, "identities": identities,
              "stage1_allowed_image_view": "neutral_front only",
              "stage1_allowed_reference_count": 1,
              "stage2_gallery_views": VIEWS,
              "stage_barrier": "selection-freeze.json must exist before any other same-person image is decoded or embedded",
              "target_face_rms": TARGET_RMS,
              "objective_conditions": OBJECTIVE_CONDITIONS,
              "final_conditions": CONDITIONS,
              "arms": [arm for arm, _ in ARMS], "maximum_proposals_per_arm": 64,
              "seed": SEED, "calibration_threshold": calibration.threshold,
              "runtime_note": "Native local CPU runtime; it does not establish browser or phone processing time",
              "versions": {"python": sys.version, "opencv": cv2.__version__,
                           "numpy": np.__version__, "pillow": Image.__version__,
                           "cv2_threads": cv2.getNumThreads()},
              "sha256": {"manifest": digest(args.manifest),
                         "selection": digest(args.selection),
                         "calibration": digest(args.calibration),
                         "yunet": digest(args.yunet), "sface": digest(args.sface),
                         **{str(path.relative_to(source_root)): digest(path)
                            for path in source_files}}}
    write_json(out / "header.json", header)
    query_log = out / "queries.jsonl"
    run_log = out / "run-log.jsonl"
    selected = {}
    source_cache = {}
    try:
        # STAGE 1. Do not decode or embed any view other than neutral_front.
        for identity in identities:
            identity_start = perf_counter()
            source_path = groups[identity]["neutral_front"]
            source = decode_image(source_path)
            source_result = model.embed(source)
            if source_result.status != "valid":
                raise ValueError(f"Source embedding inconclusive: {identity}: {source_result.reason}")
            source_reference = Reference(f"{identity}:neutral_front", identity,
                                         source_result.feature)
            art = Artwork(source, source_result.selected_box, source_result.landmarks)
            source_cache[identity] = {"source": source, "detection": source_result,
                                      "source_path": source_path,
                                      "source_reference": source_reference}
            cache = {}
            choices = {}
            if args.subset == "pilot":
                choices["initial_baseline"] = freeze_output(
                    art, [1, 1, 1, 1], spatial=False,
                    path=out / "stage1" / identity / "initial_baseline.jpg")
            budgets = {}
            for arm, spatial in ARMS:
                winner, budget = search(
                    art, spatial=spatial, arm=arm, model=model,
                    source_reference=source_reference, calibration=calibration,
                    identity=identity, cache=cache, log=query_log)
                frozen = freeze_output(
                    art, winner["coefficients"], spatial=spatial,
                    path=out / "stage1" / identity / f"{arm}.jpg")
                frozen["selected_query_index"] = winner["query_index"]
                frozen["source_only_objective"] = winner["source_only_objective"]
                choices[arm] = frozen
                budgets[arm] = budget
            selected[identity] = {"source_view": "neutral_front",
                                  "source_image_sha256": digest(source_path),
                                  "source_box_xyxy": source_result.selected_box,
                                  "source_detection_count": source_result.detection_count,
                                  "selected": choices, "query_budgets": budgets,
                                  "stage1_seconds": round(perf_counter()-identity_start, 3)}
            append_json(run_log, {"event": "stage1_identity_complete", "identity": identity,
                                  "seconds": selected[identity]["stage1_seconds"],
                                  "at_utc": now()})
            print(identity, "source-only selection frozen in memory",
                  selected[identity]["stage1_seconds"], "seconds", flush=True)

        # Immutable selection barrier: no remaining gallery image has been opened.
        freeze = {"schema_version": 1, "stage": "source_only_selection_frozen",
                  "frozen_utc": now(), "identities": identities,
                  "gallery_images_decoded_or_embedded_before_freeze": [],
                  "source_views_read": {identity: ["neutral_front"] for identity in identities},
                  "query_log_sha256": digest(query_log), "selected": selected,
                  "input_sha256": header["sha256"]}
        freeze_path = out / "selection-freeze.json"
        write_json(freeze_path, freeze)
        freeze_sha = digest(freeze_path)
        append_json(run_log, {"event": "selection_frozen", "sha256": freeze_sha,
                              "at_utc": now()})
        print("Selection freeze written before gallery access:", freeze_sha, flush=True)

        # STAGE 2. Extra gallery views become accessible now. Only the pilot
        # reads its preserved oracle record, after the selection barrier.
        oracle_cases = {}
        if args.subset == "pilot":
            oracle = json.loads(args.oracle_results.read_text(encoding="utf-8"))
            oracle_cases = {(case["identity"], case["target_rms"]): case
                            for case in oracle["cases"]}
        cases = []
        for identity in identities:
            stage2_start = perf_counter()
            state = source_cache[identity]
            detection = state["detection"]
            refs = [state["source_reference"]]
            gallery_inputs = {"neutral_front": {"sha256": digest(state["source_path"]),
                                                "read_before_freeze": True}}
            for view in VIEWS[1:]:
                path = groups[identity][view]
                pixels = decode_image(path)
                embedding = model.embed(pixels)
                gallery_inputs[view] = {"sha256": digest(path),
                                        "read_before_freeze": False,
                                        "status": embedding.status,
                                        "reason": embedding.reason}
                if embedding.status == "valid":
                    refs.append(Reference(f"{identity}:{view}", identity,
                                          embedding.feature))
            if len(refs) != 4:
                append_json(run_log, {"event": "gallery_inconclusive", "identity": identity,
                                      "gallery_inputs": gallery_inputs, "at_utc": now()})
                cases.append({"identity": identity, "status": "excluded_clean_ineligible",
                              "clean_control_eligible": False,
                              "reason": "missing_or_invalid_clean_gallery_reference",
                              "gallery_inputs": gallery_inputs,
                              "stage1_seconds": selected[identity]["stage1_seconds"],
                              "stage2_seconds": round(perf_counter()-stage2_start, 3)})
                continue
            source_box = detection.selected_box
            clean = evaluate_conditions(
                export_jpeg(state["source"]), source_box, model, refs, calibration,
                identity, CONDITIONS, save_dir=out / "stage2" / identity / "clean")
            control_eligible = all(row["status"] == "valid" and
                                   row["own_identity_matched"] is True
                                   for row in clean.values())
            winners = {}
            arms = (("initial_baseline",) if args.subset == "pilot" else ()) + tuple(
                arm for arm, _ in ARMS)
            for arm in arms:
                frozen = selected[identity]["selected"][arm]
                jpeg_path = Path(frozen["jpeg_path"])
                if digest(jpeg_path) != frozen["jpeg_sha256"]:
                    raise ValueError("Selected JPEG changed after selection freeze")
                rows = evaluate_conditions(
                    jpeg_path.read_bytes(), source_box, model, refs, calibration,
                    identity, CONDITIONS,
                    save_dir=out / "stage2" / identity / arm)
                valid = all(row["status"] == "valid" for row in rows.values())
                outcome = ("excluded_clean_ineligible" if not control_eligible else
                           "inconclusive_edited" if not valid else
                           "nonmatch_all_conditions" if all(
                               not row["matching_references"] for row in rows.values())
                           else "matched")
                winners[arm] = {"coefficients": frozen["coefficients"],
                                "export_jpeg_sha256": frozen["jpeg_sha256"],
                                "source_only_objective": frozen.get("source_only_objective"),
                                "all_conditions_valid": valid,
                                "all_conditions_nonmatching": valid and
                                all(not row["matching_references"] for row in rows.values()),
                                "outcome": outcome,
                                "worst_condition_cosine":
                                max(row["maximum_cosine"] for row in rows.values())
                                if valid else None,
                                "conditions": rows,
                                "distortion": frozen["distortion"]}
            oracle_arms = {}
            if args.subset == "pilot":
                old_case = oracle_cases[(identity, TARGET_RMS)]
                oracle_arms = {row["arm"]: {"coefficients": row["coefficients"],
                                          "worst_condition_cosine": row["worst_condition_cosine"],
                                          "all_conditions_nonmatching": row["all_conditions_nonmatching"],
                                          "export_jpeg_sha256": digest(
                                              args.oracle_results.parent / row["export_jpeg"])}
                               for row in old_case["winners"]
                               if row["arm"] in winners}
            sheet_rows = [("clean", out / "stage2" / identity / "clean" / "export-face.png")]
            sheet_rows += [("source " + arm, out / "stage2" / identity / arm /
                            "export-face.png") for arm in winners]
            if args.subset == "pilot":
                sheet_rows += [("oracle " + arm, args.oracle_results.parent / identity /
                                "rms16" / arm / "export-face.png")
                               for arm in ("joint_regions", "whole_face_2x2")]
            sheet_path = out / "sheets" / f"{identity}-rms16.png"
            contact_sheet(sheet_rows, sheet_path)
            case = {"identity": identity, "status": "evaluated",
                    "gallery_inputs": gallery_inputs,
                    "clean_control_eligible": control_eligible,
                    "clean_control": clean,
                    "source_only_selected": winners,
                    "preserved_oracle": oracle_arms,
                    "stage1_seconds": selected[identity]["stage1_seconds"],
                    "stage2_seconds": round(perf_counter()-stage2_start, 3),
                    "source_only_generation_over_one_minute":
                    selected[identity]["stage1_seconds"] > 60,
                    "contact_sheet": str(sheet_path.relative_to(out)),
                    "contact_sheet_sha256": digest(sheet_path)}
            cases.append(case)
            append_json(run_log, {"event": "stage2_identity_complete", "identity": identity,
                                  "all_nonmatching": {arm: row["all_conditions_nonmatching"]
                                                      for arm, row in winners.items()},
                                  "at_utc": now()})
            print(identity, "source-only final",
                  [(arm, row["worst_condition_cosine"], row["all_conditions_nonmatching"])
                   for arm, row in winners.items()], flush=True)
        result = {"status": "complete", "completed_utc": now(),
                  "subset": args.subset,
                  "stage1_selection_freeze_sha256": freeze_sha,
                  "selection_freeze_path": str(freeze_path),
                  "oracle_results_sha256": digest(args.oracle_results)
                  if args.subset == "pilot" else None,
                  "source_access_stage": "neutral_front only before freeze; extra gallery views after freeze; oracle results only for pilot after freeze",
                  "cases": cases, "wall_seconds": round(perf_counter()-start, 3),
                  "query_proposals": sum(1 for _ in query_log.open(encoding="utf-8")),
                  "query_log_sha256": digest(query_log),
                  "run_log_sha256": digest(run_log)}
        write_json(out / "results.json", result)
        print("Source-only ablation complete:", out, result["wall_seconds"], "seconds", flush=True)
    except Exception as exc:
        append_json(run_log, {"event": "error", "type": type(exc).__name__,
                              "message": str(exc), "at_utc": now()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("pilot", "confirm"), default="pilot")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        default=Path(__file__).with_name("screen-v1-selection.json"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--yunet", type=Path, required=True)
    parser.add_argument("--sface", type=Path, required=True)
    parser.add_argument("--oracle-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
