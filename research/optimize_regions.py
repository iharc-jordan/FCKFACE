"""Two-identity development H2b screen of signed artificial artwork.

Four matched 64-query arms compare joint facial regions, separately optimized
regions combined afterward, best single region, and a spatial 2x2 control.
Only FRLL development photos are read; all images and scores stay outside Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.evaluation import Reference, evaluate_gallery
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab import patterns
from fckface_lab.recognition import SFaceModel

VIEWS = ("neutral_front", "smiling_front", "neutral_left_3quarter", "neutral_right_3quarter")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960", "half_restore", "crop90", "blur")
OBJECTIVE_CONDITIONS = ("export", "jpeg75_420", "blur")
RMS_TARGETS = (8.0, 16.0)
SEED = 0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_json(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def xywh(box) -> tuple[float, float, float, float]:
    left, top, right, bottom = map(float, box)
    return left, top, right - left, bottom - top


def crop_png(image: np.ndarray, box) -> bytes:
    l, t, r, b = box
    mx, my = .14 * (r-l), .14 * (b-t)
    bounds = (max(0, int(l-mx)), max(0, int(t-my)),
              min(image.shape[1], int(r+mx)), min(image.shape[0], int(b+my)))
    stream = BytesIO()
    Image.fromarray(image[bounds[1]:bounds[3], bounds[0]:bounds[2]]).save(stream, "PNG")
    return stream.getvalue()


class Artwork:
    """Same fixed dot/ring/screenprint carrier with four coefficient fields."""

    def __init__(self, source: np.ndarray, box_xyxy, landmarks: np.ndarray):
        self.source = source
        self.box_xyxy = box_xyxy
        self.box = np.asarray(xywh(box_xyxy), dtype=np.float64)
        x, y, bw, bh = self.box
        self.left = max(0, int(np.floor(x - .12*bw)))
        self.top = max(0, int(np.floor(y - .12*bh)))
        self.right = min(source.shape[1], int(np.ceil(x + 1.12*bw)))
        self.bottom = min(source.shape[0], int(np.ceil(y + 1.12*bh)))
        self.base = source[self.top:self.bottom, self.left:self.right]
        self.local_box = self.box - np.array([self.left, self.top, 0, 0])
        self.landmarks = landmarks - np.array([self.left, self.top])
        mask = patterns.face_mask(self.base.shape, self.local_box)
        self.face = mask > 0
        regions = patterns._region_masks(self.base.shape, self.local_box, self.landmarks)
        ux, uy = patterns._face_coordinates(self.base.shape, self.local_box, self.landmarks)
        layers = [patterns._carrier(ux, uy, period, shape, SEED, layer, True)
                  for layer, (period, shape) in enumerate(
                      ((8.0, "dot"), (16.0, "ring"), (24.0, "screenprint")))]
        carrier = sum(layers) / 3.0
        self.regional = np.stack([carrier * mask * regions[name]
                                  for name in patterns.FACE_REGIONS]).astype(np.float32)
        # Whole-face control: bilinear coefficients on a 2x2 spatial grid.
        u = np.clip((ux / 128.0), 0, 1)
        v = np.clip((uy / 128.0), 0, 1)
        self.spatial = np.stack([carrier * mask * weight for weight in
                                 ((1-u)*(1-v), u*(1-v), (1-u)*v, u*v)]).astype(np.float32)
        base_float = self.base.astype(np.float32)
        self.negative_headroom = base_float.min(axis=2)
        self.positive_headroom = 255 - base_float.max(axis=2)

    def render(self, coefficients: np.ndarray, target: float, *, spatial=False
               ) -> tuple[np.ndarray, dict]:
        coeff = np.asarray(coefficients, dtype=np.float32)
        if coeff.shape != (4,) or not np.all(np.isfinite(coeff)) or np.any(np.abs(coeff) > 1):
            raise ValueError("Four finite signed coefficients in [-1,1] required")
        planes = self.spatial if spatial else self.regional
        requested = np.tensordot(coeff, planes, axes=(0, 0))
        if not np.any(requested):
            return self.source.copy(), {"face_rms": 0.0, "max_delta": 0, "rms_matched": False}

        def channel_delta(scale: float) -> np.ndarray:
            rounded = np.rint(requested * scale)
            return np.clip(rounded, np.maximum(-64, -self.negative_headroom),
                           np.minimum(64, self.positive_headroom)).astype(np.int16)

        def rms(delta: np.ndarray) -> float:
            face = delta[self.face].astype(np.float32)
            return float(np.sqrt(np.mean(face * face))) if face.size else 0.0

        lo, hi = 0.0, 1.0
        while rms(channel_delta(hi)) < target and hi < 65536:
            hi *= 2
        if rms(channel_delta(hi)) >= target:
            for _ in range(13):
                mid = (lo + hi) / 2
                if rms(channel_delta(mid)) < target:
                    lo = mid
                else:
                    hi = mid
        delta = channel_delta(hi)
        result = self.source.copy()
        result[self.top:self.bottom, self.left:self.right] = (
            self.base.astype(np.int16) + delta[..., None]).astype(np.uint8)
        achieved = rms(delta)
        return result, {"face_rms": achieved, "target_face_rms": target,
                        "rms_mismatch": achieved-target,
                        "rms_matched": abs(achieved-target) <= .25,
                        "max_delta": int(np.max(np.abs(delta))),
                        "nonzero_pixels": int(np.count_nonzero(delta))}


def evaluate_conditions(jpeg: bytes, box, model, refs, calibration, identity,
                        names: tuple[str, ...], *, save_dir: Path | None = None):
    variants = make_variants(jpeg, box)
    rows = {}
    for name in names:
        variant = variants[name]
        entry = {"status": variant.status, "reason": variant.reason,
                 "target_box_xyxy": variant.target_box}
        if variant.status == "valid" and variant.image is not None:
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / f"{name}.jpg").write_bytes(variant.jpeg)
                (save_dir / f"{name}-face.png").write_bytes(crop_png(variant.image, variant.target_box))
                entry["jpeg_sha256"] = digest(save_dir / f"{name}.jpg")
            probe = model.embed(variant.image, expected_box=variant.target_box)
            gallery = evaluate_gallery(
                probe, refs, calibration, own_identity=identity,
                own_source_name=f"{identity}:neutral_front",
                expected_reference_names=[f"{identity}:{view}" for view in VIEWS],
            )
            entry.update({"status": gallery.status, "reason": gallery.reason,
                          "detection_count": probe.detection_count,
                          "per_reference": dict(gallery.per_reference),
                          "maximum_cosine": max(gallery.per_reference.values())
                          if gallery.status == "valid" else None,
                          "own_source_cosine": gallery.own_source_score,
                          "matching_references": list(gallery.matching_references),
                          "own_identity_matched": gallery.own_identity_matched})
        rows[name] = entry
    return rows


def objective(art: Artwork, coefficients: np.ndarray, target: float, *, spatial: bool,
              model, refs, calibration, identity, cache: dict
              ) -> tuple[float, dict, bytes, bool]:
    image, metrics = art.render(coefficients, target, spatial=spatial)
    jpeg = export_jpeg(image)
    key = hashlib.sha256(jpeg).hexdigest()
    if key in cache:
        score, evidence = cache[key]
        return score, evidence, jpeg, True
    if not metrics["rms_matched"]:
        result = (1.0, {"distortion": metrics, "reason": "rms_unattainable"})
        cache[key] = result
        return *result, jpeg, False
    conditions = evaluate_conditions(jpeg, art.box_xyxy, model, refs, calibration,
                                     identity, OBJECTIVE_CONDITIONS)
    values = [conditions[name].get("maximum_cosine") for name in OBJECTIVE_CONDITIONS]
    if any(value is None for value in values):
        result = (1.0, {"distortion": metrics, "conditions": conditions,
                        "reason": "invalid_or_inconclusive_condition"})
    else:
        result = (max(values), {"distortion": metrics, "conditions": conditions})
    cache[key] = result
    return *result, jpeg, False


def record_query(path: Path, *, identity: str, target: float, arm: str, index: int,
                 coefficients: np.ndarray, score: float, evidence: dict,
                 jpeg: bytes, duration: float, cached: bool) -> dict:
    row = {"identity": identity, "target_rms": target, "arm": arm, "query_index": index,
           "coefficients": [float(value) for value in coefficients],
           "objective_max_cosine_or_failure_one": score, "evidence": evidence,
           "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
           "seconds": round(duration, 4), "cache_hit": cached}
    append_json(path, row)
    return row


def search_adaptive(art, target, arm, *, spatial, query_path, model, refs,
                    calibration, identity, cache) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED)
    best = None
    unique = 0
    proposals = 0
    consecutive_reuse = 0
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
        score, evidence, jpeg, cached = objective(
            art, coeff, target, spatial=spatial, model=model, refs=refs,
            calibration=calibration, identity=identity, cache=cache)
        row = record_query(query_path, identity=identity, target=target, arm=arm,
                           index=index, coefficients=coeff, score=score,
                           evidence=evidence, jpeg=jpeg,
                           duration=perf_counter()-start, cached=cached)
        proposals += 1
        unique += not cached
        consecutive_reuse = consecutive_reuse + 1 if cached else 0
        if best is None or score < best["objective_max_cosine_or_failure_one"]:
            best = row
        if consecutive_reuse >= 8:
            break
    return best, {"proposals": proposals, "unique_objective_calls": unique,
                  "maximum_budget": 64}


def search_individual(art, target, *, query_path, model, refs,
                      calibration, identity, cache) -> tuple[list[dict], dict, dict]:
    best_by_region = []
    best_single = None
    # Exact RMS projection makes scalar magnitude redundant within a region.
    # Both signs exhaust the meaningful one-region parameter space.
    values = (-1.0, 1.0)
    unique = 0
    proposals = 0
    for region in range(4):
        best = None
        for local_index, value in enumerate(values):
            coeff = np.zeros(4, dtype=np.float32)
            coeff[region] = value
            start = perf_counter()
            score, evidence, jpeg, cached = objective(
                art, coeff, target, spatial=False, model=model, refs=refs,
                calibration=calibration, identity=identity, cache=cache)
            row = record_query(query_path, identity=identity, target=target,
                               arm="single_region_shared", index=region*2+local_index,
                               coefficients=coeff, score=score,
                               evidence=evidence, jpeg=jpeg,
                               duration=perf_counter()-start, cached=cached)
            proposals += 1
            unique += not cached
            if best is None or score < best["objective_max_cosine_or_failure_one"]:
                best = row
            if best_single is None or score < best_single["objective_max_cosine_or_failure_one"]:
                best_single = row
        best_by_region.append(best)
    return best_by_region, best_single, {"proposals": proposals,
                                        "unique_objective_calls": unique,
                                        "maximum_budget_per_derived_arm": 64,
                                        "shared_between_arms": True}


def save_winner(art, target, arm, coeff, *, spatial, output, model,
                refs, calibration, identity, selected_from):
    image, metrics = art.render(np.asarray(coeff, dtype=np.float32), target, spatial=spatial)
    jpeg = export_jpeg(image)
    destination = output / identity / f"rms{int(target)}" / arm
    rows = evaluate_conditions(jpeg, art.box_xyxy, model, refs, calibration,
                               identity, CONDITIONS, save_dir=destination)
    valid = all(row["status"] == "valid" for row in rows.values())
    nonmatching = valid and all(not row["matching_references"] for row in rows.values())
    return {"arm": arm, "coefficients": list(map(float, coeff)),
            "spatial_grid": spatial, "selected_from": selected_from,
            "distortion": metrics, "all_conditions_valid": valid,
            "all_conditions_nonmatching": nonmatching,
            "worst_condition_cosine": max(row["maximum_cosine"] for row in rows.values())
            if valid else None, "conditions": rows,
            "export_jpeg": str((destination / "export.jpg").relative_to(output))}


def contact_sheet(rows, destination):
    width, height, caption = 250, 300, 32
    sheet = Image.new("RGB", (len(rows)*width, height+caption), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(rows):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            sheet.paste(image, (index*width+(width-image.width)//2, caption))
        draw.text((index*width+5, 8), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def run(args) -> None:
    start = perf_counter()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Output must remain outside Git")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory for a complete H2b run")
    groups = grouped_images(args.manifest, "development")
    identities = sorted(groups)[:2]
    calibration_artifact = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration_artifact["status"] != "complete" or calibration_artifact["dataset"]["split"] != "calibration":
        raise ValueError("Calibration artifact is incomplete or from the wrong split")
    calibration = Calibration(**calibration_artifact["calibration"])
    for name, path in (("yunet", args.yunet), ("sface", args.sface)):
        if digest(path) != calibration_artifact["pipeline"]["model_artifacts"][name]["sha256"]:
            raise ValueError(f"{name} hash differs from frozen calibration")
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    model = SFaceModel(args.yunet, args.sface)
    source_files = [Path(__file__), root / "research/fckface_lab/patterns.py",
                    root / "research/fckface_lab/recognition.py",
                    root / "research/fckface_lab/imaging.py",
                    root / "research/fckface_lab/evaluation.py"]
    header = {"schema_version": 1, "hypothesis": "H2b signed coefficients for fixed artificial artwork",
              "scope": "first two sorted FRLL development IDs; SFace only; no heldout evidence",
              "started_utc": now(), "identities": identities, "reference_views": VIEWS,
              "objective_conditions": OBJECTIVE_CONDITIONS, "final_conditions": CONDITIONS,
              "rms_targets": RMS_TARGETS, "maximum_channel_delta": 64, "seed": SEED,
              "queries_per_arm": 64,
              "shared_queries": "independent_combined and best_single derive from the same 4x2 exhaustive sign-only regional queries; scalar magnitude is redundant after exact RMS projection",
              "renderer": "existing face-relative dot/ring/screenprint carriers, signed regional or bilinear spatial coefficients; exact RMS projection with shared RGB headroom",
              "calibration_threshold": calibration.threshold,
              "versions": {"python": sys.version, "opencv": cv2.__version__,
                           "numpy": np.__version__, "pillow": Image.__version__,
                           "opencv_threads": cv2.getNumThreads()},
              "sha256": {"manifest": digest(args.manifest), "calibration": digest(args.calibration),
                         "yunet": digest(args.yunet), "sface": digest(args.sface),
                         **{str(p.relative_to(root)): digest(p) for p in source_files}}}
    write_json(out / "header.json", header)
    log = out / "run-log.jsonl"
    query_path = out / "queries.jsonl"
    cases = []
    try:
        for identity in identities:
            began = perf_counter()
            refs = []
            source = detection = None
            input_rows = {}
            for view in VIEWS:
                path = groups[identity][view]
                pixels = decode_image(path)
                embedded = model.embed(pixels)
                input_rows[view] = {"sha256": digest(path), "status": embedded.status,
                                    "reason": embedded.reason}
                if embedded.status == "valid":
                    refs.append(Reference(f"{identity}:{view}", identity, embedded.feature))
                if view == "neutral_front":
                    source, detection = pixels, embedded
            if len(refs) != 4 or detection.status != "valid":
                append_json(log, {"event": "identity_inconclusive", "identity": identity,
                                  "references": input_rows, "at_utc": now()})
                continue
            art = Artwork(source, detection.selected_box, detection.landmarks)
            clean_jpeg = export_jpeg(source)
            control = evaluate_conditions(clean_jpeg, detection.selected_box,
                                          model, refs, calibration, identity, CONDITIONS,
                                          save_dir=out / identity / "clean")
            eligible = all(row["status"] == "valid" and row["own_identity_matched"] is True
                           for row in control.values())
            append_json(log, {"event": "clean_control", "identity": identity,
                              "eligible": eligible, "references": input_rows,
                              "conditions": control, "at_utc": now()})
            if not eligible:
                continue
            for target in RMS_TARGETS:
                case_start = perf_counter()
                cache = {}
                baseline = save_winner(art, target, "initial_baseline", [1,1,1,1],
                                       spatial=False, output=out, model=model,
                                       refs=refs, calibration=calibration,
                                       identity=identity, selected_from="fixed_all_positive")
                joint, joint_budget = search_adaptive(art, target, "joint_regions", spatial=False,
                                        query_path=query_path, model=model, refs=refs,
                                        calibration=calibration, identity=identity, cache=cache)
                spatial, spatial_budget = search_adaptive(art, target, "whole_face_2x2", spatial=True,
                                          query_path=query_path, model=model, refs=refs,
                                          calibration=calibration, identity=identity, cache=cache)
                by_region, single, individual_budget = search_individual(
                    art, target, query_path=query_path, model=model, refs=refs,
                    calibration=calibration, identity=identity, cache=cache)
                combined = [row["coefficients"][index] for index, row in enumerate(by_region)]
                selected = [baseline,
                    save_winner(art, target, "joint_regions", joint["coefficients"],
                                spatial=False, output=out, model=model, refs=refs,
                                calibration=calibration, identity=identity,
                                selected_from=joint["query_index"]),
                    save_winner(art, target, "independent_combined", combined,
                                spatial=False, output=out, model=model, refs=refs,
                                calibration=calibration, identity=identity,
                                selected_from=[row["query_index"] for row in by_region]),
                    save_winner(art, target, "best_single", single["coefficients"],
                                spatial=False, output=out, model=model, refs=refs,
                                calibration=calibration, identity=identity,
                                selected_from=single["query_index"]),
                    save_winner(art, target, "whole_face_2x2", spatial["coefficients"],
                                spatial=True, output=out, model=model, refs=refs,
                                calibration=calibration, identity=identity,
                                selected_from=spatial["query_index"])]
                case_dir = out / identity / f"rms{int(target)}"
                sheet_rows = [("clean", out / identity / "clean" / "export-face.png")]
                sheet_rows += [(row["arm"], out / identity / f"rms{int(target)}" /
                                row["arm"] / "export-face.png") for row in selected]
                contact_sheet(sheet_rows, case_dir / "contact-sheet.png")
                case = {"identity": identity, "target_rms": target,
                        "references": input_rows, "clean_control": control,
                        "winners": selected,
                        "query_budgets": {"joint_regions": joint_budget,
                                          "whole_face_2x2": spatial_budget,
                                          "independent_combined": individual_budget,
                                          "best_single": individual_budget},
                        "unique_rendered_jpegs_across_arms": len(cache),
                        "seconds": round(perf_counter()-case_start, 3)}
                cases.append(case)
                append_json(log, {"event": "case_complete", "identity": identity,
                                  "target_rms": target, "seconds": case["seconds"],
                                  "at_utc": now()})
                print(identity, f"RMS {target:g}", f"{case['seconds']:.1f}s",
                      [(w["arm"], w["worst_condition_cosine"], w["all_conditions_nonmatching"])
                       for w in selected], flush=True)
            append_json(log, {"event": "identity_complete", "identity": identity,
                              "seconds": round(perf_counter()-began, 3), "at_utc": now()})
        result = {"status": "complete", "completed_utc": now(), "header": "header.json",
                  "cases": cases, "wall_seconds": round(perf_counter()-start, 3),
                  "query_count": sum(1 for _ in query_path.open(encoding="utf-8"))
                  if query_path.exists() else 0,
                  "query_log_sha256": digest(query_path) if query_path.exists() else None,
                  "run_log_sha256": digest(log)}
        write_json(out / "results.json", result)
        print(f"H2b complete: {out} ({result['query_count']} objective calls)", flush=True)
    except Exception as exc:
        append_json(log, {"event": "error", "type": type(exc).__name__,
                          "message": str(exc), "at_utc": now()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--yunet", type=Path, required=True)
    parser.add_argument("--sface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
