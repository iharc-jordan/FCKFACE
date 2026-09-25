"""Combine the frozen H1 pilot and six-ID confirmation without model selection.

This is a descriptive development report. The compared renderers also differ
in face-relative scale, so a score difference does not identify its cause.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from fckface_lab.imaging import decode_image
from fckface_lab.recognition import SFaceModel


METHODS = ("uniform_dots", "global_multiscale", "face_relative_multiscale")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960", "half_restore", "crop90", "blur")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fitted_scale(landmarks: np.ndarray) -> float:
    """Same five-point least-squares similarity magnitude as patterns.py."""
    canonical = np.array([
        [.32 * 128, .35 * 128], [.68 * 128, .35 * 128],
        [.50 * 128, .56 * 128], [.37 * 128, .76 * 128],
        [.63 * 128, .76 * 128]], dtype=np.float64)
    a = landmarks.astype(np.float64) - landmarks.mean(axis=0)
    b = canonical - canonical.mean(axis=0)
    u, singular, vt = np.linalg.svd(a.T @ b)
    diagonal = np.diag([1.0, np.linalg.det(u @ vt)])
    transform = (u @ diagonal @ vt) * (
        np.sum(singular * np.diag(diagonal)) / np.sum(a * a))
    return float(np.linalg.norm(transform[:, 0]))


def read_run(root: Path):
    header = json.loads((root / "header.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    rows = [json.loads(line) for line in (root / "cases.jsonl").open(encoding="utf-8") if line.strip()]
    if digest(root / "cases.jsonl") != summary["cases_jsonl_sha256"]:
        raise ValueError(f"Case records changed after completion: {root}")
    return header, summary, rows


def main(pilot: Path, confirm: Path, yunet: Path, sface: Path, out: Path):
    if out.exists():
        raise FileExistsError(out)
    pilot_h, pilot_s, pilot_rows = read_run(pilot)
    confirm_h, confirm_s, confirm_rows = read_run(confirm)
    if (pilot_h["identities"] != ["frll-001", "frll-003"]
            or confirm_h["identities"] != ["frll-004", "frll-007", "frll-011",
                                         "frll-013", "frll-018", "frll-019"]
            or confirm_h["family"] != "h1"
            or pilot_h["calibration"]["threshold"] != confirm_h["calibration"]["threshold"]
            or pilot_h["input_sha256"]["patterns"] != confirm_h["input_sha256"]["patterns"]
            or pilot_h["input_sha256"]["recognition"] != confirm_h["input_sha256"]["recognition"]
            or pilot_h["input_sha256"]["yunet"] != confirm_h["input_sha256"]["yunet"]
            or pilot_h["input_sha256"]["sface"] != confirm_h["input_sha256"]["sface"]):
        raise ValueError("Pilot and confirmation protocols are incompatible")
    if digest(yunet) != pilot_h["input_sha256"]["yunet"] or digest(sface) != pilot_h["input_sha256"]["sface"]:
        raise ValueError("Native models changed")
    all_rows = [row for row in pilot_rows + confirm_rows if row["method"] in METHODS]
    ids = pilot_h["identities"] + confirm_h["identities"]
    if len(all_rows) != 8 * 2 * 2 * 3 or len({(r["identity"], r["seed"],
                                               r["target_face_rms"], r["method"])
                                              for r in all_rows}) != len(all_rows):
        raise ValueError("Expected one result for every frozen H1 case")
    lookup = {(row["identity"], row["seed"], row["target_face_rms"], row["method"]): row
              for row in all_rows}
    cv2.setNumThreads(2)
    model = SFaceModel(yunet, sface)
    summaries = {row["identity"]: row for row in pilot_s["identities"] + confirm_s["identities"]}
    scales = {}
    for identity in ids:
        source = summaries[identity]["references"]["neutral_front"]
        image = decode_image(Path(source["path"]))
        detection = model.embed(image)
        if detection.status != "valid" or not np.allclose(detection.selected_box, source["box_xyxy"], atol=.01):
            raise ValueError(f"Source detection differs on replay: {identity}")
        fit = fitted_scale(detection.landmarks)
        box_width = detection.selected_box[2] - detection.selected_box[0]
        global_scale = 128.0 / box_width
        scales[identity] = {"source_path": source["path"], "source_sha256": source["sha256"],
                            "source_box_xyxy": source["box_xyxy"],
                            "face_relative_fit_scale": fit,
                            "global_bbox_width_scale": global_scale,
                            "fit_over_global_scale_ratio": fit / global_scale}
    entries = []
    pair_differences = []
    unmatched = []
    for identity in ids:
        for seed in (0, 1):
            for target in (4, 8):
                cases = {method: lookup[identity, seed, target, method] for method in METHODS}
                for method, case in cases.items():
                    if not case["raw_distortion"]["rms_matched"]:
                        unmatched.append({"identity": identity, "seed": seed,
                                          "target_face_rms": target, "method": method,
                                          "achieved_face_rms": case["raw_distortion"]["face_rms"]})
                for condition in CONDITIONS:
                    methods = {}
                    for method, case in cases.items():
                        result = case["conditions"][condition]
                        evaluation = result["evaluation"]
                        image_path = ((pilot if identity in pilot_h["identities"] else confirm)
                                      / identity / f"seed{seed}-rms{target}" / method
                                      / f"{condition}.jpg")
                        methods[method] = {
                            "status": evaluation["status"], "reason": evaluation["reason"],
                            "gallery_match": bool(evaluation["matching_references"])
                            if evaluation["status"] == "valid" else None,
                            "matching_references": evaluation["matching_references"],
                            "maximum_own_identity_cosine": max(evaluation["per_reference"].values())
                            if evaluation["status"] == "valid" else None,
                            "raw_rms_matched": case["raw_distortion"]["rms_matched"],
                            "post_jpeg_face_rms": result.get("post_jpeg_distortion", {}).get("face_rms"),
                            "post_jpeg_changed_fraction_face": result.get("post_jpeg_distortion", {}).get("changed_fraction_face"),
                            "image_path": str(image_path), "image_sha256": result.get("jpeg_sha256"),
                        }
                    a, b = methods["face_relative_multiscale"], methods["global_multiscale"]
                    pair = {"face_relative_minus_global_max_cosine": (
                        a["maximum_own_identity_cosine"] - b["maximum_own_identity_cosine"]
                        if a["status"] == b["status"] == "valid" else None),
                            "post_jpeg_rms_absolute_difference": (
                                abs(a["post_jpeg_face_rms"] - b["post_jpeg_face_rms"])
                                if a["post_jpeg_face_rms"] is not None and b["post_jpeg_face_rms"] is not None
                                else None),
                            "post_jpeg_changed_fraction_absolute_difference": (
                                abs(a["post_jpeg_changed_fraction_face"] - b["post_jpeg_changed_fraction_face"])
                                if a["post_jpeg_changed_fraction_face"] is not None and b["post_jpeg_changed_fraction_face"] is not None
                                else None)}
                    if pair["face_relative_minus_global_max_cosine"] is not None:
                        pair_differences.append((identity, seed, target, condition,
                                                 pair["face_relative_minus_global_max_cosine"]))
                    entries.append({"identity": identity, "seed": seed,
                                    "target_face_rms": target, "condition": condition,
                                    "methods": methods, "paired_difference": pair})
    per_identity = {}
    for identity in ids:
        differences = [value for ident, _, _, _, value in pair_differences if ident == identity]
        per_identity[identity] = {
            "eligible_clean_control": summaries[identity]["status"] == "eligible",
            "relative_lower_than_global": sum(value < 0 for value in differences),
            "paired_conditions": len(differences),
            "mean_relative_minus_global_max_cosine": mean(differences),
            "gallery_nonmatches_by_method": {method: sum(
                entry["methods"][method]["gallery_match"] is False
                for entry in entries if entry["identity"] == identity)
                for method in METHODS},
            "sheet_paths": [str((pilot if identity in pilot_h["identities"] else confirm)
                                / "sheets" / f"{identity}-seed{seed}-rms{target}.png")
                            for seed in (0, 1) for target in (4, 8)],
        }
    per_condition = {}
    for condition in CONDITIONS:
        values = [v for _, _, _, c, v in pair_differences if c == condition]
        per_condition[condition] = {"relative_lower_than_global": sum(v < 0 for v in values),
                                    "pairs": len(values), "mean_difference": mean(values)}
    report = {
        "scope": "Descriptive development-only SFace fixed-renderer comparison; no privacy success or causal anchoring proof",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": pilot_h["calibration"]["threshold"],
        "inputs_sha256": {"pilot_header": digest(pilot / "header.json"),
                          "pilot_cases": digest(pilot / "cases.jsonl"),
                          "confirm_header": digest(confirm / "header.json"),
                          "confirm_cases": digest(confirm / "cases.jsonl"),
                          "analysis_code": digest(Path(__file__)),
                          "yunet": digest(yunet), "sface": digest(sface)},
        "identities": ids, "condition_count": len(CONDITIONS),
        "candidate_count": len(all_rows), "comparison_count": len(entries) * len(METHODS),
        "valid_comparisons": sum(entry["methods"][method]["status"] == "valid"
                                 for entry in entries for method in METHODS),
        "gallery_nonmatches": sum(entry["methods"][method]["gallery_match"] is False
                                  for entry in entries for method in METHODS),
        "raw_rms_unmatched": unmatched,
        "relative_lower_than_global": sum(value < 0 for *_, value in pair_differences),
        "relative_global_pairs": len(pair_differences),
        "mean_relative_minus_global_max_cosine": mean(value for *_, value in pair_differences),
        "scale_ratios": scales,
        "per_identity": per_identity, "per_condition": per_condition,
        "per_identity_seed_target_condition": entries,
        "limitations": [
            "All eight identities and SFace are development evidence; no independent transfer test.",
            "Every edited condition still matched an own-identity gallery reference.",
            "Face-relative and image-anchored carriers use different fitted spatial scales; score differences cannot be attributed to anchoring alone.",
            "Two seeds and seven correlated processing conditions do not add independent identities.",
            "Quantized RMS misses are listed and excluded from exact matched-strength interpretation.",
            "Root rejected skin-coloured H3 graphic shading as a product candidate; H2 fixed presets had zero pilot nonmatches.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "candidate_count", "comparison_count", "valid_comparisons", "gallery_nonmatches",
        "relative_lower_than_global", "relative_global_pairs",
        "mean_relative_minus_global_max_cosine")}, indent=2))
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "confirm", "yunet", "sface", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    main(args.pilot.resolve(), args.confirm.resolve(), args.yunet.resolve(),
         args.sface.resolve(), args.out.resolve())
