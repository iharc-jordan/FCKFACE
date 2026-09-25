"""Calibrate development SFace on FRLL calibration identities only.

Writes local biometric scores and embeddings outside Git. This is development
calibration, not validation of an altered image or a public release gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from fckface_lab.calibration import CalibrationPair, calibrate
from fckface_lab.datasets import VIEWS, digest, grouped_images
from fckface_lab.evaluation import cosine
from fckface_lab.imaging import decode_image
from fckface_lab.recognition import SFaceModel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def add_log(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    files = ("calibrate_sface.py", "fckface_lab/calibration.py",
             "fckface_lab/datasets.py", "fckface_lab/evaluation.py",
             "fckface_lab/imaging.py", "fckface_lab/recognition.py", "models.json")
    return {name: digest(base / name) for name in files}


def save_overlay(image: np.ndarray, box: tuple[float, float, float, float],
                 landmarks: np.ndarray, path: Path) -> None:
    """Private diagnostic for manual verification of selection and landmarks."""
    canvas = Image.fromarray(image.copy(), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(tuple(round(value) for value in box), outline=(255, 0, 0), width=5)
    for x, y in landmarks:
        draw.ellipse((round(x) - 6, round(y) - 6, round(x) + 6, round(y) + 6),
                     fill=(0, 255, 0), outline=(0, 0, 0), width=2)
    canvas.save(path, format="JPEG", quality=92, subsampling=0)


def read_calibration_records(manifest_path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "frll" or manifest.get("split_version") != "frll-identities-v1":
        raise ValueError("Unexpected dataset or identity split version")
    groups = grouped_images(manifest_path, "calibration")
    if len(groups) != 20 or any(set(views) != set(VIEWS) for views in groups.values()):
        raise ValueError("Expected all ten views for each of 20 calibration identities")
    rows = sorted((row for row in manifest["images"] if row["split"] == "calibration"),
                  key=lambda row: (row["identity"], row["view"]))
    if len(rows) != 200 or len({(row["identity"], row["view"]) for row in rows}) != 200:
        raise ValueError("Calibration image records are missing or duplicated")
    for row in rows:
        if manifest["identity_splits"].get(row["identity"]) != "calibration":
            raise ValueError("Calibration record conflicts with frozen identity split")
        expected = groups[row["identity"]][row["view"]]
        actual = (manifest_path.parent / row["path"]).resolve()
        if actual != expected or not actual.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Calibration path mismatch or path escape")
        row["absolute_path"] = str(actual)
    return rows, manifest


def verify_models(yunet_path: Path, sface_path: Path) -> dict[str, dict]:
    registry = json.loads((Path(__file__).resolve().parent / "models.json").read_text())
    results = {}
    for name, path in (("yunet", yunet_path), ("sface", sface_path)):
        entry = registry["development"][name]
        if path.name != entry["filename"] or not path.is_file():
            raise ValueError(f"Wrong or missing {name} model artifact")
        sha256 = digest(path)
        if sha256 != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{name} artifact hash or size differs from registry")
        results[name] = {"path": str(path.resolve()), "sha256": sha256,
                         "bytes": path.stat().st_size, "source": entry["source"]}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yunet", type=Path, required=True)
    parser.add_argument("--sface", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    source_root = Path(__file__).resolve().parents[1]
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError("Calibration output must be outside the repository")
    if (output / "calibration.json").exists():
        raise FileExistsError("Frozen calibration.json already exists; choose a new output")
    rows, manifest = read_calibration_records(manifest_path)
    models = verify_models(args.yunet, args.sface)
    cv2.setNumThreads(2)
    started = utc_now()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run-log.jsonl"
    add_log(log_path, {"event": "start", "at_utc": started, "image_count": len(rows)})
    try:
        recognizer = SFaceModel(args.yunet, args.sface)
        features: list[np.ndarray | None] = []
        image_results: list[dict] = []
        diagnostic_views = ("neutral_front", "smiling_front",
                            "neutral_left_3quarter", "neutral_right_3quarter")
        diagnostic_images: list[dict] = []
        diagnostic_identities: set[str] = set()
        for index, row in enumerate(rows):
            began = perf_counter()
            path = Path(row["absolute_path"])
            reason = None
            feature = None
            status = "invalid"
            detections = 0
            try:
                if digest(path) != row["sha256"]:
                    reason = "image_sha256_mismatch"
                else:
                    image = decode_image(path)
                    embedded = recognizer.embed(image)
                    status, reason = embedded.status, embedded.reason
                    detections = embedded.detection_count
                    feature = embedded.feature
                    if (status == "valid" and row["view"] in diagnostic_views
                            and row["view"] not in {d["view"] for d in diagnostic_images}
                            and row["identity"] not in diagnostic_identities
                            and embedded.selected_box is not None
                            and embedded.landmarks is not None):
                        diagnostic_dir = output / "diagnostics"
                        diagnostic_dir.mkdir(exist_ok=True)
                        diagnostic_path = diagnostic_dir / f"{row['view']}.jpg"
                        save_overlay(image, embedded.selected_box,
                                     embedded.landmarks, diagnostic_path)
                        diagnostic_images.append({"view": row["view"],
                                                  "identity": row["identity"],
                                                  "relative_path": diagnostic_path.relative_to(output).as_posix(),
                                                  "sha256": digest(diagnostic_path)})
                        diagnostic_identities.add(row["identity"])
            except (OSError, ValueError, cv2.error) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            duration = perf_counter() - began
            result = {"index": index, "identity": row["identity"], "view": row["view"],
                      "image_id": f"{row['identity']}:{row['view']}",
                      "sha256": row["sha256"], "status": status,
                      "reason": reason, "detection_count": detections,
                      "seconds": round(duration, 6)}
            image_results.append(result)
            features.append(feature if status == "valid" else None)
            add_log(log_path, {"event": "image", "at_utc": utc_now(), **result})
            if (index + 1) % 20 == 0:
                print(f"Embedded {index + 1}/200; valid {sum(f is not None for f in features)}", flush=True)

        valid_feature = next((f for f in features if f is not None), None)
        if valid_feature is None:
            raise ValueError("No valid calibration embeddings")
        matrix = np.zeros((len(features), valid_feature.size), dtype=np.float32)
        for index, feature in enumerate(features):
            if feature is not None:
                if feature.shape != valid_feature.shape:
                    raise ValueError("Embedding dimensions differ")
                matrix[index] = feature
        npz_path = output / "embeddings.npz"
        np.savez_compressed(npz_path, features=matrix,
                            image_ids=np.asarray([row["image_id"] for row in image_results]),
                            identities=np.asarray([row["identity"] for row in rows]),
                            views=np.asarray([row["view"] for row in rows]),
                            valid=np.asarray([feature is not None for feature in features]),
                            status=np.asarray([row["status"] for row in image_results]))

        scores_path = output / "scores.jsonl"
        pair_evidence: list[tuple[str, str, CalibrationPair]] = []
        strata: dict[str, Counter] = defaultdict(Counter)
        with scores_path.open("w", encoding="utf-8") as stream:
            for i, j in combinations(range(len(rows)), 2):
                left, right = rows[i], rows[j]
                same_identity = left["identity"] == right["identity"]
                same_view = left["view"] == right["view"]
                stratum = ("genuine" if same_identity else "impostor") + (
                    "_same_view" if same_view else "_different_view")
                if features[i] is None or features[j] is None:
                    score, reason = None, "one_or_both_embeddings_invalid"
                else:
                    score = cosine(features[i], features[j])
                    reason = None
                strata[stratum]["total"] += 1
                strata[stratum]["valid" if score is not None else "invalid"] += 1
                pair = CalibrationPair(same_identity, score)
                pair_evidence.append((left["identity"], right["identity"], pair))
                stream.write(json.dumps({"left_index": i, "right_index": j,
                                         "same_identity": same_identity,
                                         "same_view": same_view,
                                         "stratum": stratum, "score": score,
                                         "reason": reason}, allow_nan=False) + "\n")

        calibration = calibrate([row[2] for row in pair_evidence], target_fmr=.001)
        loo_thresholds = []
        for identity in sorted({row["identity"] for row in rows}):
            subset = [pair for left, right, pair in pair_evidence
                      if left != identity and right != identity]
            loo_thresholds.append({"removed_identity": identity,
                                   "threshold": calibrate(subset, target_fmr=.001).threshold})
        completed = utc_now()
        add_log(log_path, {"event": "complete", "at_utc": completed,
                           "valid_images": sum(f is not None for f in features),
                           "pair_count": len(pair_evidence),
                           "threshold": calibration.threshold})
        artifact = {
            "schema_version": 1, "purpose": "development_native_sface_calibration_only",
            "status": "complete", "started_utc": started, "completed_utc": completed,
            "dataset": {"name": "frll", "split": "calibration",
                        "split_version": manifest["split_version"],
                        "identity_count": 20, "image_count": 200,
                        "manifest_sha256": digest(manifest_path),
                        "identity_ids": sorted({row["identity"] for row in rows}),
                        "views": list(VIEWS)},
            "pipeline": {"detector": "OpenCV FaceDetectorYN YuNet",
                         "recognizer": "OpenCV FaceRecognizerSF SFace",
                         "face_selection": "exactly_one_detected_face",
                         "alignment": "FaceRecognizerSF.alignCrop five-point YuNet",
                         "embedding": "L2-normalized FaceRecognizerSF.feature",
                         "comparison": "cosine >= threshold is match",
                         "image_decode": "EXIF-transposed ICC-to-sRGB RGB",
                         "detector_input": {
                             "maximum_side_px": 640,
                             "score_threshold": 0.9,
                             "scale": "min(1, 640/max(original_width, original_height))",
                             "size_rounding": "Python round(original_dimension * scale), minimum 1",
                             "downscale_interpolation": "cv2.INTER_AREA",
                             "coordinate_mapping": "YuNet XYWH and five landmark x coordinates * original_width/detector_width; y coordinates * original_height/detector_height",
                             "target_box": "XYXY in original decoded pixels; unique overlap selection",
                             "recognizer_input": "original full-resolution decoded image",
                         },
                         "cv2_threads": cv2.getNumThreads(),
                         "source_sha256": source_hashes(), "model_artifacts": models,
                         "versions": {"python": platform.python_version(),
                                      "opencv": cv2.__version__, "numpy": np.__version__,
                                      "pillow": pillow_version}},
            "calibration": calibration.as_dict(),
            "pair_strata": {key: dict(value) for key, value in sorted(strata.items())},
            "images": image_results,
            "diagnostic_overlays": diagnostic_images,
            "threshold_uncertainty": {
                "leave_one_identity_out": loo_thresholds,
                "minimum": min(item["threshold"] for item in loo_thresholds),
                "maximum": max(item["threshold"] for item in loo_thresholds),
                "caveat": "Pair scores share only 20 identities; empirical FMR and leave-one-identity-out range are not population confidence bounds. Detection failures reduce valid coverage. This is development SFace, not held-out validation."},
            "private_artifacts": {"scores.jsonl": digest(scores_path),
                                  "embeddings.npz": digest(npz_path),
                                  "run-log.jsonl": digest(log_path)},
        }
        write_json(output / "calibration.json", artifact)
        print(json.dumps({"calibration_json": str(output / "calibration.json"),
                          "threshold": calibration.threshold,
                          "empirical_fmr": calibration.empirical_fmr,
                          "empirical_fnmr": calibration.empirical_fnmr,
                          "valid_coverage": calibration.valid_coverage,
                          "valid_images": sum(f is not None for f in features)}), flush=True)
        return 0
    except Exception as exc:
        add_log(log_path, {"event": "error", "at_utc": utc_now(),
                           "type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    sys.exit(main())
