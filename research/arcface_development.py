"""Pinned InsightFace buffalo_l development calibration and frozen H7 scoring.

Run `calibrate` first; `score` refuses to open H7 exports until the calibration
artifact exists and matches this exact adapter and upstream model/code hashes.
All photos, embeddings, model files, and result artifacts stay outside Git.
InsightFace pretrained weights are non-commercial research-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from calibrate_sface import read_calibration_records
from fckface_lab.calibration import Calibration, CalibrationPair, calibrate
from fckface_lab.datasets import VIEWS, digest, grouped_images
from fckface_lab.evaluation import Reference, cosine, evaluate_gallery
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.recognition import EmbeddingResult, select_face


DATA = Path.home() / "Downloads" / "FCKFACE-data"
ARC = DATA / "arcface-development-v1"
OUT = DATA / "runs" / "arcface-development-v1"
H7 = DATA / "runs" / "alignment-dots-v1"
MANIFEST = DATA / "frll" / "manifest.json"
PIN = "1480e705287bc5d59f923b46c260ec6e3e4150f6"
IDS = ("frll-001", "frll-003")
GALLERY_VIEWS = ("neutral_front", "smiling_front", "neutral_left_3quarter", "neutral_right_3quarter")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960", "half_restore", "crop90", "blur")
ARMS = ("fixed_condition_landmarks", "refresh_edited_landmarks")
EXPECTED = {
    "buffalo_l.zip": "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    "scrfd.py": "c06275fe207bd02b08b462cfb318054787058cd0005b2eac9004afb0e1943c64",
    "arcface_onnx.py": "2df3524243b8e807b0e4756d98bd3f8fd7456feaf6d04da7502a24a54d74e2bc",
    "face_align.py": "a8b34ddd06716f5ca7e7501415cbd3789ccbce18db9e557b35e537695290d922",
    "coreml_cache.py": "76c5dadc8008e889a563ec0b7e6793ce5d73cc36c0baf182ea21aac5c0309984",
    "onnxruntime_utils.py": "0b5482f03f5f1a37db89569880268b6da090a00263134680f94a8b9eb87bd87f",
}
UPSTREAM = {
    "scrfd.py": "insightface_official/model_zoo/scrfd.py",
    "arcface_onnx.py": "insightface_official/model_zoo/arcface_onnx.py",
    "face_align.py": "insightface_official/utils/face_align.py",
    "coreml_cache.py": "insightface_official/model_zoo/coreml_cache.py",
    "onnxruntime_utils.py": "insightface_official/model_zoo/onnxruntime_utils.py",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_inputs() -> dict:
    hashes = {}
    for name, expected in EXPECTED.items():
        path = ARC / UPSTREAM[name] if name in UPSTREAM else ARC / name
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Pinned InsightFace asset differs: {name}: {actual}")
        hashes[name] = {"sha256": actual, "bytes": path.stat().st_size}
    return {"repository": "https://github.com/deepinsight/insightface",
            "commit": PIN, "model_pack": "buffalo_l", "assets": hashes,
            "model_terms": "non-commercial research only; not for public redistribution",
            "source_code_terms": "MIT per upstream repository"}


class ArcFaceModel:
    """Official SCRFD/ArcFaceONNX/face_align classes, pinned outside Git."""

    def __init__(self):
        sys.path.insert(0, str(ARC))
        from insightface_official.model_zoo.scrfd import SCRFD
        from insightface_official.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface_official.utils import face_align

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 2
        providers = ["CPUExecutionProvider"]
        detector_path = ARC / "det_10g.onnx"
        recognizer_path = ARC / "w600k_r50.onnx"
        det_session = ort.InferenceSession(str(detector_path), sess_options=options, providers=providers)
        rec_session = ort.InferenceSession(str(recognizer_path), sess_options=options, providers=providers)
        self.detector = SCRFD(model_file=str(detector_path), session=det_session,
                              static_shape_sessions=False)
        self.detector.prepare(0, input_size=(640, 640), det_thresh=0.5)
        self.recognizer = ArcFaceONNX(model_file=str(recognizer_path), session=rec_session)
        self.face_align = face_align
        if (not self.detector.use_kps or self.detector.input_sizes != [(640, 640)]
                or self.detector.input_mean != 127.5 or self.detector.input_std != 128.0
                or self.recognizer.input_size != (112, 112)
                or self.recognizer.input_mean != 127.5 or self.recognizer.input_std != 127.5):
            raise ValueError("Unexpected official buffalo_l preprocessing contract")

    def embed(self, image_rgb: np.ndarray, expected_box=None) -> EmbeddingResult:
        if (image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8
                or min(image_rgb.shape[:2]) < 16):
            return EmbeddingResult("invalid", reason="invalid_rgb_image")
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        try:
            boxes, kps = self.detector.detect(bgr, input_size=(640, 640))
        except (cv2.error, ValueError, RuntimeError):
            return EmbeddingResult("inconclusive", reason="detector_error")
        count = len(boxes)
        if kps is None or len(kps) != count:
            return EmbeddingResult("inconclusive", reason="missing_five_point_landmarks",
                                   detection_count=count)
        faces = np.concatenate((boxes[:, :2], boxes[:, 2:4] - boxes[:, :2],
                                kps.reshape(count, 10), boxes[:, 4:5]), axis=1)
        selected, reason = select_face(faces, expected_box)
        if selected is None:
            return EmbeddingResult("inconclusive", reason=reason, detection_count=count)
        box = (float(selected[0]), float(selected[1]),
               float(selected[0] + selected[2]), float(selected[1] + selected[3]))
        points = selected[4:14].reshape(5, 2).copy()
        try:
            crop = self.face_align.norm_crop(bgr, landmark=points, image_size=112)
            raw = np.asarray(self.recognizer.get_feat(crop), np.float32).reshape(-1)
        except (cv2.error, ValueError, RuntimeError):
            return EmbeddingResult("inconclusive", reason="alignment_or_feature_error",
                                   selected_box=box, detection_count=count, landmarks=points)
        norm = float(np.linalg.norm(raw))
        if raw.size != 512 or not np.all(np.isfinite(raw)) or not np.isfinite(norm) or norm <= 0:
            return EmbeddingResult("inconclusive", reason="invalid_feature",
                                   selected_box=box, detection_count=count, landmarks=points)
        return EmbeddingResult("valid", raw / norm, selected_box=box,
                               detection_count=count, landmarks=points)


def calibrate_phase() -> None:
    if OUT.exists():
        raise FileExistsError(f"Research output exists: {OUT}")
    provenance = verify_inputs()
    rows, manifest = read_calibration_records(MANIFEST)
    cv2.setNumThreads(2)
    model = ArcFaceModel()
    OUT.mkdir(parents=True)
    shutil.copyfile(Path(__file__), OUT / "executed-source.py")
    started = now()
    features = []
    images = []
    try:
        for index, row in enumerate(rows):
            path = Path(row["absolute_path"])
            t0 = perf_counter()
            if digest(path) != row["sha256"]:
                raise ValueError(f"Calibration image hash changed: {path}")
            embedded = model.embed(decode_image(path))
            images.append({"image_id": f"{row['identity']}:{row['view']}",
                           "identity": row["identity"], "view": row["view"],
                           "sha256": row["sha256"], "status": embedded.status,
                           "reason": embedded.reason, "detection_count": embedded.detection_count,
                           "seconds": perf_counter() - t0})
            features.append(embedded.feature if embedded.status == "valid" else None)
            if (index + 1) % 20 == 0:
                print(f"ArcFace calibration {index + 1}/200 valid={sum(x is not None for x in features)}", flush=True)
        matrix = np.zeros((len(rows), 512), np.float32)
        for i, feature in enumerate(features):
            if feature is not None:
                matrix[i] = feature
        np.savez_compressed(OUT / "embeddings.npz", features=matrix,
                            image_ids=np.asarray([row["image_id"] for row in images]),
                            valid=np.asarray([feature is not None for feature in features]))
        pairs = []
        strata = defaultdict(Counter)
        with (OUT / "scores.jsonl").open("w", encoding="utf-8") as stream:
            for i, j in combinations(range(len(rows)), 2):
                left, right = rows[i], rows[j]
                same = left["identity"] == right["identity"]
                stratum = ("genuine" if same else "impostor") + (
                    "_same_view" if left["view"] == right["view"] else "_different_view")
                score = cosine(features[i], features[j]) if features[i] is not None and features[j] is not None else None
                pairs.append((left["identity"], right["identity"], CalibrationPair(same, score)))
                strata[stratum]["valid" if score is not None else "invalid"] += 1
                stream.write(json.dumps({"left_index": i, "right_index": j,
                                         "same_identity": same, "stratum": stratum,
                                         "score": score}, allow_nan=False) + "\n")
        calibration = calibrate([row[2] for row in pairs], target_fmr=.001)
        loo = []
        for identity in sorted({row["identity"] for row in rows}):
            subset = [pair for left, right, pair in pairs if left != identity and right != identity]
            loo.append({"removed_identity": identity,
                        "threshold": calibrate(subset, target_fmr=.001).threshold})
        result = {"schema_version": 1, "status": "complete", "purpose": "development_arcface_calibration_only",
                  "started_utc": started, "completed_utc": now(),
                  "pipeline": {"detector": "official InsightFace SCRFD det_10g ONNX, 640x640, threshold .5, NMS .4",
                               "selection": "exactly one face, or unique IoU >= .5 to clean source box",
                               "alignment": "official face_align.norm_crop BGR112 five-point similarity cv2.warpAffine",
                               "recognizer": "official ArcFaceONNX w600k_r50 BGR input, blob swapRB=True, (RGB-127.5)/127.5",
                               "embedding": "L2-normalized 512D", "comparison": "cosine >= threshold is match",
                               "decode": "EXIF-transposed, ICC-to-sRGB RGB", "provider": "CPUExecutionProvider",
                               "cv2_threads": cv2.getNumThreads(), "ort_threads": 2,
                               "provenance": provenance, "adapter_sha256": digest(OUT / "executed-source.py")},
                  "dataset": {"manifest_sha256": digest(MANIFEST), "split_version": manifest["split_version"],
                              "identity_count": 20, "image_count": 200, "ids": sorted({row["identity"] for row in rows})},
                  "calibration": calibration.as_dict(), "strata": {key: dict(value) for key, value in strata.items()},
                  "leave_one_identity_out": loo, "images": images,
                  "embeddings_sha256": digest(OUT / "embeddings.npz"),
                  "scores_sha256": digest(OUT / "scores.jsonl")}
        write_json(OUT / "calibration.json", result)
        print("ArcFace threshold frozen", calibration.threshold, flush=True)
    except Exception as exc:
        write_json(OUT / "calibration-failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                       "images_completed": len(images), "at_utc": now()})
        raise


def checked_result(embedded, references, calibration, identity):
    gallery = evaluate_gallery(embedded, references, calibration, own_identity=identity,
                               own_source_name=f"{identity}:neutral_front",
                               expected_reference_names=[f"{identity}:{view}" for view in GALLERY_VIEWS])
    return {"status": gallery.status, "reason": gallery.reason,
            "nonmatch": gallery.nonmatch, "own_source_cosine": gallery.own_source_score,
            "maximum_cosine": max(gallery.per_reference.values()) if gallery.per_reference else None,
            "per_reference": dict(gallery.per_reference),
            "matching_references": list(gallery.matching_references),
            "own_identity_matched": gallery.own_identity_matched,
            "detection_count": embedded.detection_count}


def score_phase() -> None:
    cal_path = OUT / "calibration.json"
    if not cal_path.exists() or (OUT / "score.json").exists():
        raise FileNotFoundError("Frozen calibration required and score must be new")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal["status"] != "complete" or cal["pipeline"]["provenance"] != verify_inputs():
        raise ValueError("ArcFace calibration pipeline differs")
    if cal["pipeline"]["adapter_sha256"] != digest(Path(__file__)):
        raise ValueError("Adapter changed after calibration")
    if cal["dataset"]["manifest_sha256"] != digest(MANIFEST):
        raise ValueError("FRLL manifest changed after calibration")
    calibration = Calibration(**cal["calibration"])
    frozen = json.loads((H7 / "frozen.json").read_text(encoding="utf-8"))
    audit = json.loads((H7 / "artifact-audit.json").read_text(encoding="utf-8"))
    if (not frozen["all_outputs_frozen_before_gallery"] or len(frozen["cases"]) != 2
            or {row["identity"] for row in frozen["cases"]} != set(IDS)
            or set(audit["identities"]) != set(IDS)
            or audit["runner_sha256"] != digest(H7 / "executed-source.py")):
        raise ValueError("H7 frozen set differs")
    groups = grouped_images(MANIFEST, "development")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {(row["identity"], row["view"]): row["sha256"] for row in manifest["images"]}
    # All input hashes are verified before any selected JPEG is opened.
    for case in frozen["cases"]:
        identity = case["identity"]
        for view in GALLERY_VIEWS:
            path = groups[identity][view]
            expected = audit["identities"][identity]["gallery"][view]["sha256"]
            if digest(path) != expected or expected != known[(identity, view)]:
                raise ValueError(f"Source/gallery hash changed: {identity}/{view}")
        for arm in ARMS:
            selected = H7 / identity / arm / "selected.jpg"
            if digest(selected) != case["arms"][arm]["selected_jpeg_sha256"]:
                raise ValueError(f"Frozen H7 JPEG hash changed: {identity}/{arm}")
    cv2.setNumThreads(2)
    model = ArcFaceModel()
    started = now()
    results = []
    try:
        for case in frozen["cases"]:
            identity = case["identity"]
            source = decode_image(groups[identity]["neutral_front"])
            clean = model.embed(source)
            if clean.status != "valid":
                raise RuntimeError(f"ArcFace clean source invalid: {identity}: {clean.reason}")
            references = []
            for view in GALLERY_VIEWS:
                embedded = clean if view == "neutral_front" else model.embed(decode_image(groups[identity][view]))
                if embedded.status != "valid":
                    raise RuntimeError(f"ArcFace reference invalid: {identity}/{view}: {embedded.reason}")
                references.append(Reference(f"{identity}:{view}", identity, embedded.feature))
            box = clean.selected_box
            controls = {}
            clean_variants = make_variants(export_jpeg(source), box)
            for condition in CONDITIONS:
                variant = clean_variants[condition]
                embedded = model.embed(variant.image, expected_box=variant.target_box) if variant.status == "valid" else EmbeddingResult("inconclusive", reason=variant.reason)
                controls[condition] = checked_result(embedded, references, calibration, identity)
            arms = {}
            for arm in ARMS:
                selected = H7 / identity / arm / "selected.jpg"
                variants = make_variants(selected.read_bytes(), box)
                conditions = {}
                for condition in CONDITIONS:
                    variant = variants[condition]
                    expected_sha = audit["identities"][identity]["arms"][arm][condition]["sha256"]
                    if variant.jpeg is None or hashlib.sha256(variant.jpeg).hexdigest() != expected_sha:
                        raise ValueError(f"Exact condition bytes differ: {identity}/{arm}/{condition}")
                    embedded = model.embed(variant.image, expected_box=variant.target_box) if variant.status == "valid" else EmbeddingResult("inconclusive", reason=variant.reason)
                    conditions[condition] = checked_result(embedded, references, calibration, identity)
                arms[arm] = {"selected_sha256": digest(selected), "conditions": conditions}
            results.append({"identity": identity, "source_sha256": digest(groups[identity]["neutral_front"]),
                            "clean_controls": controls, "arms": arms})
            print(f"ArcFace scored {identity}", flush=True)
        output = {"schema_version": 1, "status": "complete", "purpose": "development_transfer_H7_only",
                  "started_utc": started, "completed_utc": now(),
                  "calibration_sha256": digest(cal_path), "threshold": calibration.threshold,
                  "h7_frozen_sha256": digest(H7 / "frozen.json"),
                  "h7_artifact_audit_sha256": digest(H7 / "artifact-audit.json"),
                  "conditions": CONDITIONS, "gallery_views": GALLERY_VIEWS,
                  "cases": results}
        write_json(OUT / "score.json", output)
        print("ArcFace H7 scoring complete", flush=True)
    except Exception as exc:
        write_json(OUT / "score-failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                 "identities_completed": len(results), "at_utc": now()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "score"))
    args = parser.parse_args()
    (calibrate_phase if args.phase == "calibrate" else score_phase)()
