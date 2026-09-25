"""Local development-only native YuNet/SFace references for browser parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--yunet", required=True, type=Path)
    parser.add_argument("--sface", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve(strict=True)
    out = args.out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Test images, tensors and embeddings must stay outside Git")
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(dataset.read_text(encoding="utf-8"))["images"]
    detector = cv2.FaceDetectorYN.create(str(args.yunet), "", (320, 320), score_threshold=.9)
    recognizer = cv2.FaceRecognizerSF.create(str(args.sface), "")
    cases = []
    for index, identity in enumerate(("frll-024", "frll-036")):
        row = next(r for r in entries if r["identity"] == identity and r["view"] == "neutral_front" and r["split"] == "development")
        source = (dataset.parent / row["path"]).resolve(strict=True)
        if not source.is_relative_to(dataset.parent.resolve()) or digest(source) != row["sha256"]:
            raise ValueError("Source path or hash mismatch")
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot decode {identity}")
        height, width = bgr.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        detector_w = max(1, round(width * scale))
        detector_h = max(1, round(height * scale))
        frame = cv2.resize(bgr, (detector_w, detector_h), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        detector.setInputSize((detector_w, detector_h))
        _, faces = detector.detect(frame)
        if faces is None or len(faces) != 1:
            raise ValueError(f"Expected one native detection for {identity}, found {0 if faces is None else len(faces)}")
        face = faces[0].copy()
        mapped = face.copy()
        mapped[[0, 2, 4, 6, 8, 10, 12]] *= width / detector_w
        mapped[[1, 3, 5, 7, 9, 11, 13]] *= height / detector_h
        aligned = recognizer.alignCrop(bgr, mapped)
        feature = np.asarray(recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        pad_w = (detector_w + 31) // 32 * 32
        pad_h = (detector_h + 31) // 32 * 32
        padded = cv2.copyMakeBorder(frame, 0, pad_h - detector_h, 0, pad_w - detector_w,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        blob = cv2.dnn.blobFromImage(padded)
        prefix = f"yunet{index}"
        files = {
            "source_png": out / f"{prefix}-source.png",
            "detector_blob": out / f"{prefix}-detector.f32",
            "aligned_png": out / f"{prefix}-aligned.png",
        }
        cv2.imwrite(str(files["source_png"]), bgr)
        cv2.imwrite(str(files["aligned_png"]), aligned)
        files["detector_blob"].write_bytes(np.ascontiguousarray(blob.astype("<f4", copy=False)).tobytes())
        cases.append({
            "label": prefix, "identity": identity, "split": "development", "source_sha256": row["sha256"],
            "original_size": [width, height], "detector_size": [detector_w, detector_h],
            "padded_size": [pad_w, pad_h], "detector_shape": [1, 3, pad_h, pad_w],
            "native_detection_frame": face.tolist(), "native_detection_mapped": mapped.tolist(),
            "native_feature": feature.tolist(),
            "assets": {name: {"name": path.name, "sha256": digest(path)} for name, path in files.items()},
        })
    report = {"scope": "Development YuNet/SFace pipeline parity only; no privacy claim",
              "opencv_version": cv2.__version__, "detector_sha256": digest(args.yunet),
              "recognizer_sha256": digest(args.sface), "cases": cases}
    (out / "yunet-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cases": [c["identity"] for c in cases], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
