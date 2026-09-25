"""Export two permitted development SFace inputs and native reference features.

Run with the old CPU Python environment. Outputs belong outside the repository.
No held-out identity is opened, and no photograph is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--yunet", required=True, type=Path)
    parser.add_argument("--sface", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    yunet = args.yunet.resolve(strict=True)
    sface = args.sface.resolve(strict=True)
    out = args.out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Test images and embeddings must stay outside the repository")
    out.mkdir(parents=True, exist_ok=True)

    records = json.loads(manifest.read_text(encoding="utf-8"))["images"]
    candidates = sorted((row for row in records if row["split"] == "development"
                         and row["view"] == "neutral_front"), key=lambda row: row["identity"])
    detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), score_threshold=0.9)
    recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
    result = []
    for row in candidates:
        source = (manifest.parent / row["path"]).resolve(strict=True)
        if not source.is_relative_to(manifest.parent.resolve()):
            raise ValueError("Source escaped the permitted dataset directory")
        if sha(source) != row["sha256"]:
            raise ValueError("Source checksum mismatch")
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        height, width = bgr.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(bgr)
        if faces is None or len(faces) != 1:
            continue
        aligned = recognizer.alignCrop(bgr, faces[0])
        if aligned.shape != (112, 112, 3) or aligned.dtype != np.uint8:
            continue
        reference = np.asarray(recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        if reference.shape != (128,) or not np.isfinite(reference).all():
            continue
        # OpenCV 4.13 FaceRecognizerSF.feature uses scale=1, mean=0,
        # swapRB=true, crop=false: NCHW RGB float32 in the 0..255 range.
        tensor = cv2.dnn.blobFromImage(aligned, 1, (112, 112), (0, 0, 0), True, False)
        if tensor.shape != (1, 3, 112, 112):
            raise ValueError("Unexpected SFace input tensor shape")
        tensor = np.ascontiguousarray(tensor.astype("<f4", copy=False))
        label = f"case{len(result)}"
        crop_path = out / f"{label}.png"
        tensor_path = out / f"{label}.f32"
        cv2.imwrite(str(crop_path), aligned)
        tensor_path.write_bytes(tensor.tobytes())
        result.append({
            "label": label,
            "identity": row["identity"],
            "split": "development",
            "source_sha256": row["sha256"],
            "crop_sha256": sha(crop_path),
            "tensor_sha256": sha(tensor_path),
            "tensor_shape": [1, 3, 112, 112],
            "tensor_dtype": "float32-le",
            "reference_feature": reference.tolist(),
        })
        if len(result) == 2:
            break
    if len(result) != 2:
        raise RuntimeError("Fewer than two single-face development images were valid")
    report = {
        "scope": "Native SFace forward parity only; no match or privacy claim",
        "opencv_version": cv2.__version__,
        "recognizer_sha256": sha(sface),
        "detector_sha256": sha(yunet),
        "cases": result,
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cases": [row["identity"] for row in result],
                      "recognizer_sha256": report["recognizer_sha256"],
                      "opencv_version": cv2.__version__, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
