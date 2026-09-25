"""Native OpenCV YuNet detection and SFace alignment/embedding.

Both ONNX paths are explicit, external inputs. Do not copy model weights into
this repository. This model supplies development evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .imaging import Box

DETECTOR_MAX_SIDE = 640
DETECTOR_SCORE_THRESHOLD = .9


@dataclass(frozen=True)
class EmbeddingResult:
    status: str  # valid, invalid, or inconclusive
    feature: np.ndarray | None = None
    reason: str | None = None
    selected_box: Box | None = None
    detection_count: int = 0
    landmarks: np.ndarray | None = None  # selected YuNet eyes/nose/mouth (5, 2)


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _face_box(face: np.ndarray) -> Box:
    x, y, w, h = map(float, face[:4])
    return (x, y, x + w, y + h)


def detector_frame(bgr: np.ndarray) -> np.ndarray:
    """Downscale once for stable YuNet scoring; retain original for SFace."""
    h, w = bgr.shape[:2]
    scale = min(1.0, DETECTOR_MAX_SIDE / max(w, h))
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return (cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
            if scale < 1 else bgr)


def restore_face_coordinates(faces: np.ndarray | None,
                             original_size: tuple[int, int],
                             detector_size: tuple[int, int]) -> np.ndarray | None:
    """Map YuNet XYWH and five landmarks to original-image pixel space."""
    if faces is None:
        return None
    original_w, original_h = original_size
    detector_w, detector_h = detector_size
    if (original_w, original_h) == (detector_w, detector_h):
        return faces
    result = faces.copy()
    result[:, [0, 2, 4, 6, 8, 10, 12]] *= original_w / detector_w
    result[:, [1, 3, 5, 7, 9, 11, 13]] *= original_h / detector_h
    return result


def select_face(faces: np.ndarray | None, expected_box: Box | None = None
                ) -> tuple[np.ndarray | None, str | None]:
    """Require one face or a unique, strong overlap with the expected target."""
    if faces is None or len(faces) == 0:
        return None, "no_face"
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] < 15
            or not np.all(np.isfinite(faces))
            or np.any(faces[:, 2:4] <= 0)):
        return None, "invalid_detection"
    if expected_box is None:
        return (faces[0], None) if len(faces) == 1 else (None, "multiple_faces")
    if (len(expected_box) != 4 or not all(np.isfinite(expected_box))
            or expected_box[0] >= expected_box[2]
            or expected_box[1] >= expected_box[3]):
        return None, "invalid_target_box"
    ranks = sorted(((_iou(_face_box(f), expected_box), i)
                    for i, f in enumerate(faces)), reverse=True)
    if ranks[0][0] < .5:
        return None, "target_not_found"
    if len(ranks) > 1 and ranks[0][0] - ranks[1][0] < .2:
        return None, "ambiguous_target"
    return faces[ranks[0][1]], None


class SFaceModel:
    """Single-process local model; instantiate once per experiment process."""

    def __init__(self, detector_path: str | Path, recognizer_path: str | Path):
        detector_path = Path(detector_path)
        recognizer_path = Path(recognizer_path)
        if not detector_path.is_file() or not recognizer_path.is_file():
            raise FileNotFoundError("Both YuNet and SFace ONNX paths must exist")
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320),
            score_threshold=DETECTOR_SCORE_THRESHOLD,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def embed(self, image_rgb: np.ndarray, expected_box: Box | None = None
              ) -> EmbeddingResult:
        if (not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8
                or image_rgb.ndim != 3 or image_rgb.shape[2] != 3
                or min(image_rgb.shape[:2]) < 16):
            return EmbeddingResult("invalid", reason="invalid_rgb_image")
        h, w = image_rgb.shape[:2]
        if expected_box is not None and (
                len(expected_box) != 4 or not all(np.isfinite(expected_box))
                or not (0 <= expected_box[0] < expected_box[2] <= w)
                or not (0 <= expected_box[1] < expected_box[3] <= h)):
            return EmbeddingResult("invalid", reason="invalid_target_box")
        try:
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            detector_input = detector_frame(bgr)
            detector_h, detector_w = detector_input.shape[:2]
            self.detector.setInputSize((detector_w, detector_h))
            _, faces = self.detector.detect(detector_input)
            faces = restore_face_coordinates(faces, (w, h), (detector_w, detector_h))
        except cv2.error:
            return EmbeddingResult("inconclusive", reason="detector_error")
        count = 0 if faces is None else len(faces)
        face, reason = select_face(faces, expected_box)
        if face is None:
            return EmbeddingResult("inconclusive", reason=reason, detection_count=count)
        try:
            aligned = self.recognizer.alignCrop(bgr, face)
            feature = np.asarray(self.recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        except (cv2.error, ValueError, TypeError):
            return EmbeddingResult("inconclusive", reason="alignment_or_feature_error",
                                   selected_box=_face_box(face), detection_count=count,
                                   landmarks=face[4:14].reshape(5, 2).copy())
        norm = float(np.linalg.norm(feature))
        if not np.all(np.isfinite(feature)) or not np.isfinite(norm) or norm <= 0:
            return EmbeddingResult("inconclusive", reason="invalid_feature",
                                   selected_box=_face_box(face), detection_count=count,
                                   landmarks=face[4:14].reshape(5, 2).copy())
        return EmbeddingResult("valid", feature / norm, selected_box=_face_box(face),
                               detection_count=count,
                               landmarks=face[4:14].reshape(5, 2).copy())
