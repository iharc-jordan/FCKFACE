"""Development-only author GhostFaceNet H5 with YuNet face selection.

The serialized author graph is loaded directly; DeepFace's reconstructed graph
is intentionally not used. Model weights and biometric outputs stay external.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Must precede TensorFlow import.

import cv2
import numpy as np
from skimage import transform
import tensorflow as tf

from .imaging import Box
from .recognition import (DETECTOR_SCORE_THRESHOLD, EmbeddingResult,
                          _face_box, detector_frame, restore_face_coordinates,
                          select_face)

TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                     [56.0252, 71.7366], [41.5493, 92.3655],
                     [70.729904, 92.2041]], dtype=np.float32)


def author_align(rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Author's RGB five-point similarity warp and uint8 truncation."""
    fitted = transform.SimilarityTransform()
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid YuNet landmarks")
    if not fitted.estimate(points, TEMPLATE):
        raise ValueError("Similarity transform failed")
    warped = transform.warp(rgb, fitted.inverse, output_shape=(112, 112))
    if warped.shape != (112, 112, 3) or not np.isfinite(warped).all():
        raise ValueError("Invalid aligned crop")
    return (warped * 255).astype(np.uint8)


def author_normalize(crop: np.ndarray) -> np.ndarray:
    if crop.shape != (112, 112, 3) or crop.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 aligned crop")
    return (crop.astype(np.float32) - 127.5) * .0078125


def unit_feature(value: np.ndarray) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if feature.shape != (512,) or not np.isfinite(feature).all() or norm <= 0:
        raise ValueError("Invalid GhostFaceNet feature")
    return feature / norm


class GhostFaceModel:
    """One author H5 graph and one YuNet detector per local process."""

    def __init__(self, detector_path: str | Path, weights_path: str | Path):
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.model = tf.keras.models.load_model(str(weights_path), compile=False)
        if self.model.input_shape != (None, 112, 112, 3) or self.model.output_shape != (None, 512):
            raise ValueError("Unexpected author H5 input or output shape")
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320),
            score_threshold=DETECTOR_SCORE_THRESHOLD)

    def feature_from_crop(self, crop: np.ndarray) -> np.ndarray:
        data = author_normalize(crop)[None, ...]
        return unit_feature(self.model(data, training=False).numpy())

    def selftest(self) -> dict:
        """Synthetic author preprocessing, graph repeatability, and shape check."""
        swatches = np.array([0, 127, 128, 255], np.uint8)
        transformed = (swatches.astype(np.float32) - 127.5) * .0078125
        expected = np.array([-127.5 / 128, -.5 / 128,
                             .5 / 128, 127.5 / 128], np.float32)
        if not np.array_equal(transformed, expected):
            raise ValueError("Author RGB normalization parity failed")
        black = np.zeros((112, 112, 3), np.uint8)
        if author_align(black, TEMPLATE).any():
            raise ValueError("Author identity alignment parity failed")
        crop = (np.arange(112 * 112 * 3, dtype=np.uint32) % 256).astype(np.uint8).reshape(112, 112, 3)
        data = author_normalize(crop)[None, ...]
        if not np.array_equal(data, ((crop.astype(np.float32) - 127.5) / 128)[None, ...]):
            raise ValueError("Author synthetic tensor parity failed")
        first = np.asarray(self.model(data, training=False).numpy(), np.float32).reshape(-1)
        second = np.asarray(self.model(data, training=False).numpy(), np.float32).reshape(-1)
        if not np.array_equal(first, second):
            raise ValueError("Author H5 graph is not repeatable")
        unit_feature(first)
        return {"synthetic_crop_sha256": __import__("hashlib").sha256(crop.tobytes()).hexdigest(),
                "synthetic_raw_first_eight": first[:8].tolist(),
                "synthetic_raw_norm": float(np.linalg.norm(first)),
                "repeat_max_abs": float(np.max(np.abs(first - second))),
                "swatch_normalized": transformed.tolist(),
                "model_input": list(self.model.input_shape),
                "model_output": list(self.model.output_shape),
                "parameter_count": int(self.model.count_params())}

    def embed(self, image_rgb: np.ndarray, expected_box: Box | None = None) -> EmbeddingResult:
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
            frame = detector_frame(bgr)
            fh, fw = frame.shape[:2]
            self.detector.setInputSize((fw, fh))
            _, faces = self.detector.detect(frame)
            faces = restore_face_coordinates(faces, (w, h), (fw, fh))
        except cv2.error:
            return EmbeddingResult("inconclusive", reason="detector_error")
        count = 0 if faces is None else len(faces)
        face, reason = select_face(faces, expected_box)
        if face is None:
            return EmbeddingResult("inconclusive", reason=reason, detection_count=count)
        points = face[4:14].reshape(5, 2).copy()
        try:
            crop = author_align(image_rgb, points)
            feature = self.feature_from_crop(crop)
        except (ValueError, cv2.error, tf.errors.OpError):
            return EmbeddingResult("inconclusive", reason="alignment_or_feature_error",
                                   selected_box=_face_box(face), detection_count=count,
                                   landmarks=points)
        return EmbeddingResult("valid", feature, selected_box=_face_box(face),
                               detection_count=count, landmarks=points)
