"""Private development-only Intel OMZ 0095 adapter; no calibration policy here.

Alignment follows Intel's Apache-2.0 FaceIdentifier demo (OMZ commit
a6946b6d6ce42cbf4278df20275fab199655fc7d), loaded from the verified
private snapshot. YuNet640 replaces the demo detector/landmark regressor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from fckface_lab.recognition import detector_frame, restore_face_coordinates, select_face
from openvino_development import MODEL_NAME, load_official_demo, verify_artifacts


YUNET_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'


@dataclass(frozen=True)
class Result:
    status: str
    reason: str | None = None
    feature: np.ndarray | None = None
    raw_sha256: str | None = None
    input_sha256: str | None = None
    selected_row: np.ndarray | None = None
    detection_count: int = 0
    selected_box: tuple[float, float, float, float] | None = None
    landmarks: np.ndarray | None = None


class OpenVINO0095:
    def __init__(self, artifacts: Path, yunet: Path):
        verify_artifacts(artifacts)
        if not yunet.is_file() or hashlib.sha256(yunet.read_bytes()).hexdigest() != YUNET_SHA256:
            raise ValueError('Wrong YuNet model')
        cv2.setNumThreads(2)
        self.detector = cv2.FaceDetectorYN.create(str(yunet), '', (320, 320), score_threshold=.9)
        self.official = load_official_demo(artifacts)
        self.identifier = self.official.FaceIdentifier.__new__(self.official.FaceIdentifier)
        self.core = ov.Core()
        model = self.core.read_model(str(artifacts / f'{MODEL_NAME}.xml'))
        if len(model.inputs) != 1 or len(model.outputs) != 1 or list(model.inputs[0].shape) != [1, 3, 128, 128]:
            raise ValueError('Wrong official model signature')
        self.compiled = self.core.compile_model(model, 'CPU',
                                                {'INFERENCE_NUM_THREADS': 2,
                                                 'INFERENCE_PRECISION_HINT': 'f32'})
        self.actual_precision = str(self.compiled.get_property('INFERENCE_PRECISION_HINT'))
        self.actual_threads = int(self.compiled.get_property('INFERENCE_NUM_THREADS'))
        if self.actual_precision != "<Type: 'float32'>" or self.actual_threads != 2:
            raise ValueError(f'Unexpected OpenVINO CPU configuration: {self.actual_precision}/{self.actual_threads}')

    def embed(self, bgr: np.ndarray, expected_box=None) -> Result:
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
                or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 16):
            return Result('invalid', 'invalid_bgr_image')
        height, width = bgr.shape[:2]
        if expected_box is not None and (len(expected_box) != 4 or
                not all(np.isfinite(expected_box)) or
                not (0 <= expected_box[0] < expected_box[2] <= width) or
                not (0 <= expected_box[1] < expected_box[3] <= height)):
            return Result('invalid', 'invalid_target_box')
        try:
            frame = detector_frame(bgr)
            frame_h, frame_w = frame.shape[:2]
            self.detector.setInputSize((frame_w, frame_h))
            _, faces = self.detector.detect(frame)
            faces = restore_face_coordinates(faces, (width, height), (frame_w, frame_h))
        except cv2.error:
            return Result('inconclusive', 'detector_error')
        count = 0 if faces is None else len(faces)
        face, reason = select_face(faces, expected_box)
        if face is None:
            return Result('inconclusive', reason, detection_count=count)
        row = np.asarray(face, np.float32).copy()
        x, y, w, h = map(float, row[:4])
        box = (x, y, x + w, y + h)
        landmarks = row[4:14].reshape(5, 2).copy()
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x + w)), min(height, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return Result('inconclusive', 'empty_roi', selected_row=row,
                          detection_count=count, selected_box=box, landmarks=landmarks)
        try:
            roi = bgr[y0:y1, x0:x1].copy()
            relative = (landmarks.astype(np.float64) - np.array((x0, y0))) / np.array((x1 - x0, y1 - y0))
            self.identifier._align_rois([roi], [relative])
            tensor = self.official.resize_input(roi, [1, 3, 128, 128], True)
            if tensor.dtype != np.uint8 or tensor.shape != (1, 3, 128, 128):
                raise ValueError('Unexpected official demo input tensor')
            raw = np.asarray(self.compiled([tensor.astype(np.float32)])[self.compiled.output(0)],
                             np.float32).reshape(-1)
            norm = float(np.linalg.norm(raw))
            if raw.shape != (256,) or not np.isfinite(raw).all() or not np.isfinite(norm) or norm <= 0:
                raise ValueError('Invalid OpenVINO embedding')
        except (cv2.error, ValueError, RuntimeError) as exc:
            return Result('inconclusive', f'alignment_or_feature_error:{type(exc).__name__}',
                          selected_row=row, detection_count=count, selected_box=box, landmarks=landmarks)
        return Result('valid', feature=raw / norm,
                      raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                      input_sha256=hashlib.sha256(tensor.tobytes()).hexdigest(),
                      selected_row=row, detection_count=count,
                      selected_box=box, landmarks=landmarks)
