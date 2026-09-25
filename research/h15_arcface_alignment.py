"""H15 fixed official ArcFace crop and model-free RGB residual pullback.

The forward path matches pinned InsightFace face_align.norm_crop and
ArcFaceONNX.get_feat preprocessing. Only the additive residual has a surrogate
gradient: JPEG, uint8 rounding, detector and warp interpolation are detached.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage import transform

TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def matrix(landmarks: np.ndarray) -> np.ndarray:
    """Official five-point 112px SimilarityTransform, source -> Arc crop."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite ArcFace landmarks")
    fitted = transform.SimilarityTransform()
    if not fitted.estimate(points, TEMPLATE):
        raise ValueError("ArcFace similarity fit failed")
    return fitted.params[:2, :]


def exact_blob(image_rgb: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """BGR cv2 linear warp/zero border, then official swapRB + normalization."""
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("ArcFace forward expects uint8 RGB")
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    crop = cv2.warpAffine(bgr, affine, (112, 112), borderValue=0.0)
    return cv2.dnn.blobFromImages(
        [crop], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5),
        swapRB=True).astype(np.float32, copy=False)


def residual_blob(delta, carrier, affine: np.ndarray, offset: tuple[int, int]):
    """Differentiable fixed-grid pullback into the exact NCHW RGB blob units."""
    import torch.nn.functional as F
    from optimize_alignment_dots import warp_grid

    forward = np.vstack((affine, [0., 0., 1.]))
    grid = warp_grid(carrier, forward, offset)
    residual_rgb = F.grid_sample(delta.unsqueeze(0), grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=True)
    return residual_rgb / 127.5


def synthetic_pullback_check() -> dict:
    """No recognizer or photograph: finite-difference residual VJP check."""
    import torch

    rng = np.random.default_rng(1515)
    landmarks = TEMPLATE * 1.13 + np.array([19., 23.], np.float32)
    affine = matrix(landmarks)
    class CarrierShape:
        left, top = 12, 10
        roi = np.zeros((146, 150, 3), dtype=np.uint8)
    carrier = CarrierShape()
    offset = (3, 5)
    raw = rng.normal(size=(3, 146, 150)).astype(np.float32)
    weights = torch.from_numpy(rng.normal(size=(1, 3, 112, 112)).astype(np.float32))
    delta = torch.tensor(raw, requires_grad=True)
    objective = (residual_blob(delta, carrier, affine, offset) * weights).sum()
    gradient = torch.autograd.grad(objective, delta)[0].numpy()
    rows = []
    for coord in ((0, 50, 49), (1, 73, 61), (2, 95, 80)):
        plus, minus = raw.copy(), raw.copy()
        plus[coord] += .1
        minus[coord] -= .1
        actual_step = float(plus[coord] - minus[coord])
        with torch.no_grad():
            high = float((residual_blob(torch.from_numpy(plus), carrier, affine, offset)*weights).sum())
            low = float((residual_blob(torch.from_numpy(minus), carrier, affine, offset)*weights).sum())
        fd = (high-low)/actual_step
        rows.append({"coordinate": coord, "analytic": float(gradient[coord]),
                     "finite_difference": fd, "absolute_error": abs(fd-float(gradient[coord]))})
    # Also check the exact forward's channel order and normalization against
    # the official OpenCV blob operation on an unrelated deterministic tensor.
    image = rng.integers(0, 256, size=(180, 184, 3), dtype=np.uint8)
    crop = cv2.warpAffine(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), affine,
                          (112, 112), borderValue=0.0)
    expected = cv2.dnn.blobFromImages([crop], 1/127.5, (112, 112),
                                     (127.5,)*3, swapRB=True)
    forward_equal = bool(np.array_equal(exact_blob(image, affine), expected))
    # Independent geometry oracle: a ramp in ROI x/y must recover the inverse
    # affine source coordinate, plus crop offset, minus the ROI origin.
    yy, xx = np.mgrid[:146, :150].astype(np.float32)
    ramps = torch.from_numpy(np.stack((xx, yy, np.zeros_like(xx))))
    inverse = cv2.invertAffineTransform(affine)
    ramp_rows = []
    with torch.no_grad():
        sampled = residual_blob(ramps, carrier, affine, offset).numpy()[0]
    for x, y in ((56, 56), (45, 76), (73, 43)):
        sx = inverse[0, 0]*x + inverse[0, 1]*y + inverse[0, 2] + offset[0] - carrier.left
        sy = inverse[1, 0]*x + inverse[1, 1]*y + inverse[1, 2] + offset[1] - carrier.top
        if not (1 < sx < 148 and 1 < sy < 144):
            raise AssertionError("Synthetic ramp output point left ROI interior")
        observed = sampled[:2, y, x]
        expected_xy = np.array([sx, sy]) / 127.5
        ramp_rows.append({"output_xy": [x, y], "expected_roi_xy": [float(sx), float(sy)],
                          "observed_normalized": observed.tolist(),
                          "max_abs_error": float(np.max(np.abs(observed-expected_xy)))})
    return {"purpose": "H15 model-free fixed-grid pullback and BGR-to-RGB blob",
            "seed": 1515, "offset": offset, "epsilon": .1,
            "forward_exact_equal": forward_equal, "coordinate_checks": rows,
            "independent_coordinate_ramps": ramp_rows,
            "passed": forward_equal and all(row["absolute_error"] <= 2e-4 for row in rows)
                      and all(row["max_abs_error"] <= 1e-5 for row in ramp_rows)}
