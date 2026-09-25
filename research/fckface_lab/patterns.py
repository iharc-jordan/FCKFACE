"""Deterministic appearance hypotheses for local face-cloaking research.

These are image effects, not a validated privacy method. Coordinates use RGB
uint8 images, XYWH boxes, and YuNet's five-landmark order (eyes, nose, mouth).
No detector, recognizer, file I/O, or model weights are needed here.
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


FACE_REGIONS = ("eyes", "cheeks", "nose", "lower")
METHOD_DEFAULTS: dict[str, dict] = {
    "unedited": {},
    "uniform_dots": {"amplitude": 6.0, "period": 16.0},
    "global_multiscale": {"amplitude": 16.0, "periods": (8.0, 16.0, 24.0)},
    "face_relative_multiscale": {"amplitude": 16.0, "periods": (8.0, 16.0, 24.0)},
    "region_joint": {"amplitude": 22.0, "region_amplitudes": (0.8, 0.8, 0.8, 0.8)},
    "region_eyes": {"amplitude": 22.0},
    "region_cheeks": {"amplitude": 22.0},
    "region_nose": {"amplitude": 22.0},
    "region_lower": {"amplitude": 22.0},
    "graphic_halftone": {"step": 36.0, "strength": 1.0, "selection": ("eyes", "nose", "lower")},
    "uniform_smooth": {"sigma": 5.0, "strength": 1.0, "selection": ("eyes", "nose", "lower")},
    "opaque_tint": {"strength": 0.25, "selection": ("eyes", "nose", "lower")},
    "ordinary_halftone": {"step": 12.0, "strength": 1.0, "selection": ("eyes", "nose", "lower")},
}


def _input(image: np.ndarray, bbox, landmarks):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    box = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if box.size >= 14 and landmarks is None:  # YuNet row: XYWH, five XY pairs, score.
        landmarks = box[4:14].reshape(5, 2)
        box = box[:4]
    if box.size != 4 or not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
        raise ValueError("bbox must be finite XYWH with positive width and height")
    if landmarks is not None:
        landmarks = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
        if not np.isfinite(landmarks).all():
            raise ValueError("landmarks must contain finite coordinates")
    return box, landmarks


def face_mask(image_shape, bbox) -> np.ndarray:
    """Soft oval face support, exactly zero beyond its ellipse."""
    h, w = image_shape[:2]
    x, y, bw, bh = np.asarray(bbox, dtype=np.float64).reshape(-1)[:4]
    yy, xx = np.ogrid[:h, :w]
    distance = np.sqrt(((xx - (x + 0.5 * bw)) / (0.48 * bw)) ** 2 +
                       ((yy - (y + 0.52 * bh)) / (0.53 * bh)) ** 2)
    t = np.clip((1.0 - distance) / 0.09, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _face_coordinates(shape, box, landmarks):
    """Map all five landmarks to an upright, translation-free 128px face."""
    h, w = shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    if landmarks is None:
        x, y, bw, bh = box
        return (xx - x) * (128.0 / bw), (yy - y) * (128.0 / bh)
    x, y, bw, bh = box
    canonical = np.array([
        [0.32 * 128, 0.35 * 128], [0.68 * 128, 0.35 * 128],
        [0.50 * 128, 0.56 * 128], [0.37 * 128, 0.76 * 128],
        [0.63 * 128, 0.76 * 128],
    ], dtype=np.float64)
    a = landmarks - landmarks.mean(axis=0)
    b = canonical - canonical.mean(axis=0)
    denom = float(np.sum(a * a))
    if denom < 1e-6:
        raise ValueError("landmarks do not define a usable face")
    cross = a.T @ b
    u, singular, vt = np.linalg.svd(cross)
    diagonal = np.diag([1.0, np.linalg.det(u @ vt)])
    transform = (u @ diagonal @ vt) * (np.sum(singular * np.diag(diagonal)) / denom)
    # Row-vector convention: p' = (p - source_mean) @ transform + target_mean.
    dx = xx - landmarks[:, 0].mean()
    dy = yy - landmarks[:, 1].mean()
    return (dx * transform[0, 0] + dy * transform[1, 0] + canonical[:, 0].mean(),
            dx * transform[0, 1] + dy * transform[1, 1] + canonical[:, 1].mean())


def _cell_gain(cx, cy, seed, layer):
    """Stable per-cell gains in [0.65, 1]; no process-dependent hash or RNG."""
    a = cx.astype(np.int64).astype(np.uint64)
    b = cy.astype(np.int64).astype(np.uint64)
    z = (a * np.uint64(0x9E3779B185EBCA87) ^
         b * np.uint64(0xC2B2AE3D27D4EB4F) ^
         np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF) ^
         np.uint64(layer * 0x165667B19E3779F9))
    z ^= z >> np.uint64(33)
    z *= np.uint64(0xFF51AFD7ED558CCD)
    z ^= z >> np.uint64(33)
    return 0.65 + 0.35 * ((z & np.uint64(65535)).astype(np.float32) / 65535.0)


def _carrier(xx, yy, period, shape, seed, layer, variable):
    if not np.isfinite(period) or period < 4:
        raise ValueError("carrier period must be at least four pixels")
    cx = np.floor(xx / period)
    cy = np.floor(yy / period)
    fx = xx / period - cx - 0.5
    fy = yy / period - cy - 0.5
    radius = np.sqrt(fx * fx + fy * fy)
    if shape == "dot":
        dark = radius < 0.27
    elif shape == "ring":
        dark = (radius > 0.28) & (radius < 0.40)
    elif shape == "screenprint":
        dark = np.abs(fx + fy) < 0.13
    else:
        raise ValueError(f"unknown carrier shape: {shape}")
    # Approximate zero-mean achromatic tiles; exact metrics are measured after
    # rasterization, mask, headroom limits, and optional export by the runner.
    coverage = {"dot": np.pi * 0.27 ** 2,
                "ring": np.pi * (0.40 ** 2 - 0.28 ** 2),
                "screenprint": 2 * 0.13 - 0.13 ** 2}[shape]
    carrier = (coverage - dark.astype(np.float32)) / (1.0 - coverage)
    return carrier * (_cell_gain(cx, cy, seed, layer) if variable else 1.0)


def _region_masks(shape, box, landmarks):
    h, w = shape[:2]
    x, y, bw, bh = box
    if landmarks is None:
        landmarks = np.array([
            [x + .32 * bw, y + .35 * bh], [x + .68 * bw, y + .35 * bh],
            [x + .50 * bw, y + .56 * bh], [x + .37 * bw, y + .76 * bh],
            [x + .63 * bw, y + .76 * bh]], dtype=np.float64)
    yy, xx = np.ogrid[:h, :w]
    left, right, nose, mouth_l, mouth_r = landmarks
    def blob(cx, cy, sx, sy):
        return np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)).astype(np.float32)
    eyes = np.maximum(blob(*left, .14 * bw, .105 * bh), blob(*right, .14 * bw, .105 * bh))
    cheek_y = .45 * (left[1] + right[1]) + .55 * nose[1]
    cheeks = np.maximum(blob(.5 * (left[0] + nose[0]), cheek_y, .16 * bw, .13 * bh),
                        blob(.5 * (right[0] + nose[0]), cheek_y, .16 * bw, .13 * bh))
    lower = blob(.5 * (mouth_l[0] + mouth_r[0]), .5 * (mouth_l[1] + mouth_r[1]), .25 * bw, .13 * bh)
    out = {"eyes": eyes, "cheeks": cheeks,
           "nose": blob(*nose, .12 * bw, .15 * bh), "lower": lower}
    # Region support is finite; the zero edge is useful for exact preservation.
    return {name: np.clip((value - .18) / .50, 0, 1).astype(np.float32)
            for name, value in out.items()}


def _selection(regions, selection):
    if isinstance(selection, str):
        selection = (selection,)
    unknown = set(selection) - set(FACE_REGIONS)
    if unknown or not selection:
        raise ValueError(f"selection must use face regions {FACE_REGIONS}")
    return np.maximum.reduce([regions[name] for name in selection])


def _apply_delta(image, delta, support):
    # One shared channel change retains the source RGB colour and avoids
    # chromatic clipping, including on dark skin and bright highlights.
    base = image.astype(np.float32)
    requested = np.rint(delta * support)
    negative = base.min(axis=2)
    positive = (255.0 - base).min(axis=2)
    safe = np.clip(requested, -negative, positive).astype(np.int16)
    return (image.astype(np.int16) + safe[..., None]).astype(np.uint8)


def render(image: np.ndarray, bbox, landmarks=None, method="unedited",
           params: Mapping | None = None, seed: int = 0) -> np.ndarray:
    """Return a new RGB array; image pixels beyond face support are unchanged.

    ``bbox`` may be a YuNet row when ``landmarks`` is omitted. ``params``
    overrides the values in ``METHOD_DEFAULTS``. Amplitudes are channel levels,
    capped by the local RGB headroom; no geometry or source pixels are changed.
    """
    box, landmarks = _input(image, bbox, landmarks)
    if method not in METHOD_DEFAULTS:
        raise ValueError(f"unknown method: {method}")
    if method == "unedited":
        return image.copy()
    # A source photo can be much larger than its face. Work only around the
    # selected face so a candidate grid remains cheap and memory bounded.
    x, y, bw, bh = box
    left = max(0, int(np.floor(x - .12 * bw)))
    top = max(0, int(np.floor(y - .12 * bh)))
    right = min(image.shape[1], int(np.ceil(x + 1.12 * bw)))
    bottom = min(image.shape[0], int(np.ceil(y + 1.12 * bh)))
    if right <= left or bottom <= top:
        return image.copy()
    local_box = box - np.array([left, top, 0, 0])
    local_landmarks = landmarks - np.array([left, top]) if landmarks is not None else None
    patch = _render_patch(image[top:bottom, left:right], local_box,
                          local_landmarks, method, params, seed, (left, top))
    result = image.copy()
    result[top:bottom, left:right] = patch
    return result


def _render_patch(image, box, landmarks, method, params, seed, origin):
    settings = {**METHOD_DEFAULTS[method], **dict(params or {})}
    mask = face_mask(image.shape, box)
    if not np.any(mask):
        return image.copy()
    regions = _region_masks(image.shape, box, landmarks)
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    global_xx, global_yy = xx + origin[0], yy + origin[1]

    if method in ("uniform_dots", "global_multiscale", "face_relative_multiscale") or method.startswith("region_"):
        if method == "uniform_dots":
            carrier = _carrier(global_xx, global_yy, float(settings["period"]), "dot", seed, 0, False)
            support = mask
        else:
            if method == "global_multiscale":
                # Equal nominal cycles per face width, but phase stays tied
                # to the image origin rather than the selected face.
                uv = global_xx * (128.0 / box[2]), global_yy * (128.0 / box[2])
            else:
                uv = _face_coordinates(image.shape, box, landmarks)
            periods = tuple(float(v) for v in settings.get("periods", (8.0, 16.0, 24.0)))
            shapes = tuple(settings.get("shapes", ("dot", "ring", "screenprint")))
            if len(periods) != len(shapes) or not periods:
                raise ValueError("periods and shapes must have the same nonzero length")
            layers = [_carrier(*uv, period, shape, seed, i, True)
                      for i, (period, shape) in enumerate(zip(periods, shapes))]
            carrier = sum(layers) / len(layers)
            if method.startswith("region_"):
                if method == "region_joint":
                    gains = np.asarray(settings["region_amplitudes"], dtype=np.float32)
                    if gains.shape != (4,) or not np.isfinite(gains).all() or np.any((gains < 0) | (gains > 1)):
                        raise ValueError("region_amplitudes must be four values in [0, 1]")
                    region_gain = sum(g * regions[name] for g, name in zip(gains, FACE_REGIONS))
                    region_gain = np.clip(region_gain, 0, 1)
                else:
                    region_gain = regions[method.removeprefix("region_")]
                support = mask * region_gain
            else:
                support = mask
        amplitude = float(settings["amplitude"])
        if not np.isfinite(amplitude) or not 0 <= amplitude <= 64:
            raise ValueError("amplitude must be in [0, 64]")
        return _apply_delta(image, amplitude * carrier, support)

    support = mask * _selection(regions, settings["selection"])
    gray = image.astype(np.float32).mean(axis=2)
    strength = float(settings["strength"])
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    if method in ("graphic_halftone", "ordinary_halftone"):
        step = float(settings["step"])
        if not np.isfinite(step) or not 4 <= step <= 64:
            raise ValueError("step must be in [4, 64]")
        if method == "graphic_halftone":
            # Visible round screen dots in canonical face coordinates.
            ux, uy = _face_coordinates(image.shape, box, landmarks)
            fx = ux / 10.0 - np.floor(ux / 10.0) - .5
            fy = uy / 10.0 - np.floor(uy / 10.0) - .5
            threshold = np.clip(np.pi * (fx * fx + fy * fy), 0, 1)
        else:
            # Ordinary 4x4 ordered dithering, with the *same* tone step and
            # region coverage, is a matched structural control.
            bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                              [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32)
            threshold = (bayer[global_yy.astype(np.int32) % 4, global_xx.astype(np.int32) % 4] + .5) / 16
        quotient = gray / step
        screened = step * (np.floor(quotient) + (quotient % 1 > threshold))
        delta = (screened - gray) * strength
    elif method == "uniform_smooth":
        sigma = float(settings["sigma"])
        if not np.isfinite(sigma) or not 0 < sigma <= 16:
            raise ValueError("sigma must be in (0, 16]")
        smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        delta = (smoothed - gray) * strength
    else:  # opaque_tint: uniform gray-tinted overlay, with original chroma.
        delta = (float(np.median(gray[mask > .5])) - gray) * strength
    return _apply_delta(image, delta, support)


def actual_distortion(original: np.ndarray, edited: np.ndarray, bbox,
                      landmarks=None) -> dict[str, float | int]:
    """Measure the actual uint8 RGB result in the face support."""
    box, _ = _input(original, bbox, landmarks)
    if edited.shape != original.shape or edited.dtype != np.uint8:
        raise ValueError("edited must have the same uint8 RGB shape")
    left, top, right, bottom = _face_roi(original.shape, box)
    local_box = box - np.array([left, top, 0, 0])
    support = face_mask(original[top:bottom, left:right].shape, local_box) > 0
    patch_difference = (edited[top:bottom, left:right].astype(np.float32) -
                        original[top:bottom, left:right].astype(np.float32))
    face = patch_difference[support]
    changed_patch = np.any(patch_difference != 0, axis=2)
    changed_all = int(np.count_nonzero(np.any(edited != original, axis=2)))
    changed_face = int((changed_patch & support).sum())
    squared_sum = float(np.sum(patch_difference * patch_difference, dtype=np.float64))
    # Pixels beyond the selected face are normally identical. Account for an
    # arbitrary edited input as well, so this remains an honest measurement.
    if changed_all > int(changed_patch.sum()):
        outside_patch = edited.astype(np.float32) - original.astype(np.float32)
        squared_sum = float(np.sum(outside_patch * outside_patch, dtype=np.float64))
    return {
        "face_rms": float(np.sqrt(np.mean(face * face))) if face.size else 0.0,
        "image_rms": float(np.sqrt(squared_sum / original.size)),
        "face_support_pixels": int(support.sum()),
        "changed_face_pixels": changed_face,
        "changed_fraction_face": float(changed_face / max(1, support.sum())),
        "changed_outside_face_pixels": changed_all - changed_face,
    }


def _face_roi(shape, box):
    x, y, bw, bh = box
    left = max(0, int(np.floor(x - .12 * bw)))
    top = max(0, int(np.floor(y - .12 * bh)))
    right = min(shape[1], int(np.ceil(x + 1.12 * bw)))
    bottom = min(shape[0], int(np.ceil(y + 1.12 * bh)))
    return left, top, right, bottom


def scale_to_rms(original: np.ndarray, proposal: np.ndarray, bbox,
                 target_rms: float, landmarks=None, tolerance: float = 0.15):
    """Dim an existing proposal toward an achievable face RMS; report mismatch.

    The factor is restricted to [0, 1]. Quantized effects cannot be safely
    amplified beyond the proposal: an unattainable target is reported, rather
    than silently replacing their structure with arbitrary extra distortion.
    """
    if not np.isfinite(target_rms) or target_rms < 0:
        raise ValueError("target_rms must be finite and nonnegative")
    baseline = actual_distortion(original, proposal, bbox, landmarks)
    if target_rms >= baseline["face_rms"]:
        result = proposal.copy()
    elif target_rms == 0:
        result = original.copy()
    else:
        box, _ = _input(original, bbox, landmarks)
        left, top, right, bottom = _face_roi(original.shape, box)
        base = original[top:bottom, left:right].astype(np.float32)
        delta = proposal[top:bottom, left:right].astype(np.float32) - base
        local_box = box - np.array([left, top, 0, 0])
        support = face_mask(base.shape, local_box) > 0
        base_face = base[support]
        delta_face = delta[support]
        lo, hi = 0.0, 1.0
        for _ in range(16):
            mid = (lo + hi) / 2
            trial_face = np.clip(np.rint(base_face + mid * delta_face), 0, 255)
            face_difference = trial_face - base_face
            rms = float(np.sqrt(np.mean(face_difference * face_difference))) if face_difference.size else 0.0
            if rms < target_rms:
                lo = mid
            else:
                hi = mid
        result = original.copy()
        result[top:bottom, left:right] = np.clip(np.rint(base + hi * delta), 0, 255).astype(np.uint8)
    metrics = actual_distortion(original, result, bbox, landmarks)
    metrics.update(target_face_rms=float(target_rms),
                   rms_mismatch=float(metrics["face_rms"] - target_rms),
                   rms_matched=bool(abs(metrics["face_rms"] - target_rms) <= tolerance))
    return result, metrics
