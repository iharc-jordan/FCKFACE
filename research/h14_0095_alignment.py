"""H14 source-only 0095 prep and fixed-geometry 128px residual pullback.

Native prep is run separately in the accepted OpenVINO environment. The Torch
optimizer imports only the geometry functions; it never imports OpenVINO.
No gallery view is read here. The exact forward path is BGR uint8, integer
YuNet ROI, the pinned demo's SVD inverse-map *nearest* warp, then linear resize.
The float residual VJP is a documented BPDA approximation, not a JPEG gradient.

FaceIdentifier alignment adapted from OpenVINO Open Model Zoo revision
a6946b6d6ce42cbf4278df20275fab199655fc7d.
Copyright (c) 2018-2024 Intel Corporation. Licensed under Apache-2.0;
see research/NOTICE and the repository LICENSE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.recognition import detector_frame, restore_face_coordinates, select_face


IDS = ("frll-001", "frll-003")
OBJECTIVE = ("export", "jpeg75_420", "blur", "crop90")
TEMPLATE = np.array(((30.2946 / 96, 51.6963 / 112),
                     (65.5318 / 96, 51.5014 / 112),
                     (48.0252 / 96, 71.7366 / 112),
                     (33.5493 / 96, 92.3655 / 112),
                     (62.7299 / 96, 92.2041 / 112)), np.float64)
CAL_SHA = "f33a0e6fa03a4853013eea9fdda9c6c2d0c008a2f6655fd52064bb0187d1275a"
PIPELINE_SHA = "b3e54fef8527dc1e68e026f74668f9f043ffabedbdc685c649ca66c1e8693d05"
ERRATUM_SHA = "ed21096fd501f9620d8c6bdc6860ff059b48ddcf3a9b8b60dd7cfeec776110ef"
YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def clean_row(detector, rgb: np.ndarray, expected_box=None) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    frame = detector_frame(bgr)
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    faces = restore_face_coordinates(faces, (rgb.shape[1], rgb.shape[0]),
                                     (frame.shape[1], frame.shape[0]))
    face, reason = select_face(faces, expected_box)
    if face is None:
        raise ValueError(f"Clean 0095 condition face inconclusive: {reason}")
    return np.asarray(face, np.float32).copy()


def geometry(image_shape: tuple[int, ...], row: np.ndarray) -> dict:
    """Return the official demo's destination-to-ROI-source SVD transform."""
    height, width = image_shape[:2]
    row = np.asarray(row, np.float32).reshape(-1)
    if row.size < 14 or not np.isfinite(row[:14]).all():
        raise ValueError("Invalid frozen YuNet row")
    x, y, w, h = map(float, row[:4])
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, int(x + w)), min(height, int(y + h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty integer-clipped 0095 ROI")
    roi_w, roi_h = x1 - x0, y1 - y0
    scale = np.array((roi_w, roi_h), np.float64)
    desired = TEMPLATE * scale
    relative = ((row[4:14].reshape(5, 2).astype(np.float64) -
                 np.array((x0, y0), np.float64)) / scale)
    actual = relative * scale
    desired_mean, actual_mean = desired.mean(axis=0), actual.mean(axis=0)
    desired_zero, actual_zero = desired - desired_mean, actual - actual_mean
    desired_std, actual_std = desired_zero.std(), actual_zero.std()
    if desired_std <= 0 or actual_std <= 0:
        raise ValueError("Degenerate 0095 landmark similarity")
    desired_zero /= desired_std
    actual_zero /= actual_std
    u, _, vt = np.linalg.svd(desired_zero.T @ actual_zero)
    rotation = (u @ vt).T
    linear = rotation * (actual_std / desired_std)
    offset = actual_mean - linear @ desired_mean
    transform = np.column_stack((linear, offset))
    return {"roi": (x0, y0, x1, y1), "matrix": transform,
            "image_shape": (height, width)}


def exact_input(rgb: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Demo-equivalent uint8 BGR NCHW 128 input; official parity is mandatory."""
    geom = geometry(rgb.shape, row)
    x0, y0, x1, y1 = geom["roi"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    roi = bgr[y0:y1, x0:x1].copy()
    warped = cv2.warpAffine(roi, geom["matrix"], (x1-x0, y1-y0),
                            flags=cv2.WARP_INVERSE_MAP)
    resized = cv2.resize(warped, (128, 128), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized.transpose(2, 0, 1))


def residual_bgr128(delta, carrier, geom: dict, crop_offset: tuple[int, int]):
    """Torch VJP: exact OpenCV nearest index map, then float linear resize.

    `delta` is CHW RGB in the original carrier ROI. The exact JPEG/blur/round
    forward is supplied separately; this fixed spatial path is BPDA. For crop90,
    the variant's pixels are shifted back to original coordinates explicitly.
    """
    import torch
    import torch.nn.functional as F

    x0, y0, x1, y1 = geom["roi"]
    rw, rh = x1-x0, y1-y0
    index_image = np.arange(rw*rh, dtype=np.float32).reshape(rh, rw)
    warped_index = cv2.warpAffine(index_image, geom["matrix"], (rw, rh),
                                  flags=cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    source_index = np.rint(warped_index).astype(np.int64)
    ix = x0 + source_index % rw + crop_offset[0] - carrier.left
    iy = y0 + source_index // rw + crop_offset[1] - carrier.top
    height, width = carrier.roi.shape[:2]
    valid = ((source_index >= 0) & (ix >= 0) & (ix < width) &
             (iy >= 0) & (iy < height))
    gather = np.full(source_index.shape, width*height, dtype=np.int64)
    gather[valid] = iy[valid]*width + ix[valid]
    padded = F.pad(delta.reshape(3, -1), (0, 1), value=0.)
    warped_rgb = padded[:, torch.from_numpy(gather.reshape(-1))].reshape(1, 3, rh, rw)
    resized_rgb = F.interpolate(warped_rgb, size=(128, 128), mode="bilinear",
                                align_corners=False)[0]
    return resized_rgb[[2, 1, 0], :, :]  # official model uses BGR


def cv2_float_residual(delta_rgb: np.ndarray, carrier, geom: dict,
                       crop_offset: tuple[int, int]) -> np.ndarray:
    """Model-free reference for testing the BPDA residual, not an image edit."""
    x0, y0, x1, y1 = geom["roi"]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ix = xx + crop_offset[0] - carrier.left
    iy = yy + crop_offset[1] - carrier.top
    h, w = carrier.roi.shape[:2]
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    roi = np.zeros((y1-y0, x1-x0, 3), np.float32)
    roi[valid] = delta_rgb.transpose(1, 2, 0)[iy[valid], ix[valid]]
    warped = cv2.warpAffine(roi, geom["matrix"], (x1-x0, y1-y0),
                            flags=cv2.WARP_INVERSE_MAP)
    resized = cv2.resize(warped, (128, 128), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))


def synthetic_pullback_check() -> dict:
    """No photos/models: compare float BPDA forward and directional CV2 FD."""
    from types import SimpleNamespace
    import torch

    rng = np.random.default_rng(14095)
    carrier = SimpleNamespace(left=8, top=6, roi=np.zeros((105, 105, 3), np.uint8))
    row = np.array((20.3, 14.5, 68.2, 82.1,
                    36., 46., 68., 45., 52., 64., 40., 81., 65., 80., .99), np.float32)
    geom = geometry((120, 120, 3), row)
    offset = (6, 5)  # exercises crop90's variant-to-original shift
    base = rng.normal(0, .1, (3, 105, 105)).astype(np.float32)
    direction = rng.normal(0, .1, base.shape).astype(np.float32)
    weight = rng.normal(0, 1, (3, 128, 128)).astype(np.float32)
    pytorch = residual_bgr128(torch.from_numpy(base), carrier, geom, offset).detach().numpy()
    native = cv2_float_residual(base, carrier, geom, offset)
    max_forward_difference = float(np.max(np.abs(pytorch-native)))
    theta = torch.tensor(.2, dtype=torch.float32, requires_grad=True)
    result = (residual_bgr128(torch.from_numpy(base) + theta*torch.from_numpy(direction),
                              carrier, geom, offset)*torch.from_numpy(weight)).sum()
    result.backward()
    analytic = float(theta.grad)
    epsilon = .01
    def value(t: float) -> float:
        return float(np.sum(cv2_float_residual(base + t*direction, carrier, geom, offset)*weight))
    numeric = (value(.2+epsilon)-value(.2-epsilon))/(2*epsilon)
    absolute_error = abs(analytic-numeric)
    relative_error = absolute_error/max(abs(numeric), 1.)
    passed = (np.isfinite(analytic) and np.isfinite(numeric) and
              max_forward_difference <= .02 and
              (relative_error <= .1 or absolute_error <= .001))
    return {"passed": bool(passed), "max_float_forward_difference": max_forward_difference,
            "analytic_directional_derivative": analytic,
            "cv2_float_directional_finite_difference": numeric,
            "absolute_error": absolute_error, "relative_error": relative_error,
            "crop_offset": offset, "epsilon": epsilon,
            "limitation": "Fixed-geometry float BPDA only; JPEG and detector are nondifferentiable"}


def run_prep(args) -> None:
    """Write only two native clean source references and clean-condition rows."""
    from openvino_0095_adapter import OpenVINO0095

    repo = Path(__file__).resolve().parents[1]
    if args.out.exists() or args.out.resolve().is_relative_to(repo):
        raise ValueError("0095 prep output must be new and outside Git")
    for path, expected in ((args.calibration, CAL_SHA), (args.pipeline, PIPELINE_SHA),
                           (args.erratum, ERRATUM_SHA), (args.yunet, YUNET_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Wrong frozen prep dependency: {path.name}")
    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (digest(args.manifest) != cal["dataset"]["manifest_sha256"] or
            cal["calibration"]["threshold"] != .43059808165428554):
        raise ValueError("0095 calibration/manifest mismatch")
    sources = {}
    for identity in IDS:
        found = [r for r in manifest["images"] if r["identity"] == identity and
                 r["view"] == "neutral_front" and r["split"] == "development"]
        if len(found) != 1:
            raise ValueError(f"No unique development source: {identity}")
        path = (args.manifest.parent / found[0]["path"]).resolve()
        if not path.is_relative_to(args.manifest.parent.resolve()):
            raise ValueError("Source escaped FRLL dataset root")
        sources[identity] = (path, found[0]["sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", {
        "purpose": "H14 source-only official OpenVINO 0095 prep before optimization",
        "identities": IDS, "source_view": "neutral_front", "conditions": OBJECTIVE,
        "manifest_sha256": digest(args.manifest), "calibration_sha256": CAL_SHA,
        "pipeline_sha256": PIPELINE_SHA, "erratum_sha256": ERRATUM_SHA,
        "yunet_sha256": YUNET_SHA, "helper_sha256": digest(Path(__file__)),
        "sources": {k: v[1] for k, v in sources.items()},
        "gallery_access": "none", "model_scope": "official calibrated development 0095"})
    model = OpenVINO0095(args.artifacts, args.yunet)
    template = np.asarray(model.official.FaceIdentifier.REFERENCE_LANDMARKS, np.float64)
    if not np.array_equal(template, TEMPLATE):
        raise ValueError("Official 0095 demo template differs")
    cases = []
    for identity in IDS:
        path, expected = sources[identity]
        if digest(path) != expected:
            raise ValueError(f"Source changed: {identity}")
        source = decode_image(path)
        # The calibrated 0095 gallery reads original JPEGs with cv2.imread.
        # Keep that source reference exact; condition JPEGs use shared PIL decode.
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != source.shape:
            raise ValueError("Native/PIL source dimensions differ")
        native = model.embed(bgr)
        if native.status != "valid" or native.selected_row is None:
            raise ValueError(f"Invalid 0095 clean source: {identity}/{native.reason}")
        source_row = native.selected_row
        source_tensor = exact_input(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), source_row)
        if hashlib.sha256(source_tensor.tobytes()).hexdigest() != native.input_sha256:
            raise ValueError("Independent clean 0095 input differs from official")
        variants = make_variants(export_jpeg(source), native.selected_box)
        conditions = {}
        for condition in OBJECTIVE:
            variant = variants[condition]
            if variant.status != "valid" or variant.image is None or variant.jpeg is None:
                raise ValueError(f"Invalid clean 0095 condition: {identity}/{condition}")
            row = clean_row(model.detector, variant.image, variant.target_box)
            independent = exact_input(variant.image, row)
            x0, y0, x1, y1 = geometry(variant.image.shape, row)["roi"]
            roi = cv2.cvtColor(variant.image, cv2.COLOR_RGB2BGR)[y0:y1, x0:x1].copy()
            relative = (row[4:14].reshape(5, 2).astype(np.float64) - (x0, y0)) / (x1-x0, y1-y0)
            model.identifier._align_rois([roi], [relative])
            official = model.official.resize_input(roi, [1, 3, 128, 128], True)
            if not np.array_equal(independent, official[0]):
                raise ValueError(f"0095 condition input parity failed: {identity}/{condition}")
            conditions[condition] = {"row": row.tolist(),
                                     "jpeg_sha256": hashlib.sha256(variant.jpeg).hexdigest(),
                                     "input_sha256": hashlib.sha256(independent.tobytes()).hexdigest()}
        feature = np.asarray(native.feature, np.float32)
        cases.append({"identity": identity, "source_sha256": expected,
                      "source_row": source_row.tolist(), "source_box": native.selected_box,
                      "source_cv2_bgr_sha256": hashlib.sha256(bgr.tobytes()).hexdigest(),
                      "source_input_sha256": native.input_sha256,
                      "source_feature": feature.tolist(),
                      "source_feature_sha256": hashlib.sha256(feature.tobytes()).hexdigest(),
                      "conditions": conditions})
    write_json(args.out / "prep.json", {"status": "complete", "purpose": "H14_source_only_0095",
                                        "protocol_sha256": digest(args.out / "protocol.json"),
                                        "artifact_record_sha256": digest(args.artifacts / "artifact-record.json"),
                                        "model_xml_sha256": digest(args.artifacts / "face-reidentification-retail-0095.xml"),
                                        "model_bin_sha256": digest(args.artifacts / "face-reidentification-retail-0095.bin"),
                                        "actual_precision": model.actual_precision,
                                        "cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "calibration", "pipeline", "erratum", "artifacts",
                 "yunet", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    run_prep(parser.parse_args())
