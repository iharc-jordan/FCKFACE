"""Verify Intel OMZ 0095 artifacts and demo-exact preprocessing on two prior development faces.

No calibration, scoring of edited images, browser conversion, or held-out access.
All biometric outputs are written outside the repository. The official demo source
is loaded from the separately hashed, private Open Model Zoo snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

os.environ.setdefault('OPENVINO_TELEMETRY', '0')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import cv2
import numpy as np
import openvino as ov


IDS = ('frll-024', 'frll-036')
OMZ_COMMIT = 'a6946b6d6ce42cbf4278df20275fab199655fc7d'
MODEL_NAME = 'face-reidentification-retail-0095'
README_LANDMARKS = np.array([
    (.31556875, .4615741071428571), (.6826229166666667, .4615741071428571),
    (.5002625, .6405053571428571), (.349471875, .8246919642857142),
    (.6534364583333333, .8246919642857142),
], np.float64)


def digest(path: Path, algorithm: str = 'sha256') -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def verify_artifacts(root: Path) -> dict:
    record = load_json(root / 'artifact-record.json')
    if record['omz_commit'] != OMZ_COMMIT or record['model_format'] != 'OpenVINO IR FP32 XML plus BIN':
        raise ValueError('Wrong official snapshot or model format')
    for item in record['files']:
        path = root / item['name']
        if path.resolve(strict=True).parent != root.resolve(strict=True):
            raise ValueError('Artifact escaped private directory')
        if path.stat().st_size != item['size'] or digest(path) != item['sha256']:
            raise ValueError(f'Artifact changed: {item["name"]}')
        if item.get('sha384') and digest(path, 'sha384') != item['sha384']:
            raise ValueError(f'Official SHA384 mismatch: {item["name"]}')
    manifest = (root / 'model.yml').read_text(encoding='utf-8')
    for item in record['files']:
        if item['name'] in (f'{MODEL_NAME}.xml', f'{MODEL_NAME}.bin'):
            if item['url'] not in manifest or item['sha384'] not in manifest:
                raise ValueError('Official metadata does not identify model artifact')
    if 'license: https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/LICENSE' not in manifest:
        raise ValueError('Official artifact license metadata changed')
    return record


def load_official_demo(root: Path):
    """Import unmodified pinned demo code from private files, without installing OMZ."""
    for name in ('model_api', 'model_api.models'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    load('model_api.models.utils', root / 'model_api_utils.py')
    load('utils', root / 'face_demo_utils.py')
    load('ie_module', root / 'ie_module.py')
    return load('face_identifier', root / 'face_identifier.py')


def independent_align(roi: np.ndarray, landmarks_rel: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Independent SVD port of the official FaceIdentifier affine and resize."""
    height, width = roi.shape[:2]
    size = np.array((width, height))
    desired = np.asarray(template, np.float64) * size
    actual = np.asarray(landmarks_rel, np.float64) * size
    desired_mean = desired.mean(axis=0)
    actual_mean = actual.mean(axis=0)
    desired_zero = desired - desired_mean
    actual_zero = actual - actual_mean
    desired_std = desired_zero.std()
    actual_std = actual_zero.std()
    u, _, vt = np.linalg.svd(desired_zero.T @ actual_zero)
    rotation = (u @ vt).T
    linear = rotation * (actual_std / desired_std)
    offset = actual_mean - linear @ desired_mean
    transform = np.column_stack((linear, offset))
    aligned = cv2.warpAffine(roi, transform, (width, height), flags=cv2.WARP_INVERSE_MAP)
    return cv2.resize(aligned, (128, 128), interpolation=cv2.INTER_LINEAR)


def run(args: argparse.Namespace) -> None:
    cv2.setNumThreads(2)
    root = args.artifacts.resolve(strict=True)
    record = verify_artifacts(root)
    dest = args.output.resolve()
    if dest.exists() or dest.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Output must be a new external directory')
    prior = load_json(args.combined_manifest)
    frll = load_json(args.frll_manifest)
    if [case['identity'] for case in prior['cases']] != list(IDS):
        raise ValueError('Only the two prior development identities are allowed')
    official = load_official_demo(root)
    template = np.asarray(official.FaceIdentifier.REFERENCE_LANDMARKS, np.float64)
    if np.allclose(template, README_LANDMARKS, atol=1e-12):
        raise ValueError('Unexpected: demo template no longer differs from README')
    core = ov.Core()
    model = core.read_model(str(root / f'{MODEL_NAME}.xml'))
    if len(model.inputs) != 1 or len(model.outputs) != 1 or list(model.inputs[0].shape) != [1, 3, 128, 128]:
        raise ValueError('Unexpected IR tensor signature')
    compiled = core.compile_model(model, 'CPU', {'INFERENCE_NUM_THREADS': 2})
    dest.mkdir(parents=True)
    cases = []
    for prior_case in prior['cases']:
        started = time.perf_counter()
        identity = prior_case['identity']
        entry = next((row for row in frll['images'] if row['identity'] == identity and
                      row['view'] == 'neutral_front' and row['split'] == 'development'), None)
        if entry is None or entry['sha256'] != prior_case['source_jpeg_sha256']:
            raise ValueError('Source split/hash mismatch')
        source = (args.frll_manifest.parent / entry['path']).resolve(strict=True)
        if not source.is_relative_to(args.frll_manifest.parent.resolve()) or digest(source) != entry['sha256']:
            raise ValueError('Source escaped dataset or changed')
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        prior_path = args.combined_manifest.parent / prior_case['portrait_png']
        if bgr is None or digest(prior_path) != prior_case['portrait_png_sha256']:
            raise ValueError('Decoded portrait unavailable or prior fixture changed')
        prior_bgr = cv2.imread(str(prior_path), cv2.IMREAD_COLOR)
        if prior_bgr is None or not np.array_equal(bgr, prior_bgr):
            raise ValueError('Current JPEG decoding differs from frozen prior portrait')
        mapped = np.asarray(prior_case['mapped_yunet_row_xywh_landmarks_score'], np.float32)
        x, y, width, height = map(float, mapped[:4])
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(bgr.shape[1], int(x + width)), min(bgr.shape[0], int(y + height))
        if x1 <= x0 or y1 <= y0:
            raise ValueError('Empty YuNet ROI')
        roi = bgr[y0:y1, x0:x1].copy()
        landmarks = mapped[4:14].reshape(5, 2).astype(np.float64)
        relative = (landmarks - np.array((x0, y0))) / np.array((x1 - x0, y1 - y0))
        official_roi = roi.copy()
        identifier = official.FaceIdentifier.__new__(official.FaceIdentifier)
        identifier._align_rois([official_roi], [relative.copy()])
        official_nchw = official.resize_input(official_roi, [1, 3, 128, 128], True)
        independent = independent_align(roi, relative, template)
        same = np.array_equal(official_nchw[0].transpose(1, 2, 0), independent)
        max_pixel = int(np.max(np.abs(official_nchw[0].transpose(1, 2, 0).astype(np.int16) -
                                      independent.astype(np.int16))))
        if not same or official_nchw.dtype != np.uint8:
            raise ValueError(f'Official-demo alignment parity failed for {identity}: {max_pixel}')
        output = compiled([official_nchw.astype(np.float32)])[compiled.output(0)]
        raw = np.asarray(output, np.float32).reshape(-1)
        if raw.shape != (256,) or not np.isfinite(raw).all() or np.linalg.norm(raw) == 0:
            raise ValueError('Invalid official model embedding')
        crop_path = dest / f'{identity}-official-demo-crop.png'
        if not cv2.imwrite(str(crop_path), official_nchw[0].transpose(1, 2, 0)):
            raise ValueError('Could not save private aligned crop')
        cases.append({'identity': identity, 'split': 'development', 'source_jpeg_sha256': entry['sha256'],
                      'yunet_roi_xyxy': [x0, y0, x1, y1], 'yunet_landmarks_xy': landmarks.tolist(),
                      'landmarks_relative_to_integer_roi': relative.tolist(), 'official_demo_crop_sha256': digest(crop_path),
                      'aligned_input_nchw_sha256': hashlib.sha256(official_nchw.tobytes()).hexdigest(),
                      'official_vs_port_pixel_max': max_pixel, 'official_vs_port_exact': same,
                      'embedding_l2_norm': float(np.linalg.norm(raw)),
                      'embedding_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                      'elapsed_seconds': time.perf_counter() - started})
    report = {'schema_version': 1, 'purpose': 'official_demo_alignment_parity_two_prior_development_faces_only',
              'not_calibration_or_privacy_evidence': True, 'omz_commit': OMZ_COMMIT,
              'artifact_record_sha256': digest(root / 'artifact-record.json'),
              'frll_manifest_sha256': digest(args.frll_manifest),
              'combined_manifest_sha256': digest(args.combined_manifest),
              'detector': 'YuNet max-side 640, substituted for OMZ detector and landmark regressor; prior frozen rows reused',
              'alignment': 'Pinned FaceIdentifier._align_rois on integer-truncated YuNet XYWH ROI; cv2 warpAffine inverse-map; cv2 linear resize',
              'input': 'BGR uint8 aligned crop -> 128x128 NCHW, cast float32 0..255, no external mean/scale',
              'comparison': 'independent SVD port must exactly equal pinned official-demo pixels',
              'versions': {'openvino': ov.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__},
              'model_sha256': {item['name']: item['sha256'] for item in record['files']
                               if item['name'].endswith(('.xml', '.bin'))}, 'cases': cases}
    (dest / 'parity-two.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'cases': cases, 'output': str(dest / 'parity-two.json')}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--frll-manifest', required=True, type=Path)
    parser.add_argument('--combined-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
