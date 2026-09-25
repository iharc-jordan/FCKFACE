"""Calibrate official Intel OMZ 0095 on 20 FRLL calibration identities only.

Private embeddings/scores stay outside Git. This is development calibration,
not edited-photo evidence or a public release gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np
import openvino as ov

from calibrate_sface import read_calibration_records
from fckface_lab.calibration import CalibrationPair, calibrate
from fckface_lab.datasets import VIEWS, digest
from fckface_lab.evaluation import cosine
from openvino_0095_adapter import OpenVINO0095, YUNET_SHA256


SFACE_CALIBRATION_HELPER_SHA_FILES = (
    'calibrate_openvino.py', 'openvino_0095_adapter.py', 'openvino_development.py',
    'calibrate_sface.py', 'fckface_lab/calibration.py', 'fckface_lab/datasets.py',
    'fckface_lab/evaluation.py', 'fckface_lab/recognition.py',
)
PREPARATION_RUNNER_SHA = 'b71c62584374ce6991cb1573920653e1a5163feb53ee0a000e1197eadf5286f1'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def log(path: Path, row: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()


def peak_memory_bytes() -> int:
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32), ('page_fault_count', ctypes.c_uint32),
                    ('peak_working_set', ctypes.c_size_t), ('working_set', ctypes.c_size_t),
                    ('quota_peak_paged_pool', ctypes.c_size_t), ('quota_paged_pool', ctypes.c_size_t),
                    ('quota_peak_nonpaged_pool', ctypes.c_size_t), ('quota_nonpaged_pool', ctypes.c_size_t),
                    ('pagefile_usage', ctypes.c_size_t), ('peak_pagefile_usage', ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.peak_working_set)


def source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    hashes = {name: digest(base / name) for name in SFACE_CALIBRATION_HELPER_SHA_FILES}
    if hashes['openvino_development.py'] != PREPARATION_RUNNER_SHA:
        raise ValueError('Reviewed preparation pipeline source changed')
    return hashes


def fixture_check(model: OpenVINO0095, manifest_path: Path, combined_path: Path,
                  preparation_path: Path) -> list[dict]:
    frll = json.loads(manifest_path.read_text(encoding='utf-8'))
    combined = json.loads(combined_path.read_text(encoding='utf-8'))
    prepared = json.loads(preparation_path.read_text(encoding='utf-8'))
    if [case['identity'] for case in combined['cases']] != ['frll-024', 'frll-036'] or \
            [case['identity'] for case in prepared['cases']] != ['frll-024', 'frll-036']:
        raise ValueError('Wrong two prior development fixtures')
    checks = []
    for prior, frozen in zip(combined['cases'], prepared['cases']):
        entry = next((row for row in frll['images'] if row['identity'] == prior['identity'] and
                      row['view'] == 'neutral_front' and row['split'] == 'development'), None)
        if entry is None or entry['sha256'] != prior['source_jpeg_sha256']:
            raise ValueError('Fixture split/hash mismatch')
        path = (manifest_path.parent / entry['path']).resolve(strict=True)
        if not path.is_relative_to(manifest_path.parent.resolve()) or digest(path) != entry['sha256']:
            raise ValueError('Fixture source changed')
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        result = model.embed(bgr)
        if (result.status != 'valid' or result.detection_count != 1 or
                not np.array_equal(result.selected_row,
                                   np.asarray(prior['mapped_yunet_row_xywh_landmarks_score'], np.float32)) or
                result.input_sha256 != frozen['aligned_input_nchw_sha256'] or
                result.raw_sha256 != frozen['embedding_sha256']):
            raise ValueError(f'FP32 runtime or fresh detection changed reviewed fixture {prior["identity"]}')
        checks.append({'identity': prior['identity'], 'source_sha256': entry['sha256'],
                       'input_sha256': result.input_sha256, 'raw_embedding_sha256': result.raw_sha256,
                       'detector_row_equal': True, 'status': 'pass'})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--yunet', required=True, type=Path)
    parser.add_argument('--combined-manifest', required=True, type=Path)
    parser.add_argument('--preparation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started_clock = perf_counter()
    manifest = args.manifest.resolve(strict=True)
    artifacts = args.artifacts.resolve(strict=True)
    yunet = args.yunet.resolve(strict=True)
    combined = args.combined_manifest.resolve(strict=True)
    preparation = args.preparation.resolve(strict=True)
    output = args.output.resolve()
    source_root = Path(__file__).resolve().parents[1]
    if output.exists() or output == source_root or output.is_relative_to(source_root):
        raise ValueError('Choose a new private output directory outside Git')
    rows, data = read_calibration_records(manifest)
    if len(rows) != 200 or len({row['identity'] for row in rows}) != 20 or \
            any(row['view'] not in VIEWS or row['split'] != 'calibration' for row in rows):
        raise ValueError('Calibration split preflight failed')
    for row in rows:
        if digest(Path(row['absolute_path'])) != row['sha256']:
            raise ValueError(f'Calibration source hash mismatch: {row["identity"]}:{row["view"]}')
    if digest(yunet) != YUNET_SHA256:
        raise ValueError('YuNet hash mismatch')
    frozen_sources = source_hashes()
    cv2.setNumThreads(2)
    model = OpenVINO0095(artifacts, yunet)
    fixtures = fixture_check(model, manifest, combined, preparation)
    frozen = {
        'schema_version': 1, 'purpose': 'frozen_openvino_0095_development_calibration_preflight',
        'started_utc': now(), 'dataset_manifest_sha256': digest(manifest),
        'split_version': data['split_version'], 'identity_ids': sorted({r['identity'] for r in rows}),
        'calibration_images_preflight_hashed': len(rows), 'views': list(VIEWS),
        'artifact_record_sha256': digest(artifacts / 'artifact-record.json'),
        'model_xml_sha256': digest(artifacts / 'face-reidentification-retail-0095.xml'),
        'model_bin_sha256': digest(artifacts / 'face-reidentification-retail-0095.bin'),
        'yunet_sha256': digest(yunet), 'combined_manifest_sha256': digest(combined),
        'preparation_result_sha256': digest(preparation), 'source_sha256': frozen_sources,
        'pipeline': {
            'detector': 'YuNet max-side640 score.9, mapped XYWH+five landmarks; exactly one face',
            'detector_substitution': 'YuNet replaces Intel OMZ demo detector and landmark regressor',
            'image_decode': 'OpenCV imread BGR uint8, unchanged 0..255 JPEG decode',
            'roi': 'integer-truncated YuNet XYWH endpoints clipped to image',
            'alignment': 'pinned Intel FaceIdentifier SVD warpAffine WARP_INVERSE_MAP nearest, then INTER_LINEAR resize',
            'input': 'cv2 INTER_LINEAR resize 128x128 BGR NCHW float32 0..255, no external mean/scale',
            'embedding': 'official FP32 IR 256D, finite and L2 normalized',
            'comparison': 'cosine >= frozen model-specific threshold is match',
            'cpu_threads': model.actual_threads, 'cv2_threads': cv2.getNumThreads(),
            'requested_inference_precision_hint': 'f32',
            'actual_inference_precision_hint': model.actual_precision,
        },
        'fixture_parity': fixtures,
        'versions': {'python': platform.python_version(), 'opencv': cv2.__version__,
                     'numpy': np.__version__, 'openvino': ov.__version__},
    }
    output.mkdir(parents=True)
    write_json(output / 'pipeline-frozen.json', frozen)
    run_log = output / 'run-log.jsonl'
    log(run_log, {'event': 'start', 'at_utc': now(), 'image_count': len(rows),
                  'pipeline_frozen_sha256': digest(output / 'pipeline-frozen.json')})
    try:
        features: list[np.ndarray | None] = []
        images: list[dict] = []
        for index, row in enumerate(rows):
            t0 = perf_counter()
            reason = None
            result = None
            try:
                bgr = cv2.imread(row['absolute_path'], cv2.IMREAD_COLOR)
                result = model.embed(bgr)
                status, reason = result.status, result.reason
            except (OSError, ValueError, cv2.error, RuntimeError) as exc:
                status, reason = 'inconclusive', f'{type(exc).__name__}:{exc}'
            feature = result.feature if result is not None and status == 'valid' else None
            if feature is not None and feature.shape != (256,):
                raise ValueError('Invalid embedding dimension')
            features.append(feature)
            image = {'index': index, 'identity': row['identity'], 'view': row['view'],
                     'image_id': f'{row["identity"]}:{row["view"]}', 'sha256': row['sha256'],
                     'status': status, 'reason': reason,
                     'detection_count': result.detection_count if result else 0,
                     'seconds': round(perf_counter() - t0, 6)}
            images.append(image)
            log(run_log, {'event': 'image', 'at_utc': now(), **image})
            if (index + 1) % 20 == 0:
                print(f'Embedded {index + 1}/200; valid {sum(f is not None for f in features)}', flush=True)
        if not any(f is not None for f in features):
            raise ValueError('No valid calibration embeddings')
        matrix = np.zeros((len(features), 256), dtype=np.float32)
        for i, feature in enumerate(features):
            if feature is not None:
                matrix[i] = feature
        np.savez_compressed(output / 'embeddings.npz', features=matrix,
                            image_ids=np.asarray([r['image_id'] for r in images]),
                            identities=np.asarray([r['identity'] for r in rows]),
                            views=np.asarray([r['view'] for r in rows]),
                            valid=np.asarray([f is not None for f in features]),
                            status=np.asarray([r['status'] for r in images]))
        strata: dict[str, Counter] = defaultdict(Counter)
        pair_evidence: list[tuple[str, str, CalibrationPair]] = []
        with (output / 'scores.jsonl').open('w', encoding='utf-8') as stream:
            for i, j in combinations(range(len(rows)), 2):
                left, right = rows[i], rows[j]
                same = left['identity'] == right['identity']
                same_view = left['view'] == right['view']
                stratum = ('genuine' if same else 'impostor') + ('_same_view' if same_view else '_different_view')
                score = cosine(features[i], features[j]) if features[i] is not None and features[j] is not None else None
                strata[stratum]['total'] += 1
                strata[stratum]['valid' if score is not None else 'invalid'] += 1
                pair = CalibrationPair(same, score)
                pair_evidence.append((left['identity'], right['identity'], pair))
                stream.write(json.dumps({'left_index': i, 'right_index': j, 'same_identity': same,
                                         'same_view': same_view, 'stratum': stratum, 'score': score,
                                         'reason': None if score is not None else 'one_or_both_embeddings_invalid'},
                                        allow_nan=False) + '\n')
        calibrated = calibrate([pair for _, _, pair in pair_evidence], target_fmr=.001)
        loo = [{'removed_identity': identity,
                'threshold': calibrate([pair for left, right, pair in pair_evidence
                                        if left != identity and right != identity], target_fmr=.001).threshold}
               for identity in sorted({r['identity'] for r in rows})]
        report = {'schema_version': 1, 'purpose': 'development_openvino_0095_calibration_only',
                  'status': 'complete', 'completed_utc': now(),
                  'elapsed_seconds': round(perf_counter() - started_clock, 3),
                  'peak_process_working_set_bytes': peak_memory_bytes(),
                  'pipeline_frozen_sha256': digest(output / 'pipeline-frozen.json'),
                  'dataset': {'name': 'frll', 'split': 'calibration', 'split_version': data['split_version'],
                              'identity_count': 20, 'image_count': 200, 'manifest_sha256': digest(manifest),
                              'identity_ids': sorted({r['identity'] for r in rows}), 'views': list(VIEWS)},
                  'pipeline': frozen['pipeline'], 'calibration': calibrated.as_dict(),
                  'pair_strata': {key: dict(value) for key, value in sorted(strata.items())},
                  'invalid_reasons': dict(Counter(r['reason'] for r in images if r['status'] != 'valid')),
                  'leave_one_identity_out': {'thresholds': loo,
                                             'minimum': min(r['threshold'] for r in loo),
                                             'maximum': max(r['threshold'] for r in loo),
                                             'caveat': 'Only 20 identities; pairs are dependent. Empirical FMR and range are not population confidence bounds.'},
                  'images': images,
                  'artifacts': {name: digest(output / name) for name in ('embeddings.npz', 'scores.jsonl')},
                  'limitations': 'No edited photos, cross-model transfer, held-out validation or browser conversion tested.'}
        write_json(output / 'calibration.json', report)
        log(run_log, {'event': 'complete', 'at_utc': now(), 'threshold': calibrated.threshold})
        print(json.dumps({'status': 'complete', 'threshold': calibrated.threshold,
                          'calibration': calibrated.as_dict(), 'elapsed_seconds': report['elapsed_seconds'],
                          'peak_process_working_set_bytes': report['peak_process_working_set_bytes'],
                          'invalid_reasons': report['invalid_reasons'], 'output': str(output)}))
        return 0
    except BaseException as exc:
        log(run_log, {'event': 'error', 'at_utc': now(), 'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
