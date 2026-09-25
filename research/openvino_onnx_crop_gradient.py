"""Two reviewed 0095 aligned-crop parity checks; no photo detection or gallery.

Run ``reference`` in the accepted OpenVINO environment, then ``gradient`` in
the existing read-only onnx2torch environment. All artifacts remain private.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np


FROZEN_REPORT_SHA = '7402e1083bb346aaf16171301c6c01edc1ea7fa5b258a735b2e22c9533a33c44'
XML_SHA = '6cf60c341452155e35c467510c6c50a96ade5b2bd8f88c5a90902e905d8a80c3'
BIN_SHA = '21319b95e54181857f99e22dc32ec89770eca2969a1432cfa1594bffc94edd62'
ONNX_SHA = '8f9880452be0bc0842ed580123f79b93e145079b27d21cc867bd3208f4b695b3'
IDS = ('frll-024', 'frll-036')
TARGET_SEED = 7095
EPS = .1
RAW_LIMIT = 1e-3
COS_LIMIT = .99999
REL_LIMIT = .1
ABS_LIMIT = 1e-6
NONTRIVIAL = 1e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: dict) -> None:
    if path.exists():
        raise ValueError(f'Refusing to overwrite: {path}')
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def frozen(args) -> dict[str, dict]:
    if sha(args.crops / 'parity-two.json') != FROZEN_REPORT_SHA:
        raise ValueError('Reviewed two-crop protocol changed')
    evidence = json.loads((args.crops / 'parity-two.json').read_text(encoding='utf-8'))
    if (evidence['purpose'] != 'official_demo_alignment_parity_two_prior_development_faces_only'
            or [row['identity'] for row in evidence['cases']] != list(IDS)
            or any(row['split'] != 'development' or not row['official_vs_port_exact']
                   for row in evidence['cases'])
            or evidence['model_sha256'] != {
                'face-reidentification-retail-0095.xml': XML_SHA,
                'face-reidentification-retail-0095.bin': BIN_SHA,
            }):
        raise ValueError('Wrong reviewed crop cases or model')
    return {row['identity']: row for row in evidence['cases']}


def reference(args) -> None:
    import cv2
    import openvino as ov

    records = frozen(args)
    if (sha(args.artifacts / 'face-reidentification-retail-0095.xml') != XML_SHA or
            sha(args.artifacts / 'face-reidentification-retail-0095.bin') != BIN_SHA):
        raise ValueError('Accepted native model changed')
    cv2.setNumThreads(2)
    core = ov.Core()
    model = core.read_model(str(args.artifacts / 'face-reidentification-retail-0095.xml'))
    compiled = core.compile_model(model, 'CPU', {
        'INFERENCE_NUM_THREADS': 2, 'INFERENCE_PRECISION_HINT': 'f32',
    })
    if (str(compiled.get_property('INFERENCE_PRECISION_HINT')) != "<Type: 'float32'>"
            or int(compiled.get_property('INFERENCE_NUM_THREADS')) != 2):
        raise ValueError('Native model no longer uses frozen FP32/two-thread configuration')
    arrays = {}
    cases = {}
    for identity in IDS:
        row = records[identity]
        crop = args.crops / f'{identity}-official-demo-crop.png'
        if sha(crop) != row['official_demo_crop_sha256']:
            raise ValueError(f'Reviewed crop PNG changed: {identity}')
        bgr = cv2.imread(str(crop), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (128, 128, 3) or bgr.dtype != np.uint8:
            raise ValueError(f'Invalid reviewed BGR crop: {identity}')
        nchw_u8 = np.ascontiguousarray(bgr.transpose(2, 0, 1)[None])
        if hashlib.sha256(nchw_u8.tobytes()).hexdigest() != row['aligned_input_nchw_sha256']:
            raise ValueError(f'Reviewed model-input pixels changed: {identity}')
        nchw = nchw_u8.astype(np.float32)
        started = perf_counter()
        raw = np.asarray(compiled([nchw])[compiled.output(0)], np.float32)
        seconds = perf_counter() - started
        if raw.shape != (1, 256, 1, 1) or not np.isfinite(raw).all():
            raise ValueError(f'Invalid native raw output: {identity}')
        raw_hash = hashlib.sha256(raw.reshape(-1).tobytes()).hexdigest()
        if raw_hash != row['embedding_sha256']:
            raise ValueError(f'Native FP32 raw differs from previously reviewed fixture: {identity}')
        arrays[f'{identity}_input_f32'] = nchw
        arrays[f'{identity}_native_raw'] = raw
        cases[identity] = {
            'crop_png_sha256': row['official_demo_crop_sha256'],
            'aligned_input_u8_sha256': row['aligned_input_nchw_sha256'],
            'input_f32_sha256': hashlib.sha256(nchw.tobytes()).hexdigest(),
            'native_raw_sha256': raw_hash, 'native_seconds': seconds,
        }
    if args.native.exists():
        raise ValueError('Refusing to overwrite native crop arrays')
    np.savez(args.native, **arrays)
    write(args.native.with_suffix('.json'), {
        'purpose': 'two_reviewed_0095_crop_native_fp32_raw_only',
        'frozen_crop_report_sha256': FROZEN_REPORT_SHA,
        'model_xml_sha256': XML_SHA, 'model_bin_sha256': BIN_SHA,
        'native_npz_sha256': sha(args.native), 'cases': cases,
        'openvino_version': ov.__version__, 'opencv_version': cv2.__version__,
        'numpy_version': np.__version__, 'cpu_threads': 2,
    })


def loss(raw: np.ndarray, target: np.ndarray) -> float:
    values = np.asarray(raw, np.float64).reshape(-1)
    return float(np.dot(values / np.linalg.norm(values), target))


def gradient(args) -> None:
    import onnx
    from onnx2torch import convert
    import onnxruntime as ort
    import torch

    records = frozen(args)
    native_meta = json.loads(args.native.with_suffix('.json').read_text(encoding='utf-8'))
    if (sha(args.native) != native_meta['native_npz_sha256'] or
            native_meta['frozen_crop_report_sha256'] != FROZEN_REPORT_SHA or
            set(native_meta['cases']) != set(IDS) or sha(args.onnx) != ONNX_SHA):
        raise ValueError('Crop native evidence or private ONNX changed')
    if args.report.exists() or args.gradients.exists():
        raise ValueError('Refusing to overwrite crop gradient results')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = perf_counter()
    model = convert(onnx.load(str(args.onnx))).eval()
    conversion_seconds = perf_counter() - started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 2
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    target = np.random.default_rng(TARGET_SEED).standard_normal(256).astype(np.float32)
    target /= np.linalg.norm(target)
    target_t = torch.from_numpy(target.astype(np.float64))
    rows = {}
    arrays = {'target_unit': target}
    with np.load(args.native, allow_pickle=False) as fixture:
        for identity in IDS:
            x = fixture[f'{identity}_input_f32']
            expected = fixture[f'{identity}_native_raw']
            if (x.shape != (1, 3, 128, 128) or expected.shape != (1, 256, 1, 1)
                    or hashlib.sha256(x.tobytes()).hexdigest()
                    != native_meta['cases'][identity]['input_f32_sha256']
                    or hashlib.sha256(expected.reshape(-1).tobytes()).hexdigest()
                    != records[identity]['embedding_sha256']):
                raise ValueError(f'Crop fixture changed: {identity}')
            input_t = torch.from_numpy(x.copy()).requires_grad_(True)
            t0 = perf_counter()
            raw_t = model(input_t)
            forward_seconds = perf_counter() - t0
            if not isinstance(raw_t, torch.Tensor) or tuple(raw_t.shape) != expected.shape:
                raise ValueError(f'Unexpected Torch output: {identity}')
            raw = raw_t.detach().numpy()
            if not np.isfinite(raw).all():
                raise ValueError(f'Nonfinite Torch output: {identity}')
            a, b = raw.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
            raw_max = float(np.max(np.abs(raw - expected)))
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if raw_max > RAW_LIMIT or cosine < COS_LIMIT:
                raise ValueError(f'Crop forward parity failed: {identity}')
            vector = raw_t.reshape(-1).double()
            objective = torch.dot(vector / torch.linalg.vector_norm(vector), target_t)
            t0 = perf_counter()
            grad = torch.autograd.grad(objective, input_t)[0].detach().numpy()
            gradient_seconds = perf_counter() - t0
            if not np.isfinite(grad).all() or np.linalg.norm(grad) <= 0:
                raise ValueError(f'Nonfinite/zero crop gradient: {identity}')
            admissible = (x.reshape(-1) >= EPS) & (x.reshape(-1) <= 255 - EPS)
            indices = np.argsort(np.where(admissible, -np.abs(grad.reshape(-1)), np.inf),
                                 kind='stable')[:3]
            if len(indices) != 3 or not np.all(admissible[indices]):
                raise ValueError(f'Too few finite-difference coordinates: {identity}')
            points = []
            for index in indices:
                plus, minus = x.copy(), x.copy()
                plus.flat[int(index)] += EPS
                minus.flat[int(index)] -= EPS
                denominator = float(plus.flat[int(index)] - minus.flat[int(index)])
                t0 = perf_counter()
                plus_raw = session.run(None, {session.get_inputs()[0].name: plus})[0]
                minus_raw = session.run(None, {session.get_inputs()[0].name: minus})[0]
                seconds = perf_counter() - t0
                fd = (loss(plus_raw, target) - loss(minus_raw, target)) / denominator
                analytic = float(grad.flat[int(index)])
                absolute = abs(fd - analytic)
                relative = absolute / max(abs(fd), abs(analytic), 1e-12)
                nontrivial = abs(analytic) >= NONTRIVIAL
                points.append({
                    'nchw': list(map(int, np.unravel_index(int(index), x.shape))),
                    'input_value': float(x.flat[int(index)]), 'analytic': analytic,
                    'finite_difference': fd, 'absolute_error': absolute,
                    'relative_error': relative, 'nontrivial': nontrivial,
                    'pass_relative': nontrivial and relative <= REL_LIMIT,
                    'pass_absolute_or_relative': absolute <= ABS_LIMIT or relative <= REL_LIMIT,
                    'finite_difference_seconds': seconds,
                })
            success = (sum(p['pass_relative'] for p in points) >= 2
                       and all(p['pass_absolute_or_relative'] for p in points))
            rows[identity] = {
                'crop_png_sha256': records[identity]['official_demo_crop_sha256'],
                'input_f32_sha256': native_meta['cases'][identity]['input_f32_sha256'],
                'native_raw_sha256': records[identity]['embedding_sha256'],
                'raw_max_abs_vs_openvino': raw_max, 'unit_cosine_vs_openvino': cosine,
                'loss': float(objective.detach()), 'gradient_l2': float(np.linalg.norm(grad)),
                'gradient_max_abs': float(np.max(np.abs(grad))),
                'forward_seconds': forward_seconds, 'gradient_seconds': gradient_seconds,
                'coordinates': points, 'gradient_pass': success,
            }
            arrays[f'{identity}_torch_raw'] = raw
            arrays[f'{identity}_input_gradient'] = grad
    np.savez(args.gradients, **arrays)
    report = {
        'purpose': 'two_reviewed_0095_crop_onnx2torch_forward_gradient_only',
        'status': 'pass' if all(x['gradient_pass'] for x in rows.values()) else 'fail',
        'frozen_crop_report_sha256': FROZEN_REPORT_SHA,
        'native_npz_sha256': sha(args.native), 'onnx_sha256_after': sha(args.onnx),
        'target_seed': TARGET_SEED, 'target_unit_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
        'limits': {'raw_max_abs': RAW_LIMIT, 'unit_cosine_min': COS_LIMIT,
                   'epsilon_input_pixel': EPS, 'relative_error_max': REL_LIMIT,
                   'absolute_error_max': ABS_LIMIT, 'nontrivial_gradient_min': NONTRIVIAL,
                   'nontrivial_relative_pass_count_min_per_case': 2},
        'onnx2torch_conversion_seconds': conversion_seconds, 'cases': rows,
        'gradient_npz_sha256': sha(args.gradients),
        'onnx2torch_version': version('onnx2torch'), 'torch_version': torch.__version__,
        'onnx_version': onnx.__version__, 'onnxruntime_version': ort.__version__,
        'numpy_version': np.__version__, 'cpu_threads': 2,
    }
    write(args.report, report)
    if report['status'] != 'pass':
        raise ValueError('Two-crop gradient finite-difference parity failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('reference', 'gradient'))
    for name in ('crops', 'artifacts', 'onnx', 'native', 'gradients', 'report'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    (reference if args.phase == 'reference' else gradient)(args)
