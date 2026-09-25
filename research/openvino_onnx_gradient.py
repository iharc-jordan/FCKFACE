"""Synthetic-only native input-gradient check of converted OpenVINO 0095 ONNX.

Uses a verified private ONNX copy with existing read-only onnx2torch packages.
The frozen OpenVINO synthetic outputs are the independent forward reference.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
from onnx2torch import convert
import onnxruntime as ort
import torch


EXPECTED_ONNX = '8f9880452be0bc0842ed580123f79b93e145079b27d21cc867bd3208f4b695b3'
EXPECTED_REFERENCE = '94f95e7603b638b63ed65bdbc6984acfc46a846f67ce06416fbc4112816fa8f2'
CASES = ('range', 'flat', 'random')
RAW_MAX_LIMIT = 1e-3
UNIT_COS_LIMIT = .99999
EPS = .1
REL_LIMIT = .1
ABS_LIMIT = 1e-6
NONTRIVIAL_GRAD = 1e-6
TARGET_SEED = 7095


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(raw: np.ndarray, target: np.ndarray) -> float:
    vector = np.asarray(raw, np.float64).reshape(-1)
    return float(np.dot(vector / np.linalg.norm(vector), target))


def run(args) -> None:
    if args.report.exists() or args.gradients.exists():
        raise ValueError('Refusing to overwrite frozen gradient evidence')
    if (sha(args.original_onnx) != EXPECTED_ONNX or sha(args.copied_onnx) != EXPECTED_ONNX or
            sha(args.reference) != EXPECTED_REFERENCE):
        raise ValueError('Original, private ONNX copy, or native reference changed')
    native_meta = json.loads(args.reference.with_suffix('.json').read_text(encoding='utf-8'))
    if (native_meta['reference_npz_sha256'] != EXPECTED_REFERENCE or
            native_meta['precision'] != "<Type: 'float32'>" or native_meta['cpu_threads'] != 2 or
            native_meta['layout'] != 'BGR_NCHW_0_to_255_float32'):
        raise ValueError('Wrong native OpenVINO reference protocol')

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = perf_counter()
    torch_model = convert(onnx.load(str(args.copied_onnx))).eval()
    conversion_seconds = perf_counter() - started
    for parameter in torch_model.parameters():
        parameter.requires_grad_(False)
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 2
    ort_options.inter_op_num_threads = 2
    ort_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.copied_onnx), sess_options=ort_options,
                                   providers=['CPUExecutionProvider'])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError('Unexpected ONNX Runtime signature')

    target = np.random.default_rng(TARGET_SEED).standard_normal(256).astype(np.float32)
    target /= np.linalg.norm(target)
    target_torch = torch.from_numpy(target.astype(np.float64))
    records = {}
    gradient_arrays = {'target_unit': target}
    with np.load(args.reference, allow_pickle=False) as native:
        for name in CASES:
            image = np.asarray(native[f'{name}_input'], dtype=np.float32)
            expected = np.asarray(native[f'{name}_reference'], dtype=np.float32)
            if (image.shape != (1, 3, 128, 128) or expected.shape != (1, 256, 1, 1)
                    or not np.isfinite(image).all() or not np.isfinite(expected).all()
                    or np.min(image) < 0 or np.max(image) > 255):
                raise ValueError(f'Invalid synthetic case: {name}')
            input_tensor = torch.from_numpy(image.copy()).requires_grad_(True)
            t0 = perf_counter()
            raw_tensor = torch_model(input_tensor)
            forward_seconds = perf_counter() - t0
            if not isinstance(raw_tensor, torch.Tensor) or tuple(raw_tensor.shape) != tuple(expected.shape):
                raise ValueError(f'Unexpected Torch output shape: {name}')
            raw = raw_tensor.detach().numpy()
            if not np.isfinite(raw).all():
                raise ValueError(f'Nonfinite Torch output: {name}')
            raw_max = float(np.max(np.abs(raw - expected)))
            x, y = raw.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
            cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
            forward_pass = raw_max <= RAW_MAX_LIMIT and cosine >= UNIT_COS_LIMIT
            if not forward_pass:
                raise ValueError(f'Onnx2torch forward differs from frozen native output: {name}')
            unit_raw = raw_tensor.reshape(-1).double()
            loss = torch.dot(unit_raw / torch.linalg.vector_norm(unit_raw), target_torch)
            t0 = perf_counter()
            grad = torch.autograd.grad(loss, input_tensor)[0].detach().numpy()
            gradient_seconds = perf_counter() - t0
            if not np.isfinite(grad).all() or np.linalg.norm(grad) <= 0:
                raise ValueError(f'Nonfinite or zero input gradient: {name}')
            gradient_arrays[f'{name}_gradient'] = grad
            # Select the three largest admissible magnitudes. Central perturbations
            # remain in the predeclared 0..255 model input range.
            eligible = (image.reshape(-1) >= EPS) & (image.reshape(-1) <= 255 - EPS)
            ranked = np.argsort(np.where(eligible, -np.abs(grad.reshape(-1)), np.inf),
                                kind='stable')[:3]
            if len(ranked) != 3 or not all(eligible[ranked]):
                raise ValueError(f'Insufficient admissible gradient coordinates: {name}')
            coordinates = []
            for index in ranked:
                plus, minus = image.copy(), image.copy()
                plus.flat[int(index)] += EPS
                minus.flat[int(index)] -= EPS
                denominator = float(plus.flat[int(index)]) - float(minus.flat[int(index)])
                t0 = perf_counter()
                plus_raw = session.run(None, {session.get_inputs()[0].name: plus})[0]
                minus_raw = session.run(None, {session.get_inputs()[0].name: minus})[0]
                fd_seconds = perf_counter() - t0
                finite_difference = (scalar(plus_raw, target) - scalar(minus_raw, target)) / denominator
                analytic = float(grad.flat[int(index)])
                absolute = abs(analytic - finite_difference)
                relative = absolute / max(abs(analytic), abs(finite_difference), 1e-12)
                nontrivial = abs(analytic) >= NONTRIVIAL_GRAD
                coordinates.append({
                    'nchw': list(map(int, np.unravel_index(int(index), image.shape))),
                    'input_value': float(image.flat[int(index)]), 'analytic': analytic,
                    'finite_difference': finite_difference, 'absolute_error': absolute,
                    'relative_error': relative, 'nontrivial': nontrivial,
                    'pass_relative': nontrivial and relative <= REL_LIMIT,
                    'pass_absolute_or_relative': absolute <= ABS_LIMIT or relative <= REL_LIMIT,
                    'finite_difference_seconds': fd_seconds,
                })
            pass_gradient = (sum(row['pass_relative'] for row in coordinates) >= 2
                             and all(row['pass_absolute_or_relative'] for row in coordinates))
            records[name] = {
                'raw_max_abs_vs_openvino': raw_max, 'unit_cosine_vs_openvino': cosine,
                'forward_pass': forward_pass, 'loss': float(loss.detach()),
                'gradient_l2': float(np.linalg.norm(grad)),
                'gradient_max_abs': float(np.max(np.abs(grad))),
                'forward_seconds': forward_seconds, 'gradient_seconds': gradient_seconds,
                'coordinates': coordinates, 'gradient_pass': pass_gradient,
            }
    np.savez(args.gradients, **gradient_arrays)
    report = {
        'purpose': 'synthetic_openvino_0095_onnx2torch_input_gradient_only',
        'status': 'pass' if all(r['forward_pass'] and r['gradient_pass'] for r in records.values()) else 'fail',
        'original_onnx_sha256_after': sha(args.original_onnx),
        'copied_onnx_sha256_after': sha(args.copied_onnx),
        'native_reference_npz_sha256': sha(args.reference),
        'target_seed': TARGET_SEED, 'target_unit_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
        'limits': {'raw_max_abs': RAW_MAX_LIMIT, 'unit_cosine_min': UNIT_COS_LIMIT,
                   'epsilon_input_pixel': EPS, 'relative_error_max': REL_LIMIT,
                   'absolute_error_max': ABS_LIMIT, 'nontrivial_gradient_min': NONTRIVIAL_GRAD,
                   'nontrivial_relative_pass_count_min_per_case': 2},
        'onnx2torch_conversion_seconds': conversion_seconds, 'cases': records,
        'gradient_npz_sha256': sha(args.gradients),
        'onnx2torch_version': version('onnx2torch'), 'torch_version': torch.__version__,
        'onnx_version': onnx.__version__, 'onnxruntime_version': ort.__version__,
        'numpy_version': np.__version__, 'cpu_threads': 2,
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if report['status'] != 'pass':
        raise ValueError('Predeclared synthetic gradient parity failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-onnx', 'copied-onnx', 'reference', 'gradients', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
