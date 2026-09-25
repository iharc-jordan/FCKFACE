"""Synthetic-only parity for the frozen OpenVINO 0095 IR and converted ONNX.

Run ``reference`` with the accepted OpenVINO environment, then ``check`` with
an isolated ONNX Runtime environment. No detector, photos, or gallery is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


MODEL = 'face-reidentification-retail-0095'
EXPECTED = {
    'xml': '6cf60c341452155e35c467510c6c50a96ade5b2bd8f88c5a90902e905d8a80c3',
    'bin': '21319b95e54181857f99e22dc32ec89770eca2969a1432cfa1594bffc94edd62',
}
CASES = ('range', 'flat', 'random')
RAW_MAX_LIMIT = 1e-3
UNIT_COS_LIMIT = .99999


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard_model(path: Path) -> dict[str, str]:
    hashes = {ext: sha(path / f'{MODEL}.{ext}') for ext in EXPECTED}
    if hashes != EXPECTED:
        raise ValueError(f'Official frozen model bytes changed: {path}')
    return hashes


def inputs() -> dict[str, np.ndarray]:
    shape = (1, 3, 128, 128)
    rng = np.random.default_rng(0)
    return {
        'range': np.linspace(0, 255, np.prod(shape), dtype=np.float32).reshape(shape),
        'flat': np.full(shape, 128, dtype=np.float32),
        'random': rng.integers(0, 256, shape, dtype=np.uint8).astype(np.float32),
    }


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def reference(args) -> None:
    import openvino as ov

    hashes = guard_model(args.original)
    guard_model(args.copied)
    core = ov.Core()
    model = core.read_model(str(args.original / f'{MODEL}.xml'))
    compiled = core.compile_model(model, 'CPU', {
        'INFERENCE_NUM_THREADS': 2, 'INFERENCE_PRECISION_HINT': 'f32',
    })
    precision = str(compiled.get_property('INFERENCE_PRECISION_HINT'))
    threads = int(compiled.get_property('INFERENCE_NUM_THREADS'))
    if precision != "<Type: 'float32'>" or threads != 2:
        raise ValueError(f'Native precision/threads differ: {precision}/{threads}')
    arrays = inputs()
    outputs = {}
    seconds = {}
    for name in CASES:
        started = perf_counter()
        output = np.asarray(compiled([arrays[name]])[compiled.output(0)], dtype=np.float32)
        seconds[name] = perf_counter() - started
        if output.shape != (1, 256, 1, 1) or not np.isfinite(output).all():
            raise ValueError(f'Invalid native output: {name}')
        outputs[f'{name}_input'] = arrays[name]
        outputs[f'{name}_reference'] = output
    np.savez(args.reference, **outputs)
    write(args.reference.with_suffix('.json'), {
        'purpose': 'synthetic_0095_openvino_fp32_reference',
        'official_model_sha256': hashes, 'reference_npz_sha256': sha(args.reference),
        'input_shape': [1, 3, 128, 128], 'layout': 'BGR_NCHW_0_to_255_float32',
        'output_shape': [1, 256, 1, 1], 'precision': precision, 'cpu_threads': threads,
        'openvino_version': ov.__version__, 'numpy_version': np.__version__,
        'cases_seconds': seconds,
    })


def dims(value) -> list[int | str]:
    return [d.dim_value if d.HasField('dim_value') else d.dim_param for d in value.type.tensor_type.shape.dim]


def check(args) -> None:
    import onnx
    import onnxruntime as ort

    hashes = guard_model(args.original)
    guard_model(args.copied)
    reference_meta = json.loads(args.reference.with_suffix('.json').read_text(encoding='utf-8'))
    if (sha(args.reference) != reference_meta['reference_npz_sha256'] or
            reference_meta['official_model_sha256'] != hashes or
            reference_meta['precision'] != "<Type: 'float32'>" or
            reference_meta['cpu_threads'] != 2):
        raise ValueError('Reference differs from accepted frozen native configuration')
    model = onnx.load(str(args.onnx), load_external_data=True)
    onnx.checker.check_model(model)
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1 or dims(graph.input[0]) != [1, 3, 128, 128] or \
            dims(graph.output[0]) != [1, 256, 1, 1]:
        raise ValueError('ONNX IO differs from original 0095 IR')
    domains = sorted({node.domain for node in graph.node})
    if any(domain not in ('', 'ai.onnx') for domain in domains):
        raise ValueError(f'Nonstandard ONNX operator domain: {domains}')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError('ONNX Runtime signature differs')
    output_arrays = {}
    rows = {}
    with np.load(args.reference, allow_pickle=False) as evidence:
        for name in CASES:
            actual_input = evidence[f'{name}_input']
            if not np.array_equal(actual_input, inputs()[name]):
                raise ValueError(f'Synthetic input generation changed: {name}')
            reference_raw = evidence[f'{name}_reference']
            started = perf_counter()
            actual_raw = np.asarray(session.run(None, {session.get_inputs()[0].name: actual_input})[0],
                                    np.float32)
            seconds = perf_counter() - started
            if actual_raw.shape != reference_raw.shape or not np.isfinite(actual_raw).all():
                raise ValueError(f'Invalid ONNX output: {name}')
            x, y = actual_raw.astype(np.float64).ravel(), reference_raw.astype(np.float64).ravel()
            cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
            max_abs = float(np.max(np.abs(actual_raw - reference_raw)))
            rows[name] = {'raw_max_abs': max_abs, 'unit_cosine': cosine, 'seconds': seconds,
                          'passed': max_abs <= RAW_MAX_LIMIT and cosine >= UNIT_COS_LIMIT}
            output_arrays[f'{name}_onnx'] = actual_raw
    np.savez(args.onnx.with_suffix('.outputs.npz'), **output_arrays)
    report = {
        'purpose': 'synthetic_0095_onnx_conversion_parity_only',
        'status': 'pass' if all(row['passed'] for row in rows.values()) else 'fail',
        'model_sha256': hashes, 'onnx_sha256': sha(args.onnx),
        'onnx_size_bytes': args.onnx.stat().st_size,
        'onnx_opset_imports': {x.domain: x.version for x in model.opset_import},
        'onnx_ops': {op: sum(n.op_type == op for n in graph.node)
                     for op in sorted({n.op_type for n in graph.node})},
        'onnx_domains': domains, 'input_name': graph.input[0].name,
        'output_name': graph.output[0].name,
        'limits': {'raw_max_abs': RAW_MAX_LIMIT, 'unit_cosine_min': UNIT_COS_LIMIT},
        'cases': rows, 'onnx_outputs_npz_sha256': sha(args.onnx.with_suffix('.outputs.npz')),
        'native_reference_npz_sha256': sha(args.reference),
        'onnx_version': onnx.__version__, 'onnxruntime_version': ort.__version__,
        'numpy_version': np.__version__,
    }
    write(args.report, report)
    if report['status'] != 'pass':
        raise ValueError('Converted ONNX failed the frozen synthetic parity limits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('reference', 'check'))
    for flag in ('original', 'copied', 'onnx', 'reference', 'report'):
        parser.add_argument('--' + flag, required=True, type=Path)
    args = parser.parse_args()
    (reference if args.phase == 'reference' else check)(args)
