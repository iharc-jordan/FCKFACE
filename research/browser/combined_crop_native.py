"""Freeze two permitted development crops and native two-model gradient references.

Run prepare, sface, ghost as separate processes in the existing read-only Python
environment. All photo-derived output must be outside Git. No held-out data.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IDENTITIES = ('frll-024', 'frll-036')
YUNET_SHA = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
SFACE_SHA = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'
GHOST_AUTHOR_SHA = 'e4182ca2470dac3eb79974f4d5d54f2abdf66d7e231c6f3df4059a90bded1271'
GHOST_CLONE_SHA = '55d808e3eae9371c1e16e3d38074e8cc3d329cb3e7b0a4fc17884b070b06e1f0'
SEEDS = {'sface': 20260128, 'ghost': 20260512}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, np.float64).reshape(-1), np.asarray(b, np.float64).reshape(-1)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def memory() -> dict:
    """Windows process peak working set and current system available RAM."""
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32), ('page_fault_count', ctypes.c_uint32),
                    ('peak_working_set', ctypes.c_size_t), ('working_set', ctypes.c_size_t),
                    ('quota_peak_paged_pool', ctypes.c_size_t), ('quota_paged_pool', ctypes.c_size_t),
                    ('quota_peak_nonpaged_pool', ctypes.c_size_t), ('quota_nonpaged_pool', ctypes.c_size_t),
                    ('pagefile_usage', ctypes.c_size_t), ('peak_pagefile_usage', ctypes.c_size_t)]

    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('memory_load', ctypes.c_uint32),
                    ('total_physical', ctypes.c_uint64), ('available_physical', ctypes.c_uint64),
                    ('total_pagefile', ctypes.c_uint64), ('available_pagefile', ctypes.c_uint64),
                    ('total_virtual', ctypes.c_uint64), ('available_virtual', ctypes.c_uint64),
                    ('available_extended_virtual', ctypes.c_uint64)]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    status = Status()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
    handle = kernel.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'process_peak_working_set_bytes': int(counters.peak_working_set),
            'process_working_set_bytes': int(counters.working_set),
            'system_available_ram_bytes': int(status.available_physical),
            'system_total_ram_bytes': int(status.total_physical)}


def read_crop(path: Path, expected: dict) -> np.ndarray:
    if sha(path) != expected['png_sha256']:
        raise ValueError(f'Crop PNG changed: {path}')
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    if rgb.shape != (112, 112, 3) or hashlib.sha256(rgb.tobytes()).hexdigest() != expected['rgb_sha256']:
        raise ValueError(f'Crop RGB pixels changed: {path}')
    return rgb


def prepare(args: argparse.Namespace) -> None:
    from fckface_lab.ghostface import author_align
    from fckface_lab.recognition import detector_frame, restore_face_coordinates

    out = args.output.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[2]) or out.exists():
        raise ValueError('Output must be a new directory outside Git')
    if sha(args.yunet) != YUNET_SHA:
        raise ValueError('Unexpected YuNet weights')
    frll = json.loads(args.frll_manifest.read_text(encoding='utf-8'))
    ort = json.loads(args.ort_manifest.read_text(encoding='utf-8'))
    yunet = json.loads(args.yunet_manifest.read_text(encoding='utf-8'))
    if [row['identity'] for row in ort['cases']] != list(IDENTITIES) or \
            [row['identity'] for row in yunet['cases']] != list(IDENTITIES):
        raise ValueError('Prior development fixture IDs changed')
    detector = cv2.FaceDetectorYN.create(str(args.yunet), '', (320, 320), score_threshold=.9)
    recognizer = cv2.FaceRecognizerSF.create(str(args.sface), '')
    if sha(args.sface) != SFACE_SHA:
        raise ValueError('Unexpected SFace weights')
    out.mkdir(parents=True)
    cases = []
    before = memory()
    for index, identity in enumerate(IDENTITIES):
        row = next((item for item in frll['images'] if item['identity'] == identity and
                    item['view'] == 'neutral_front' and item['split'] == 'development'), None)
        if row is None or row['sha256'] != ort['cases'][index]['source_sha256'] or \
                row['sha256'] != yunet['cases'][index]['source_sha256']:
            raise ValueError('Source identity/split/hash mismatch')
        source = (args.frll_manifest.parent / row['path']).resolve(strict=True)
        if not source.is_relative_to(args.frll_manifest.parent.resolve()) or sha(source) != row['sha256']:
            raise ValueError('Source escaped dataset or checksum mismatch')
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Cannot decode source')
        h, w = bgr.shape[:2]
        frame = detector_frame(bgr)
        fh, fw = frame.shape[:2]
        detector.setInputSize((fw, fh))
        _, detections = detector.detect(frame)
        if detections is None or len(detections) != 1:
            raise ValueError(f'Expected one face for {identity}')
        mapped = restore_face_coordinates(detections, (w, h), (fw, fh))[0]
        prior = np.asarray(yunet['cases'][index]['native_detection_mapped'], np.float32)
        if not np.array_equal(mapped, prior):
            raise ValueError(f'Native YuNet landmarks changed for {identity}')
        prior_portrait = args.ort_manifest.parent / f'yunet{index}-source.png'
        prior_bgr = cv2.imread(str(prior_portrait), cv2.IMREAD_COLOR)
        if prior_bgr is None or not np.array_equal(prior_bgr, bgr):
            raise ValueError('Prior portrait is not the same decoded source')
        portrait_path = out / f'{identity}-portrait.png'
        shutil.copyfile(prior_portrait, portrait_path)
        sface_bgr = recognizer.alignCrop(bgr, mapped)
        sface_rgb = cv2.cvtColor(sface_bgr, cv2.COLOR_BGR2RGB)
        ghost_rgb = author_align(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mapped[4:14].reshape(5, 2))
        crops = {}
        for model_name, rgb in (('sface', sface_rgb), ('ghost', ghost_rgb)):
            if rgb.shape != (112, 112, 3) or rgb.dtype != np.uint8:
                raise ValueError('Invalid crop')
            path = out / f'{identity}-{model_name}.png'
            Image.fromarray(rgb, 'RGB').save(path)
            crops[model_name] = {'name': path.name, 'png_sha256': sha(path),
                                 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
        cases.append({'identity': identity, 'split': 'development', 'view': 'neutral_front',
                      'source_jpeg_sha256': row['sha256'], 'portrait_png': portrait_path.name,
                      'portrait_png_sha256': sha(portrait_path),
                      'detector_size': [fw, fh], 'mapped_yunet_row_xywh_landmarks_score': mapped.tolist(),
                      'crops': crops})
    report = {'schema_version': 1, 'purpose': 'two_prior_development_identity_crops_only',
              'source_frll_manifest_sha256': sha(args.frll_manifest),
              'source_ort_manifest_sha256': sha(args.ort_manifest),
              'source_yunet_manifest_sha256': sha(args.yunet_manifest),
              'yunet_sha256': YUNET_SHA, 'sface_sha256': SFACE_SHA,
              'opencv_version': cv2.__version__, 'cases': cases,
              'memory_before': before, 'memory_after': memory()}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': [c['identity'] for c in cases], 'memory_after': report['memory_after']}))


def sface(args: argparse.Namespace, prepared: dict) -> None:
    import onnx
    import torch
    from onnx2torch import convert

    if sha(args.sface) != SFACE_SHA:
        raise ValueError('Unexpected SFace ONNX')
    torch.set_num_threads(2)
    recognizer = cv2.FaceRecognizerSF.create(str(args.sface), '')
    model = convert(onnx.load(str(args.sface))).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_np = unit(np.random.default_rng(SEEDS['sface']).standard_normal(128).astype(np.float32))
    target = torch.from_numpy(target_np)
    references = []
    for case in prepared['cases']:
        started = time.perf_counter()
        rgb = read_crop(args.output / case['crops']['sface']['name'], case['crops']['sface'])
        cv_raw = np.asarray(recognizer.feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), np.float32).reshape(-1)
        nchw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32))
        x = torch.from_numpy(nchw.copy()).requires_grad_(True)
        raw = model(x).reshape(-1)
        loss = 1 - torch.dot(torch.nn.functional.normalize(raw, dim=0), target)
        grad = torch.autograd.grad(loss, x)[0].detach().numpy().transpose(0, 2, 3, 1).copy()
        torch_raw = raw.detach().numpy()
        maximum = float(np.max(np.abs(torch_raw - cv_raw)))
        if maximum > 1e-3 or not all(np.isfinite(v).all() for v in (cv_raw, torch_raw, grad)) or not np.any(grad):
            raise ValueError(f'SFace native parity failed: {case["identity"]} max={maximum}')
        references.append({'identity': case['identity'], 'crop_png_sha256': case['crops']['sface']['png_sha256'],
                           'input_nhwc_f32_sha256': hashlib.sha256(rgb[None].astype('<f4').tobytes()).hexdigest(),
                           'opencv_raw': cv_raw.tolist(), 'onnx2torch_raw': torch_raw.tolist(),
                           'gradient_nhwc': grad.reshape(-1).tolist(), 'cosine_loss': float(loss.detach()),
                           'native_raw_max_difference': maximum, 'elapsed_seconds': time.perf_counter() - started})
    report = {'schema_version': 1, 'model': 'sface', 'purpose': 'real_development_crop_forward_cosine_gradient_parity',
              'crop_manifest_sha256': sha(args.output / 'manifest.json'), 'model_sha256': SFACE_SHA,
              'target_seed': SEEDS['sface'], 'target_unit': target_np.tolist(),
              'limits': {'raw_max': 1e-3, 'unit_cosine_min': .99999, 'gradient_cosine_min': .99},
              'torch_version': torch.__version__, 'opencv_version': cv2.__version__,
              'cases': references, 'memory_after': memory()}
    (args.output / 'sface-reference.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'model': 'sface', 'cases': [c['identity'] for c in references],
                      'max_native_raw_difference': max(c['native_raw_max_difference'] for c in references),
                      'memory_after': report['memory_after']}))


def ghost(args: argparse.Namespace, prepared: dict) -> None:
    import tensorflow as tf

    if sha(args.author) != GHOST_AUTHOR_SHA or sha(args.clone) != GHOST_CLONE_SHA:
        raise ValueError('Unexpected Ghost author or clone H5')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    author = tf.keras.models.load_model(str(args.author), compile=False)
    clone = tf.keras.models.load_model(str(args.clone), compile=False)
    target_np = unit(np.random.default_rng(SEEDS['ghost']).standard_normal(512).astype(np.float32))
    target = tf.constant(target_np[None], dtype=tf.float32)
    references = []
    for case in prepared['cases']:
        started = time.perf_counter()
        rgb = read_crop(args.output / case['crops']['ghost']['name'], case['crops']['ghost'])
        input_np = ((rgb[None].astype(np.float32) - 127.5) / 128).astype(np.float32)
        author_raw = tf.cast(author(input_np, training=False), tf.float32).numpy().reshape(-1)
        x = tf.Variable(input_np)
        with tf.GradientTape() as tape:
            clone_raw = tf.cast(clone(x, training=False), tf.float32)
            loss = 1 - tf.reduce_sum(tf.math.l2_normalize(clone_raw, axis=1) * target)
        grad = tape.gradient(loss, x)
        if grad is None:
            raise ValueError('Ghost clone input gradient absent')
        clone_flat = clone_raw.numpy().reshape(-1)
        grad_flat = grad.numpy().reshape(-1)
        author_unit = unit(author_raw)
        clone_unit = unit(clone_flat)
        similarity = cosine(author_unit, clone_unit)
        unit_max = float(np.max(np.abs(author_unit - clone_unit)))
        if similarity < .999 or unit_max > .02 or not all(np.isfinite(v).all() for v in
                (author_raw, clone_flat, grad_flat)) or not np.any(grad_flat):
            raise ValueError(f'Ghost native parity failed: {case["identity"]} cosine={similarity} unit_max={unit_max}')
        references.append({'identity': case['identity'], 'crop_png_sha256': case['crops']['ghost']['png_sha256'],
                           'input_nhwc_f32_sha256': hashlib.sha256(input_np.astype('<f4').tobytes()).hexdigest(),
                           'author_unit': author_unit.tolist(), 'clone_raw': clone_flat.tolist(),
                           'gradient_nhwc': grad_flat.tolist(), 'cosine_loss': float(loss.numpy()),
                           'native_author_clone_unit_cosine': similarity,
                           'native_author_clone_unit_max_difference': unit_max,
                           'elapsed_seconds': time.perf_counter() - started})
    report = {'schema_version': 1, 'model': 'ghost', 'purpose': 'real_development_crop_forward_cosine_gradient_parity',
              'crop_manifest_sha256': sha(args.output / 'manifest.json'),
              'author_h5_sha256': GHOST_AUTHOR_SHA, 'clone_h5_sha256': GHOST_CLONE_SHA,
              'target_seed': SEEDS['ghost'], 'target_unit': target_np.tolist(),
              'limits': {'f32_raw_max': 1e-3, 'author_unit_cosine_min': .999,
                         'author_unit_max_difference': .02, 'gradient_cosine_min': .99},
              'tensorflow_version': tf.__version__, 'cases': references, 'memory_after': memory()}
    (args.output / 'ghost-reference.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'model': 'ghost', 'cases': [c['identity'] for c in references],
                      'min_author_clone_unit_cosine': min(c['native_author_clone_unit_cosine'] for c in references),
                      'memory_after': report['memory_after']}))


def main() -> None:
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'sface', 'ghost'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frll-manifest', type=Path)
    parser.add_argument('--ort-manifest', type=Path)
    parser.add_argument('--yunet-manifest', type=Path)
    parser.add_argument('--yunet', type=Path)
    parser.add_argument('--sface', type=Path)
    parser.add_argument('--author', type=Path)
    parser.add_argument('--clone', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        for name in ('frll_manifest', 'ort_manifest', 'yunet_manifest', 'yunet', 'sface'):
            if getattr(args, name) is None:
                parser.error(f'--{name.replace("_", "-")} required for prepare')
        prepare(args)
    else:
        if not args.output.is_dir() or not (args.output / 'manifest.json').is_file():
            parser.error('Prepared external crop manifest required')
        prepared = json.loads((args.output / 'manifest.json').read_text(encoding='utf-8'))
        if [case['identity'] for case in prepared['cases']] != list(IDENTITIES):
            parser.error('Wrong prepared identities')
        destination = args.output / f'{args.stage}-reference.json'
        if destination.exists():
            parser.error(f'Refusing to overwrite {destination}')
        for name in (('sface',) if args.stage == 'sface' else ('author', 'clone')):
            if getattr(args, name) is None:
                parser.error(f'--{name} required for {args.stage}')
        (sface if args.stage == 'sface' else ghost)(args, prepared)


if __name__ == '__main__':
    main()
