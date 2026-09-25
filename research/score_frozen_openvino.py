"""Score frozen development JPEGs with the calibrated native OpenVINO 0095.

Candidate generation and appearance screening happen before this command.
The preflight-only path opens no source/gallery photos and loads no recognizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from time import perf_counter

import cv2
from PIL import Image

from calibrate_openvino import peak_memory_bytes
from fckface_lab.calibration import Calibration
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.evaluation import Reference
from fckface_lab.imaging import decode_image, export_jpeg
from openvino_0095_adapter import OpenVINO0095
from optimize_regions import CONDITIONS, VIEWS, evaluate_conditions
from score_openvino_h7 import (
    CAL_SHA, PIPELINE_SHA, ERRATUM_SHA, RGBModel, summary, utc, write_json,
)

HELPER_SHA = 'dbc62d1c1d92ebd3a25a097798b300a7e80b27e8c3b4e1ae9fd8727278c550f3'


def checked_inputs(args):
    repo = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(repo.parent):
        raise ValueError('Use a new output directory outside the repository')
    for path, expected in (
        (args.calibration, CAL_SHA), (args.pipeline, PIPELINE_SHA),
        (args.erratum, ERRATUM_SHA), (args.candidates, args.candidates_sha256),
        (repo / 'score_openvino_h7.py', HELPER_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f'Frozen artifact changed: {path.name}')
    cal = json.loads(args.calibration.read_text(encoding='utf-8'))
    pipeline = json.loads(args.pipeline.read_text(encoding='utf-8'))
    erratum = json.loads(args.erratum.read_text(encoding='utf-8'))
    if (cal['status'] != 'complete'
            or cal['purpose'] != 'development_openvino_0095_calibration_only'
            or cal['pipeline_frozen_sha256'] != PIPELINE_SHA
            or cal['calibration']['target_fmr'] != .001
            or erratum['affected_artifacts']['calibration.json_sha256'] != CAL_SHA
            or erratum['affected_artifacts']['pipeline-frozen.json_sha256'] != PIPELINE_SHA):
        raise ValueError('Wrong calibration or preprocessing erratum')
    for name, expected in pipeline['source_sha256'].items():
        path = args.calibrator_snapshot if name == 'calibrate_openvino.py' else repo / name
        if digest(path) != expected:
            raise ValueError(f'Calibrated source changed: {name}')
    if (digest(args.manifest) != pipeline['dataset_manifest_sha256']
            or digest(args.manifest) != cal['dataset']['manifest_sha256']):
        raise ValueError('Dataset manifest differs from calibration')
    for path, key in (
        (args.artifacts / 'artifact-record.json', 'artifact_record_sha256'),
        (args.artifacts / 'face-reidentification-retail-0095.xml', 'model_xml_sha256'),
        (args.artifacts / 'face-reidentification-retail-0095.bin', 'model_bin_sha256'),
        (args.yunet, 'yunet_sha256'),
    ):
        if digest(path) != pipeline[key]:
            raise ValueError(f'Calibrated model or detector changed: {path.name}')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest.get('split_version') != cal['dataset']['split_version']:
        raise ValueError('Dataset split version changed')
    groups = grouped_images(args.manifest, 'development')
    payload = json.loads(args.candidates.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1 or not payload.get('items'):
        raise ValueError('Expected a nonempty frozen candidate manifest')
    seen = set()
    items = []
    for item in payload['items']:
        if set(item) != {'identity', 'path', 'sha256', 'method', 'arm', 'source_view'}:
            raise ValueError('Wrong candidate schema')
        identity, arm = item['identity'], item['arm']
        if (manifest['identity_splits'].get(identity) != 'development'
                or identity not in groups or not set(VIEWS).issubset(groups[identity])
                or item['source_view'] != 'neutral_front'
                or not isinstance(item['method'], str) or not item['method']
                or not isinstance(arm, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', arm)
                or (identity, arm) in seen):
            raise ValueError('Wrong or duplicate development candidate')
        path = Path(item['path']).resolve(strict=True)
        if (path.is_relative_to(repo.parent)
                or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])
                or digest(path) != item['sha256']):
            raise ValueError('Frozen private candidate changed')
        with Image.open(path) as image:
            if image.format != 'JPEG' or image.getexif() or image.info.get('icc_profile'):
                raise ValueError('Expected a metadata-stripped JPEG')
        seen.add((identity, arm))
        items.append({**item, 'path': str(path)})
    items.sort(key=lambda item: (item['identity'], item['arm']))
    known = {(row['identity'], row['view']): row['sha256'] for row in manifest['images']}
    gallery = {f'{identity}:{view}': known[(identity, view)]
               for identity in sorted({item['identity'] for item in items}) for view in VIEWS}
    return cal, groups, items, gallery


def run(args):
    started = perf_counter()
    cal, groups, items, gallery_hashes = checked_inputs(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preflight = {
        'schema_version': 1, 'purpose': 'frozen_development_0095_before_gallery_or_model_load',
        'created_utc': utc(), 'scorer_sha256': digest(Path(__file__)),
        'helper_sha256': HELPER_SHA, 'candidate_manifest_sha256': digest(args.candidates),
        'items': items, 'gallery_expected_sha256': gallery_hashes,
        'frll_manifest_sha256': digest(args.manifest),
        'calibration_sha256': CAL_SHA, 'pipeline_sha256': PIPELINE_SHA,
        'erratum_sha256': ERRATUM_SHA,
        'calibrator_snapshot_sha256': digest(args.calibrator_snapshot),
        'conditions': list(CONDITIONS), 'gallery_views': list(VIEWS),
        'threshold': cal['calibration']['threshold'], 'cpu_threads': 2,
        'expected_precision': cal['pipeline']['actual_inference_precision_hint'],
        'preflight_only': args.preflight_only,
    }
    write_json(output / 'preflight.json', preflight)
    if args.preflight_only:
        print('Preflight passed; no recognizer or source/gallery image loaded.', flush=True)
        return
    results = []
    try:
        cv2.setNumThreads(2)
        native = OpenVINO0095(args.artifacts, args.yunet)
        if (native.actual_precision != preflight['expected_precision']
                or native.actual_threads != 2):
            raise ValueError('Calibrated inference configuration changed')
        model = RGBModel(native)
        threshold = Calibration(**cal['calibration'])
        for identity in sorted({item['identity'] for item in items}):
            identity_start = perf_counter()
            references, reference_evidence = [], {}
            source_box, source_image = None, None
            for view in VIEWS:
                path = groups[identity][view]
                if digest(path) != gallery_hashes[f'{identity}:{view}']:
                    raise ValueError(f'Gallery image changed: {identity}:{view}')
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                embedded = native.embed(bgr)
                reference_evidence[view] = {
                    'status': embedded.status, 'reason': embedded.reason,
                    'detection_count': embedded.detection_count, 'sha256': digest(path),
                }
                if embedded.status == 'valid':
                    references.append(Reference(f'{identity}:{view}', identity, embedded.feature))
                if view == 'neutral_front':
                    source_box = embedded.selected_box
                    source_image = decode_image(path)
                    if bgr is None or source_image.shape != bgr.shape:
                        raise ValueError('Clean source image dimensions differ')
            controls = {}
            if source_box is not None:
                controls = evaluate_conditions(export_jpeg(source_image), source_box, model,
                                               references, threshold, identity, CONDITIONS,
                                               save_dir=output / identity / 'control')
            eligible = (len(references) == len(VIEWS) and len(controls) == len(CONDITIONS)
                        and all(row['status'] == 'valid' and row['own_identity_matched'] is True
                                for row in controls.values()))
            arms = {}
            for item in (item for item in items if item['identity'] == identity):
                path = Path(item['path'])
                if digest(path) != item['sha256']:
                    raise ValueError('Frozen candidate changed after preflight')
                conditions = {}
                if eligible:
                    if decode_image(path).shape != source_image.shape:
                        raise ValueError('Candidate dimensions differ from source')
                    conditions = evaluate_conditions(path.read_bytes(), source_box, model,
                                                     references, threshold, identity, CONDITIONS,
                                                     save_dir=output / identity / item['arm'] / 'candidate')
                arms[item['arm']] = {'method': item['method'], 'selected_sha256': item['sha256'],
                                     'conditions': conditions, 'summary': summary(conditions)}
            results.append({'identity': identity, 'eligible': eligible,
                            'reference_evidence': reference_evidence, 'controls': controls,
                            'control_summary': summary(controls), 'arms': arms,
                            'seconds': perf_counter() - identity_start})
            write_json(output / 'progress.json', {'cases': results})
        write_json(output / 'results.json', {
            'schema_version': 1, 'status': 'complete',
            'purpose': 'frozen_openvino_0095_development_scoring',
            'preflight_sha256': digest(output / 'preflight.json'),
            'threshold': threshold.threshold, 'conditions': list(CONDITIONS),
            'gallery_views': list(VIEWS), 'cases': results,
            'elapsed_seconds': perf_counter() - started,
            'peak_process_working_set_bytes': peak_memory_bytes(),
            'limitations': 'Development only; detection failures inconclusive; no release or human-appearance claim.',
        })
    except Exception as exc:
        write_json(output / 'failure.json', {'at_utc': utc(), 'type': type(exc).__name__,
                                           'message': str(exc), 'completed_cases': len(results)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidates', 'manifest', 'calibration', 'pipeline', 'erratum',
                 'calibrator-snapshot', 'artifacts', 'yunet', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidates-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    run(parser.parse_args())
