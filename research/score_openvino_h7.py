"""Diagnostic OpenVINO 0095 scoring of only four frozen H7 development JPEGs.

Freezes protocol before opening clean gallery photos. No candidate optimization,
held-out model, release gate, or new identity. Private outputs stay outside Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from openvino_0095_adapter import OpenVINO0095
from optimize_regions import CONDITIONS, VIEWS, evaluate_conditions


IDS = ('frll-001', 'frll-003')
ARMS = ('fixed_condition_landmarks', 'refresh_edited_landmarks')
CAL_SHA = 'f33a0e6fa03a4853013eea9fdda9c6c2d0c008a2f6655fd52064bb0187d1275a'
PIPELINE_SHA = 'b3e54fef8527dc1e68e026f74668f9f043ffabedbdc685c649ca66c1e8693d05'
ERRATUM_SHA = 'ed21096fd501f9620d8c6bdc6860ff059b48ddcf3a9b8b60dd7cfeec776110ef'
H7_FROZEN_SHA = 'cff80bb0a7c8d5858dfa3447d1f4190f684912e052159f587831e5ebdfd10dac'
H7_AUDIT_SHA = 'aa2121645009086c53c363b6b8228d83be99d9e85f1f13514879666bcd2fa0f8'
H7_SOURCE_SHA = 'c03c4f437eb686e4cbc268a0f89df0b924df8db92242961c268d85d4d2183e8e'


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def summary(rows: dict) -> dict:
    valid = len(rows) == len(CONDITIONS) and all(r['status'] == 'valid' for r in rows.values())
    return {'valid_all': valid,
            'nonmatch_all': valid and all(r['own_identity_matched'] is False for r in rows.values()),
            'worst_cosine': max((r['maximum_cosine'] for r in rows.values()
                                 if r.get('maximum_cosine') is not None), default=None),
            'invalid_or_inconclusive': [name for name, r in rows.items() if r['status'] != 'valid']}


class RGBModel:
    """Shared condition evaluator supplies RGB; calibrated OpenVINO uses BGR."""
    def __init__(self, native: OpenVINO0095):
        self.native = native

    def embed(self, rgb, expected_box=None):
        return self.native.embed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), expected_box)


def checked_inputs(args):
    repo = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(repo.parent):
        raise ValueError('Output must be a new private directory outside Git')
    for path, expected in ((args.calibration, CAL_SHA), (args.pipeline, PIPELINE_SHA),
                           (args.erratum, ERRATUM_SHA), (args.h7 / 'frozen.json', H7_FROZEN_SHA),
                           (args.h7 / 'artifact-audit.json', H7_AUDIT_SHA),
                           (args.h7 / 'executed-source.py', H7_SOURCE_SHA)):
        if digest(path) != expected:
            raise ValueError(f'Frozen source or artifact changed: {path.name}')
    cal = json.loads(args.calibration.read_text(encoding='utf-8'))
    pipeline = json.loads(args.pipeline.read_text(encoding='utf-8'))
    erratum = json.loads(args.erratum.read_text(encoding='utf-8'))
    if (cal['status'] != 'complete' or cal['purpose'] != 'development_openvino_0095_calibration_only'
            or cal['pipeline_frozen_sha256'] != PIPELINE_SHA
            or cal['calibration']['target_fmr'] != .001
            or erratum['affected_artifacts']['calibration.json_sha256'] != CAL_SHA
            or erratum['affected_artifacts']['pipeline-frozen.json_sha256'] != PIPELINE_SHA):
        raise ValueError('Wrong calibration or erratum')
    for name, expected in pipeline['source_sha256'].items():
        source = (args.calibrator_snapshot if name == 'calibrate_openvino.py' else repo / name)
        if digest(source) != expected:
            raise ValueError(f'Executed calibration source changed: {name}')
    if digest(args.manifest) != pipeline['dataset_manifest_sha256'] or \
            digest(args.manifest) != cal['dataset']['manifest_sha256']:
        raise ValueError('FRLL manifest differs from calibration')
    if (digest(args.artifacts / 'artifact-record.json') != pipeline['artifact_record_sha256'] or
            digest(args.artifacts / 'face-reidentification-retail-0095.xml') != pipeline['model_xml_sha256'] or
            digest(args.artifacts / 'face-reidentification-retail-0095.bin') != pipeline['model_bin_sha256'] or
            digest(args.yunet) != pipeline['yunet_sha256']):
        raise ValueError('Official model or detector differs from calibration')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest.get('split_version') != cal['dataset']['split_version']:
        raise ValueError('FRLL split version changed')
    groups = grouped_images(args.manifest, 'development')
    frozen = json.loads((args.h7 / 'frozen.json').read_text(encoding='utf-8'))
    audit = json.loads((args.h7 / 'artifact-audit.json').read_text(encoding='utf-8'))
    if (not frozen['all_outputs_frozen_before_gallery'] or
            [case['identity'] for case in frozen['cases']] != list(IDS) or
            set(frozen['header']['arms']) != set(ARMS) or
            tuple(frozen['header']['final_conditions']) != CONDITIONS or
            audit['runner_sha256'] != H7_SOURCE_SHA or set(audit['identities']) != set(IDS)):
        raise ValueError('H7 frozen set differs')
    payload = json.loads(args.candidates.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1 or len(payload.get('items', [])) != 4:
        raise ValueError('Expected exactly four candidate-manifest items')
    expected_items = {(identity, arm) for identity in IDS for arm in ARMS}
    actual_items = set()
    items = []
    for row in payload['items']:
        if set(row) != {'identity', 'path', 'sha256', 'method', 'arm', 'source_view'}:
            raise ValueError('Wrong frozen candidate schema')
        identity, arm = row['identity'], row['arm']
        if (identity not in IDS or arm not in ARMS or row['method'] != 'alignment_dots_h7' or
                row['source_view'] != 'neutral_front' or
                manifest['identity_splits'].get(identity) != 'development' or
                not set(VIEWS).issubset(groups[identity]) or
                (identity, arm) in actual_items):
            raise ValueError('Wrong or duplicate H7 development item')
        selected = (args.h7 / identity / arm / 'selected.jpg').resolve(strict=True)
        if Path(row['path']).resolve(strict=True) != selected or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('Wrong candidate path or hash syntax')
        expected_sha = next(case['arms'][arm]['selected_jpeg_sha256'] for case in frozen['cases']
                            if case['identity'] == identity)
        if row['sha256'] != expected_sha or digest(selected) != expected_sha:
            raise ValueError('Frozen selected JPEG changed')
        with Image.open(selected) as image:
            if image.format != 'JPEG' or image.getexif() or image.info.get('icc_profile'):
                raise ValueError('Selected export metadata changed')
        actual_items.add((identity, arm))
        items.append(row)
    if actual_items != expected_items:
        raise ValueError('Missing H7 candidate')
    # Metadata only here; no clean source or gallery image bytes opened yet.
    expected_gallery = {}
    known = {(r['identity'], r['view']): r['sha256'] for r in manifest['images']}
    for identity in IDS:
        for view in VIEWS:
            expected = audit['identities'][identity]['gallery'][view]['sha256']
            if expected != known[(identity, view)]:
                raise ValueError('H7 and FRLL gallery metadata disagree')
            expected_gallery[f'{identity}:{view}'] = expected
    return cal, groups, frozen, audit, sorted(items, key=lambda r: (r['identity'], r['arm'])), expected_gallery


def run(args) -> None:
    started = perf_counter()
    cal, groups, frozen, audit, items, gallery_hashes = checked_inputs(args)
    cv2.setNumThreads(2)
    native = OpenVINO0095(args.artifacts, args.yunet)
    if native.actual_precision != cal['pipeline']['actual_inference_precision_hint'] or native.actual_threads != 2:
        raise ValueError('Calibrated OpenVINO inference configuration changed')
    model = RGBModel(native)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preflight = {'schema_version': 1, 'purpose': 'frozen_H7_openvino_0095_development_scoring_before_gallery_access',
                 'created_utc': utc(), 'scorer_sha256': digest(Path(__file__)),
                 'adapter_sha256': digest(Path(__file__).with_name('openvino_0095_adapter.py')),
                 'candidate_manifest_sha256': digest(args.candidates), 'items': items,
                 'gallery_expected_sha256': gallery_hashes,
                 'frll_manifest_sha256': digest(args.manifest),
                 'calibration_sha256': digest(args.calibration),
                 'pipeline_frozen_sha256': digest(args.pipeline),
                 'erratum_sha256': digest(args.erratum),
                 'h7_frozen_sha256': digest(args.h7 / 'frozen.json'),
                 'h7_audit_sha256': digest(args.h7 / 'artifact-audit.json'),
                 'h7_executed_source_sha256': digest(args.h7 / 'executed-source.py'),
                 'calibrator_executed_source_sha256': digest(args.calibrator_snapshot),
                 'model_xml_sha256': digest(args.artifacts / 'face-reidentification-retail-0095.xml'),
                 'model_bin_sha256': digest(args.artifacts / 'face-reidentification-retail-0095.bin'),
                 'yunet_sha256': digest(args.yunet),
                 'threshold': cal['calibration']['threshold'],
                 'actual_inference_precision_hint': native.actual_precision, 'cpu_threads': native.actual_threads,
                 'conditions': list(CONDITIONS), 'gallery_views': list(VIEWS)}
    write_json(output / 'preflight.json', preflight)
    # First clean gallery file access occurs only after preflight.json is saved.
    for identity in IDS:
        for view in VIEWS:
            if digest(groups[identity][view]) != gallery_hashes[f'{identity}:{view}']:
                raise ValueError(f'Gallery JPEG changed: {identity}:{view}')
    threshold = Calibration(**cal['calibration'])
    results = []
    try:
        for identity in IDS:
            identity_start = perf_counter()
            references = []
            reference_evidence = {}
            source_box = None
            source_image = None
            for view in VIEWS:
                path = groups[identity][view]
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                embedded = native.embed(bgr)
                reference_evidence[view] = {'status': embedded.status, 'reason': embedded.reason,
                                            'detection_count': embedded.detection_count,
                                            'sha256': digest(path)}
                if embedded.status == 'valid':
                    references.append(Reference(f'{identity}:{view}', identity, embedded.feature))
                if view == 'neutral_front':
                    source_image = decode_image(path)
                    source_box = embedded.selected_box
                    if bgr is None or source_image.shape != bgr.shape:
                        raise ValueError('Clean source image dimensions differ')
            controls = {}
            if source_box is not None:
                controls = evaluate_conditions(export_jpeg(source_image), source_box, model,
                                               references, threshold, identity, CONDITIONS,
                                               save_dir=output / identity / 'control')
            eligible = (len(references) == len(VIEWS) and len(controls) == len(CONDITIONS)
                        and all(r['status'] == 'valid' and r['own_identity_matched'] is True
                                for r in controls.values()))
            arms = {}
            for arm in ARMS:
                selected = args.h7 / identity / arm / 'selected.jpg'
                expected = next(r['sha256'] for r in items if r['identity'] == identity and r['arm'] == arm)
                if digest(selected) != expected:
                    raise ValueError('Selected H7 JPEG changed after preflight')
                candidate = {}
                if source_box is not None:
                    variants = make_variants(selected.read_bytes(), source_box)
                    for condition in CONDITIONS:
                        variant = variants[condition]
                        expected_condition = audit['identities'][identity]['arms'][arm][condition]['sha256']
                        if variant.jpeg is None or digest_bytes(variant.jpeg) != expected_condition:
                            raise ValueError(f'Exact H7 condition JPEG changed: {identity}/{arm}/{condition}')
                if eligible:
                    image = decode_image(selected)
                    if image.shape != source_image.shape:
                        raise ValueError('H7 candidate dimensions differ from clean source')
                    candidate = evaluate_conditions(selected.read_bytes(), source_box, model,
                                                    references, threshold, identity, CONDITIONS,
                                                    save_dir=output / identity / arm / 'candidate')
                arms[arm] = {'selected_sha256': expected, 'conditions': candidate,
                             'summary': summary(candidate)}
            results.append({'identity': identity, 'reference_evidence': reference_evidence,
                            'eligible': eligible, 'controls': controls,
                            'control_summary': summary(controls), 'arms': arms,
                            'seconds': round(perf_counter() - identity_start, 6)})
            write_json(output / 'progress.json', {'cases': results})
            print(f'OpenVINO H7 scored {identity}: eligible={eligible}', flush=True)
        report = {'schema_version': 1, 'status': 'complete',
                  'purpose': 'development_transfer_H7_openvino_0095_only',
                  'preflight_sha256': digest(output / 'preflight.json'),
                  'threshold': threshold.threshold, 'conditions': list(CONDITIONS),
                  'gallery_views': list(VIEWS), 'cases': results,
                  'elapsed_seconds': round(perf_counter() - started, 3),
                  'peak_process_working_set_bytes': peak_memory_bytes(),
                  'limitations': 'Diagnostic third development recognizer only; not a held-out release gate.'}
        write_json(output / 'results.json', report)
    except Exception as exc:
        write_json(output / 'failure.json', {'at_utc': utc(), 'type': type(exc).__name__,
                                             'message': str(exc), 'completed_cases': len(results)})
        raise


def digest_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidates', 'manifest', 'calibration', 'pipeline', 'erratum',
                 'calibrator-snapshot', 'artifacts', 'yunet', 'h7', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
