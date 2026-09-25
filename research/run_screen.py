"""Bounded, development-only screen of fixed FCKFACE appearance hypotheses.

Uses the frozen external selection.json, native YuNet/SFace, and an independently
calibrated SFace threshold. Never reads held-out identities or recognizers.
All experiment images, embeddings, and results are written outside this repo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import grouped_images
from fckface_lab.evaluation import Reference, evaluate_gallery
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.patterns import METHOD_DEFAULTS, actual_distortion, render, scale_to_rms
from fckface_lab.recognition import SFaceModel


VIEWS = ("neutral_front", "smiling_front", "neutral_left_3quarter", "neutral_right_3quarter")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960", "half_restore", "crop90", "blur")
METHODS = (
    ("uniform_dots", {"amplitude": 64.0, "period": 16.0}),
    ("global_multiscale", {"amplitude": 64.0}),
    ("face_relative_multiscale", {"amplitude": 64.0}),
    ("region_joint", {"amplitude": 64.0}),
    ("region_eyes", {"amplitude": 64.0}),
    ("region_cheeks", {"amplitude": 64.0}),
    ("region_nose", {"amplitude": 64.0}),
    ("region_lower", {"amplitude": 64.0}),
    ("graphic_halftone", {"step": 36.0}),
    ("ordinary_halftone", {"step": 36.0}),
    ("uniform_smooth", {"sigma": 5.0}),
)
H1_METHODS = METHODS[:3]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return sha(path.read_bytes())


def xyxy_to_xywh(box) -> tuple[float, float, float, float]:
    """Explicit adapter: recognition boxes are XYXY; renderers require XYWH."""
    left, top, right, bottom = map(float, box)
    return left, top, right - left, bottom - top


def save_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace experiment artifact: {path}")
    path.write_bytes(data)
    return sha(data)


def crop_png(image: np.ndarray, box) -> bytes:
    left, top, right, bottom = box
    width, height = image.shape[1], image.shape[0]
    mx, my = .14 * (right - left), .14 * (bottom - top)
    bounds = (max(0, int(left - mx)), max(0, int(top - my)),
              min(width, int(right + mx)), min(height, int(bottom + my)))
    from io import BytesIO
    stream = BytesIO()
    Image.fromarray(image[bounds[1]:bounds[3], bounds[0]:bounds[2]]).save(stream, format="PNG")
    return stream.getvalue()


def evaluation_record(result):
    return {"status": result.status, "reason": result.reason,
            "per_reference": dict(result.per_reference),
            "matching_references": list(result.matching_references),
            "own_source_score": result.own_source_score,
            "own_identity_matched": result.own_identity_matched}


def variant_records(jpeg: bytes, box, source_variants, model, refs, calibration,
                    identity, output_dir: Path, save_crops: bool):
    variants = make_variants(jpeg, box)
    records = {}
    for name in CONDITIONS:
        variant = variants[name]
        row = {"status": variant.status, "reason": variant.reason,
               "target_box_xyxy": variant.target_box}
        if variant.status != "valid" or variant.image is None or variant.jpeg is None:
            row["evaluation"] = {"status": "inconclusive", "reason": variant.reason}
            records[name] = row
            continue
        row["jpeg_sha256"] = save_bytes(output_dir / f"{name}.jpg", variant.jpeg)
        row["jpeg_bytes"] = len(variant.jpeg)
        if save_crops:
            row["face_crop_sha256"] = save_bytes(
                output_dir / f"{name}-face.png", crop_png(variant.image, variant.target_box))
        clean = source_variants.get(name)
        if clean is not None and clean.status == "valid" and clean.image.shape == variant.image.shape:
            row["post_jpeg_distortion"] = actual_distortion(
                clean.image, variant.image, xyxy_to_xywh(variant.target_box))
        started = time.perf_counter()
        probe = model.embed(variant.image, expected_box=variant.target_box)
        row["embed_seconds"] = round(time.perf_counter() - started, 4)
        row["detector_status"] = probe.status
        row["detector_reason"] = probe.reason
        row["detection_count"] = probe.detection_count
        row["selected_box_xyxy"] = probe.selected_box
        evaluation = evaluate_gallery(
            probe, refs, calibration, own_identity=identity,
            own_source_name=f"{identity}:neutral_front",
            expected_reference_names=[f"{identity}:{view}" for view in VIEWS])
        row["evaluation"] = evaluation_record(evaluation)
        records[name] = row
    return records


def load_setup(args, methods):
    selection_path = args.selection.resolve()
    manifest_path = args.manifest.resolve()
    calibration_path = args.calibration.resolve()
    yunet = args.yunet.resolve()
    sface = args.sface.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    groups = grouped_images(manifest_path, "development")
    identities = selection["identities"]
    if (identities != sorted(groups)[:8] or selection["reference_views"] != list(VIEWS)
            or selection["processing_conditions"] != list(CONDITIONS)
            or selection["candidate_methods"] != [method for method, _ in METHODS]
            or selection["seeds"] != [0, 1] or selection["target_face_rms"] != [4, 8]):
        raise ValueError("Frozen screen selection does not match source or script")
    if (args.identity_start, args.identity_count) not in ((0, 2), (2, 6)):
        raise ValueError("Only the frozen two-ID pilot or remaining six-ID confirmation is allowed")
    if (args.identity_start, args.identity_count) == (2, 6) and methods != H1_METHODS:
        raise ValueError("Remaining six identities are restricted to H1 confirmation")
    selected = identities[args.identity_start:args.identity_start + args.identity_count]
    for identity in selected:
        if set(VIEWS) - set(groups[identity]):
            raise ValueError(f"Missing prespecified views for {identity}")
    artifact = json.loads(calibration_path.read_text(encoding="utf-8"))
    if artifact.get("status") != "complete" or artifact.get("purpose") != "development_native_sface_calibration_only":
        raise ValueError("Wrong or incomplete native SFace calibration")
    calibration = Calibration(**artifact["calibration"])
    if calibration.valid_own == 0 or calibration.valid_impostor == 0:
        raise ValueError("Calibration lacks valid own/impostor pairs")
    input_hashes = {
        "dataset_manifest": digest(manifest_path), "selection": digest(selection_path),
        "calibration": digest(calibration_path), "yunet": digest(yunet), "sface": digest(sface),
        "runner": digest(Path(__file__)),
        "patterns": digest(Path(__file__).parent / "fckface_lab" / "patterns.py"),
        "recognition": digest(Path(__file__).parent / "fckface_lab" / "recognition.py"),
        "imaging": digest(Path(__file__).parent / "fckface_lab" / "imaging.py"),
        "evaluation": digest(Path(__file__).parent / "fckface_lab" / "evaluation.py"),
    }
    pipeline = artifact["pipeline"]
    expected_hashes = pipeline["source_sha256"]
    for key, source_name in (("recognition", "fckface_lab/recognition.py"),
                             ("imaging", "fckface_lab/imaging.py"),
                             ("evaluation", "fckface_lab/evaluation.py")):
        if input_hashes[key] != expected_hashes[source_name]:
            raise ValueError(f"Calibration preprocessing code changed: {source_name}")
    for name in ("yunet", "sface"):
        if input_hashes[name] != pipeline["model_artifacts"][name]["sha256"]:
            raise ValueError(f"Calibration {name} model hash differs")
    if input_hashes["dataset_manifest"] != artifact["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from calibration split")
    detector_input = pipeline["detector_input"]
    if detector_input["maximum_side_px"] != 640 or detector_input["score_threshold"] != .9:
        raise ValueError("Calibration detector settings differ from frozen pipeline")
    return selection, groups, selected, calibration, yunet, sface, input_hashes


def contact_sheet(rows: list[tuple[str, Path]], destination: Path):
    if not rows:
        return
    cell_w, cell_h, title_h = 180, 220, 30
    cols = min(6, len(rows))
    output = Image.new("RGB", (cols * cell_w, ((len(rows) + cols - 1) // cols) *
                               (cell_h + title_h)), "white")
    draw = ImageDraw.Draw(output)
    for index, (label, path) in enumerate(rows):
        x = index % cols * cell_w
        y = index // cols * (cell_h + title_h)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            output.paste(image, (x + (cell_w - image.width) // 2, y + title_h))
        draw.text((x + 4, y + 6), label[:29], fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination)


def run(args):
    started = time.perf_counter()
    source_root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if out == source_root or out.is_relative_to(source_root):
        raise ValueError("Experiment outputs must be outside the source repository")
    methods = H1_METHODS if args.family == "h1" else METHODS
    selection, groups, identities, calibration, yunet, sface, input_hashes = load_setup(args, methods)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Experiment destination already has artifacts: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    model = SFaceModel(yunet, sface)
    header = {"screen": "screen-v1", "scope": "development-only fixed geometry; no privacy or novelty claim",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
              "pillow": Image.__version__, "opencv_threads": cv2.getNumThreads(),
              "identities": identities, "views": VIEWS, "conditions": CONDITIONS,
              "seeds": selection["seeds"], "target_face_rms": selection["target_face_rms"],
              "method_parameters": {name: {**METHOD_DEFAULTS[name], **params}
                                    for name, params in methods},
              "family": args.family,
              "scope_change": ("H1-only confirmation on six prespecified IDs after root visual "
                               "review rejected H3 graphic shading and pilot H2 had zero nonmatches"
                               if args.identity_start == 2 else "original two-ID pilot"),
              "calibration": calibration.as_dict(), "input_sha256": input_hashes,
              "image_source_sha256": {identity: {view: digest(groups[identity][view])
                                         for view in VIEWS} for identity in identities}}
    (out / "header.json").write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(identities)} development identities, threshold {calibration.threshold:.8f}", flush=True)
    all_rows = []
    summary = []
    for identity in identities:
        identity_start = time.perf_counter()
        refs = []
        ref_results = {}
        for view in VIEWS:
            path = groups[identity][view]
            pixels = decode_image(path)
            embedded = model.embed(pixels)
            ref_results[view] = {"path": str(path), "sha256": digest(path),
                                 "status": embedded.status, "reason": embedded.reason,
                                 "box_xyxy": embedded.selected_box,
                                 "detection_count": embedded.detection_count}
            if embedded.status == "valid":
                refs.append(Reference(f"{identity}:{view}", identity, embedded.feature))
            if view == "neutral_front":
                source = pixels
                source_detection = embedded
        source_box = source_detection.selected_box
        if len(refs) != len(VIEWS) or source_box is None:
            summary.append({"identity": identity, "status": "inconclusive",
                            "reason": "missing_or_invalid_reference", "references": ref_results})
            print(identity, "inconclusive references", flush=True)
            continue
        source_variants = make_variants(export_jpeg(source), source_box)
        control_dir = out / identity / "control"
        controls = variant_records(source_variants["export"].jpeg, source_box,
                                   source_variants, model, refs, calibration,
                                   identity, control_dir, save_crops=True)
        control_ok = all(controls[c]["evaluation"]["status"] == "valid" and
                         controls[c]["evaluation"]["own_identity_matched"] is True
                         for c in CONDITIONS)
        summary.append({"identity": identity, "status": "eligible" if control_ok else "ineligible",
                        "reason": None if control_ok else "clean_control_invalid_or_unmatched",
                        "references": ref_results, "controls": controls})
        if not control_ok:
            print(identity, "ineligible clean control; no candidates", flush=True)
            continue
        box_xywh = xyxy_to_xywh(source_box)
        for seed in selection["seeds"]:
            for target_rms in selection["target_face_rms"]:
                sheet_rows = [("clean", control_dir / "export-face.png")]
                for method, params in methods:
                    case_start = time.perf_counter()
                    raw = render(source, box_xywh, source_detection.landmarks,
                                 method, params, seed)
                    render_seconds = time.perf_counter() - case_start
                    matched, raw_metrics = scale_to_rms(
                        source, raw, box_xywh, target_rms,
                        source_detection.landmarks)
                    jpeg = export_jpeg(matched)
                    case_dir = out / identity / f"seed{seed}-rms{target_rms}" / method
                    conditions = variant_records(jpeg, source_box, source_variants,
                                                  model, refs, calibration, identity,
                                                  case_dir, save_crops=True)
                    row = {"identity": identity, "seed": seed,
                           "target_face_rms": target_rms, "method": method,
                           "params": {**METHOD_DEFAULTS[method], **params},
                           "raw_distortion": raw_metrics,
                           "pre_match_distortion": actual_distortion(source, raw, box_xywh),
                           "render_seconds": round(render_seconds, 4),
                           "total_seconds": round(time.perf_counter() - case_start, 4),
                           "conditions": conditions}
                    with (out / "cases.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    all_rows.append(row)
                    sheet_rows.append((method, case_dir / "export-face.png"))
                contact_sheet(sheet_rows, out / "sheets" /
                              f"{identity}-seed{seed}-rms{target_rms}.png")
        print(identity, f"completed in {time.perf_counter() - identity_start:.1f}s", flush=True)
    aggregate = {"identities": summary, "candidate_count": len(all_rows),
                 "wall_seconds": round(time.perf_counter() - started, 3),
                 "finished_utc": datetime.now(timezone.utc).isoformat(),
                 "cases_jsonl_sha256": digest(out / "cases.jsonl") if (out / "cases.jsonl").exists() else None,
                 "contact_sheets": {str(p.relative_to(out)): digest(p)
                                    for p in sorted((out / "sheets").glob("*.png"))}}
    (out / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(all_rows)} candidates in {aggregate['wall_seconds']:.1f}s: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--yunet", type=Path, required=True)
    parser.add_argument("--sface", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity-start", type=int, required=True)
    parser.add_argument("--identity-count", type=int, required=True)
    parser.add_argument("--family", choices=("h1", "all"), required=True)
    arguments = parser.parse_args()
    run(arguments)
