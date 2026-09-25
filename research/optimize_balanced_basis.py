"""H14 source-only three-arm dot mechanism test; stops after six JPEGs freeze.

A: SFace/Ghost global maximum. B: mean of their per-model condition maxima.
C: SFace/compact OpenVINO 0095 with the same balanced loss. The converted
0095 Torch model is permitted only after its private native-gradient diagnostic
and this run's exact clean-input geometry parity both pass. No gallery, ArcFace,
reserved held-out recognizer, or checkpoint score is accessed by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np

from fckface_lab.datasets import digest, grouped_images
from h14_0095_alignment import (CAL_SHA, ERRATUM_SHA, IDS, OBJECTIVE, PIPELINE_SHA,
                                YUNET_SHA, exact_input, geometry, residual_bgr128,
                                synthetic_pullback_check)


ARMS = ("sg_global_max", "sg_balanced", "s0095_balanced")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960",
              "half_restore", "crop90", "blur")
STEPS = 18
ADAM_LR = .15
ONNX_SHA = "8f9880452be0bc0842ed580123f79b93e145079b27d21cc867bd3208f4b695b3"
ONNX_BYTES = 4461366
GRADIENT_DIAGNOSTIC_SHA = "af479492c8eb83dfb1cd52aac1937d561a2a0cf1b72f303d6113a261cef1bd0b"
CROP_DIAGNOSTIC_SHA = "2002e313dfa1da324e039d9878faf53bf3fda41d2bd631431d4121e3a692ada8"
ARTIFACT_RECORD_SHA = "639a4e8714fdfc04f45de018d58e13478d2d1c21608e8d668402468cdeadb4fa"
MODEL_XML_SHA = "6cf60c341452155e35c467510c6c50a96ade5b2bd8f88c5a90902e905d8a80c3"
MODEL_BIN_SHA = "21319b95e54181857f99e22dc32ec89770eca2969a1432cfa1594bffc94edd62"
FRLL_SHA = "2509c48bd7852251f7a3778cc7e1e320e17003d1bb56cca6dbf84998b4931600"
SFACE_CAL_SHA = "03f93365189dae853e306bdbed7a96536926628dfa1302d363d2779db98c8f04"
GHOST_CAL_SHA = "7c0374b294bf20884bc3eba29b5d7a23b9629d778d382c90177ef4d633fbe90f"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checked_inputs(args):
    """Metadata/hashes only; never opens any other view or loads a recognizer."""
    from fckface_lab.calibration import Calibration

    repo = Path(__file__).resolve().parents[1]
    if args.out.exists() or args.out.resolve().is_relative_to(repo):
        raise ValueError("Output must be new and outside Git")
    expected_files = ((args.manifest, FRLL_SHA),
                      (args.sface_calibration, SFACE_CAL_SHA),
                      (args.ghostface_calibration, GHOST_CAL_SHA),
                      (args.openvino_calibration, CAL_SHA), (args.openvino_pipeline, PIPELINE_SHA),
                      (args.openvino_erratum, ERRATUM_SHA),
                      (args.openvino_onnx, ONNX_SHA),
                      (args.openvino_gradient_diagnostic, GRADIENT_DIAGNOSTIC_SHA),
                      (args.openvino_crop_diagnostic, CROP_DIAGNOSTIC_SHA),
                      (args.openvino_artifacts / "artifact-record.json", ARTIFACT_RECORD_SHA),
                      (args.yunet, YUNET_SHA))
    for path, expected in expected_files:
        if digest(path) != expected:
            raise ValueError(f"Frozen dependency changed: {path}")
    if args.openvino_onnx.stat().st_size != ONNX_BYTES:
        raise ValueError("Frozen 0095 ONNX length changed")
    diagnostic = json.loads(args.openvino_gradient_diagnostic.read_text(encoding="utf-8"))
    if (diagnostic.get("status") != "pass" or
            diagnostic.get("original_onnx_sha256_after") != ONNX_SHA or
            diagnostic.get("copied_onnx_sha256_after") != ONNX_SHA or
            not all(diagnostic["cases"][name][key] is True
                    for name in ("range", "flat", "random")
                    for key in ("forward_pass", "gradient_pass"))):
        raise ValueError("Frozen 0095 Torch input-gradient diagnostic did not pass")
    crop_diagnostic = json.loads(args.openvino_crop_diagnostic.read_text(encoding="utf-8"))
    if (crop_diagnostic.get("status") != "pass" or
            crop_diagnostic.get("onnx_sha256_after") != ONNX_SHA or
            not all(crop_diagnostic["cases"][name]["gradient_pass"] is True
                    for name in ("frll-024", "frll-036"))):
        raise ValueError("Frozen two-real-crop 0095 Torch parity did not pass")
    records = {
        "SFace": json.loads(args.sface_calibration.read_text(encoding="utf-8")),
        "GhostFaceNet": json.loads(args.ghostface_calibration.read_text(encoding="utf-8")),
        "OpenVINO0095": json.loads(args.openvino_calibration.read_text(encoding="utf-8")),
    }
    for name, models in (("SFace", ("yunet", "sface")),
                         ("GhostFaceNet", ("yunet", "ghostface"))):
        cal = records[name]
        if (cal.get("status") != "complete" or
                digest(args.manifest) != cal["dataset"]["manifest_sha256"]):
            raise ValueError(f"Wrong {name} calibration/FRLL manifest")
        for model in models:
            artifact = cal["pipeline"]["model_artifacts"][model]
            path = getattr(args, model)
            if digest(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
                raise ValueError(f"Calibrated model bytes changed: {name}/{model}")
        for name_source, sha in cal["pipeline"]["source_sha256"].items():
            if name_source.startswith("fckface_lab/") and digest(repo / "research" / name_source) != sha:
                raise ValueError(f"Calibrated shared source changed: {name_source}")
    if (records["OpenVINO0095"].get("status") != "complete" or
            digest(args.manifest) != records["OpenVINO0095"]["dataset"]["manifest_sha256"] or
            records["OpenVINO0095"]["calibration"]["threshold"] != .43059808165428554):
        raise ValueError("Wrong 0095 calibration/FRLL manifest")
    if (digest(args.openvino_artifacts / "face-reidentification-retail-0095.xml") != MODEL_XML_SHA or
            digest(args.openvino_artifacts / "face-reidentification-retail-0095.bin") != MODEL_BIN_SHA):
        raise ValueError("Official 0095 artifact bytes changed")
    prep = json.loads(args.openvino_prep.read_text(encoding="utf-8"))
    prep_protocol = args.openvino_prep.with_name("protocol.json")
    protocol = json.loads(prep_protocol.read_text(encoding="utf-8"))
    if (prep.get("status") != "complete" or prep.get("purpose") != "H14_source_only_0095" or
            prep["protocol_sha256"] != digest(prep_protocol) or
            protocol["helper_sha256"] != digest(Path(__file__).with_name("h14_0095_alignment.py")) or
            protocol["manifest_sha256"] != digest(args.manifest) or
            protocol["calibration_sha256"] != CAL_SHA or
            prep["model_xml_sha256"] != MODEL_XML_SHA or
            prep["model_bin_sha256"] != MODEL_BIN_SHA or
            prep["artifact_record_sha256"] != ARTIFACT_RECORD_SHA or
            prep["actual_precision"] != "<Type: 'float32'>" or
            tuple(row["identity"] for row in prep["cases"]) != IDS):
        raise ValueError("H14 0095 source-only native prep is not pinned")
    clone = json.loads(args.clone_diagnostic.read_text(encoding="utf-8"))
    if (not clone["gradient_finite"] or
            any(row["unit_cosine"] < .999 or row["unit_max_abs_difference"] > .02
                for row in clone["parity"].values()) or
            not any(row["relative_error"] <= .1 or row["absolute_error"] <= 1e-6
                    for row in clone["unit_l2_finite_differences"])):
        raise ValueError("Ghost float32 clone diagnostic did not pass")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = grouped_images(args.manifest, "development")
    sources = {}
    for identity in IDS:
        if identity not in groups or manifest["identity_splits"].get(identity) != "development":
            raise ValueError(f"Identity not in development: {identity}")
        matches = [r for r in manifest["images"] if r["identity"] == identity and
                   r["view"] == "neutral_front" and r["split"] == "development"]
        if len(matches) != 1:
            raise ValueError(f"No unique development source: {identity}")
        path = groups[identity]["neutral_front"]
        if digest(path) != matches[0]["sha256"]:
            raise ValueError(f"Source bytes changed: {identity}")
        sources[identity] = (path, matches[0]["sha256"])
    thresholds = {name: Calibration(**records[name]["calibration"]).threshold
                  for name in ("SFace", "GhostFaceNet")}
    thresholds["OpenVINO0095"] = records["OpenVINO0095"]["calibration"]["threshold"]
    return records, prep, sources, thresholds


def surrogate_loss(source, box, carrier, delta, points, prep_case, models,
                   clean, thresholds, arm):
    """Exact JPEG forward and fixed-grid BPDA; exactly two forwards/condition."""
    import torch
    import torch.nn.functional as F
    from fckface_lab.ghostface import author_align
    from fckface_lab.imaging import export_jpeg, make_variants
    from optimize_alignment_dots import crop_offset, ghost_matrix, warp_grid, ghost_term
    from optimize_gradient_art import affine, aligned_rgb

    jpeg = export_jpeg(carrier.render(delta))
    variants = make_variants(jpeg, box)
    scores = {"SFace": [], "GhostFaceNet": [], "OpenVINO0095": []}
    source_s = torch.from_numpy(np.asarray(clean["SFace"], np.float32))
    source_o = torch.from_numpy(np.asarray(clean["OpenVINO0095"], np.float32))
    cosines = {}
    for condition in OBJECTIVE:
        variant = variants[condition]
        if variant.status != "valid" or variant.image is None:
            raise RuntimeError(f"Invalid surrogate condition: {condition}/{variant.reason}")
        offset = crop_offset(condition, source.shape)
        landmarks = points[condition]
        sm = affine(landmarks)
        sg = warp_grid(carrier, np.vstack((sm, [0., 0., 1.])), offset)
        sd = F.grid_sample(delta.unsqueeze(0), sg, mode="bilinear",
                           padding_mode="zeros", align_corners=True)[0]
        sc = aligned_rgb(variant.image, sm)
        se = torch.from_numpy(np.ascontiguousarray(sc.transpose(2, 0, 1).astype(np.float32)))
        s_raw = models["SFace"]((se + sd - sd.detach()).unsqueeze(0)).reshape(-1)
        s_cos = F.cosine_similarity(s_raw, source_s, dim=0)
        scores["SFace"].append((s_cos-thresholds["SFace"])/(1-thresholds["SFace"]))
        cosines[f"SFace:{condition}"] = float(s_cos.detach())
        if arm in ("sg_global_max", "sg_balanced"):
            gg = warp_grid(carrier, ghost_matrix(landmarks), offset)
            gd = F.grid_sample(delta.unsqueeze(0), gg, mode="bilinear",
                               padding_mode="zeros", align_corners=True)[0]
            gc = author_align(variant.image, landmarks)
            ge = torch.from_numpy(np.ascontiguousarray(gc.transpose(2, 0, 1).astype(np.float32)))
            gp = ge + gd - gd.detach()
            g_term, g_cos = ghost_term(models["GhostFaceNet"],
                                       gp.detach().permute(1, 2, 0).numpy(),
                                       clean["GhostFaceNet"],
                                       thresholds["GhostFaceNet"], gp)
            scores["GhostFaceNet"].append(g_term)
            cosines[f"GhostFaceNet:{condition}"] = g_cos
        else:
            record = prep_case["conditions"][condition]
            row = np.asarray(record["row"], np.float32)
            exact = exact_input(variant.image, row).astype(np.float32)
            geom = geometry(variant.image.shape, row)
            od = residual_bgr128(delta, carrier, geom, offset)
            oe = torch.from_numpy(exact)
            o_raw = models["OpenVINO0095"]((oe + od - od.detach()).unsqueeze(0)).reshape(-1)
            o_cos = F.cosine_similarity(o_raw, source_o, dim=0)
            scores["OpenVINO0095"].append(
                (o_cos-thresholds["OpenVINO0095"])/(1-thresholds["OpenVINO0095"]))
            cosines[f"OpenVINO0095:{condition}"] = float(o_cos.detach())
    if arm == "sg_global_max":
        loss = torch.stack(scores["SFace"] + scores["GhostFaceNet"]).max()
    else:
        second = "GhostFaceNet" if arm == "sg_balanced" else "OpenVINO0095"
        loss = (torch.stack(scores["SFace"]).max() + torch.stack(scores[second]).max()) / 2
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("Nonfinite H14 surrogate margin")
    return loss, jpeg, cosines


def optimize_arm(source, box, points, prep_case, models, clean, thresholds, arm):
    import torch
    from optimize_gradient_art import Carrier, affine

    carrier = Carrier(source, box, affine(points["export"]), "dots_14x14")
    rng = np.random.default_rng(0)
    coefficients = torch.nn.Parameter(torch.from_numpy(
        (rng.standard_normal((3, 14, 14))*.1).astype(np.float32)))
    optimizer = torch.optim.Adam([coefficients], lr=ADAM_LR)
    logs = []
    chosen = None
    started = perf_counter()
    for step in range(1, STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        delta = carrier.project(coefficients)
        loss, jpeg, cosines = surrogate_loss(source, box, carrier, delta, points,
                                             prep_case, models, clean, thresholds, arm)
        if step in (1, 6, 12, 18):
            logs.append({"step": step, "arm": arm, "surrogate_loss": float(loss.detach()),
                         "source_cosines": cosines, "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                         "pre_jpeg_face_rms": float(torch.sqrt(torch.mean(delta.detach()[:, carrier.face]**2))),
                         "max_abs_channel_delta": float(delta.detach().abs().max())})
        if step == STEPS:
            chosen = jpeg  # forward image before update; no checkpoint selection
        loss.backward()
        gradient = coefficients.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().sum() > 0):
            raise RuntimeError(f"Missing/zero/nonfinite H14 coefficient gradient at step {step}")
        if step < STEPS:
            optimizer.step()
    if chosen is None:
        raise AssertionError("No fixed forward-step-18 JPEG")
    return chosen, logs, perf_counter()-started


def distortion_after_freeze(cases: list, frozen: list, output: Path) -> None:
    from fckface_lab.imaging import export_jpeg, make_variants
    from fckface_lab.patterns import actual_distortion

    result = {}
    for case in cases:
        source, box, identity = case["source"], case["box"], case["identity"]
        clean = make_variants(export_jpeg(source), box)
        result[identity] = {}
        for arm in ARMS:
            selected = next(row for row in frozen if row["identity"] == identity and row["arm"] == arm)
            path = Path(selected["path"])
            if digest(path) != selected["sha256"]:
                raise ValueError("Frozen H14 JPEG changed")
            variants = make_variants(path.read_bytes(), box)
            metrics = {}
            for condition in CONDITIONS:
                a, b = clean[condition], variants[condition]
                if a.status != "valid" or b.status != "valid":
                    metrics[condition] = {"status": "inconclusive", "reason": a.reason or b.reason}
                    continue
                x0, y0, x1, y1 = b.target_box
                metrics[condition] = {"status": "valid", **actual_distortion(
                    a.image, b.image, (x0, y0, x1-x0, y1-y0))}
            result[identity][arm] = metrics
    write_json(output / "distortion.json", result)
    comparisons = {}
    for identity in IDS:
        comparisons[identity] = {}
        for label, first, second in (("B_minus_A", ARMS[0], ARMS[1]),
                                     ("C_minus_B", ARMS[1], ARMS[2])):
            gaps = {}
            for condition in CONDITIONS:
                a, b = result[identity][first][condition], result[identity][second][condition]
                gaps[condition] = (abs(a["face_rms"]-b["face_rms"])
                                   if a["status"] == b["status"] == "valid" else None)
            comparisons[identity][label] = {
                "absolute_face_rms_gap_by_condition": gaps,
                "all_valid_and_within_0_25": all(value is not None and value <= .25
                                                  for value in gaps.values())}
    write_json(output / "distortion-pairing.json", comparisons)


def contact_sheets(cases: list, frozen: list, output: Path) -> None:
    from PIL import Image, ImageDraw

    for case in cases:
        identity, box = case["identity"], case["box"]
        items = [("Clean", Image.fromarray(case["source"]))]
        for arm in ARMS:
            path = next(Path(row["path"]) for row in frozen if row["identity"] == identity and row["arm"] == arm)
            items.append((arm, Image.open(path).convert("RGB")))
        for crop in (False, True):
            canvas = Image.new("RGB", (450*len(items), 560), "#f5f5f5")
            draw = ImageDraw.Draw(canvas)
            x0, y0, x1, y1 = box
            pad = .15*(x1-x0)
            rect = (max(0, int(x0-pad)), max(0, int(y0-pad)),
                    min(case["source"].shape[1], int(x1+pad)),
                    min(case["source"].shape[0], int(y1+pad)))
            for index, (label, image) in enumerate(items):
                tile = image.crop(rect) if crop else image.copy()
                tile.thumbnail((438, 510), Image.Resampling.LANCZOS)
                canvas.paste(tile, (450*index+(450-tile.width)//2, 40+(510-tile.height)//2))
                draw.text((450*index+12, 12), label, fill="#111111")
            canvas.save(output / f"{identity}-{'face' if crop else 'full'}.png")


def run(args) -> None:
    """Source-only freeze. Root review precedes every separate-gallery score."""
    import onnx
    import torch
    from onnx2torch import convert
    from fckface_lab.ghostface import GhostFaceModel
    from fckface_lab.imaging import decode_image, export_jpeg, make_variants
    from fckface_lab.recognition import SFaceModel
    from optimize_alignment_dots import detect_points, tf_value_gradient_functions
    from optimize_ensemble_dots import load_float32_clone

    pullback = synthetic_pullback_check()
    if not pullback["passed"]:
        raise ValueError("H14 0095 fixed-ROI residual pullback check failed")
    records, prep, sources, thresholds = checked_inputs(args)
    cv2.setNumThreads(2); torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / "source-snapshot"; snapshot.mkdir()
    source_hashes = {}
    for name in ("optimize_balanced_basis.py", "h14_0095_alignment.py",
                 "optimize_transfer_dots.py", "optimize_alignment_dots.py",
                 "optimize_gradient_art.py", "optimize_ensemble_dots.py"):
        path = Path(__file__).with_name(name)
        target = snapshot / name
        shutil.copyfile(path, target)
        source_hashes[name] = digest(path)
        if digest(target) != source_hashes[name]:
            raise ValueError(f"Source snapshot changed: {name}")
    package = snapshot / "fckface_lab"; package.mkdir()
    for path in sorted((Path(__file__).parent / "fckface_lab").glob("*.py")):
        target = package / path.name
        shutil.copyfile(path, target)
        source_hashes[f"fckface_lab/{path.name}"] = digest(path)
    models_dir = args.out / "model-inputs"; models_dir.mkdir()
    pinned = {}
    expected_models = {
        "yunet": records["SFace"]["pipeline"]["model_artifacts"]["yunet"],
        "sface": records["SFace"]["pipeline"]["model_artifacts"]["sface"],
        "ghostface": records["GhostFaceNet"]["pipeline"]["model_artifacts"]["ghostface"],
        "openvino_onnx": {"sha256": ONNX_SHA, "bytes": ONNX_BYTES},
    }
    for name in ("yunet", "sface", "ghostface", "openvino_onnx"):
        path = getattr(args, name)
        target = models_dir / path.name
        shutil.copyfile(path, target)
        expected = expected_models[name]
        if digest(target) != expected["sha256"] or target.stat().st_size != expected["bytes"]:
            raise ValueError(f"Private model copy differs from frozen calibration: {name}")
        pinned[name] = target
    protocol = {
        "hypothesis": "H14 balanced model gradients and compact 0095 basis, matched two-model budget",
        "scope": "two prior development identities, original neutral_front only; no gallery scoring in optimizer",
        "identities": IDS, "arms": ARMS, "steps": STEPS, "effective_updates": 17,
        "objective_conditions": OBJECTIVE, "final_conditions": CONDITIONS,
        "gradient_model_condition_forwards_per_arm": 144,
        "native_checkpoint_queries_per_arm": 0,
        "selection": "unconditional forward-step18 JPEG; no checkpoint selection",
        "carrier": "588 RGB coefficients, H5 literal 14x14 radius-.3 dots, seed0, RMS16, cap64",
        "optimizer": "Adam learning rate .15; same fixed clean-condition landmarks and JPEG-forward BPDA",
        "loss_A": "global max across SFace/GhostFaceNet normalized source margins and four conditions",
        "loss_B": "mean of SFace and GhostFaceNet per-model condition maxima",
        "loss_C": "mean of SFace and 0095 per-model condition maxima",
        "normalized_margin": "(cosine - frozen model threshold)/(1 - frozen model threshold)",
        "openvino_forward": "exact decoded edited JPEG, fixed clean YuNet row, clipped integer BGR ROI, official SVD inverse-map nearest warp, linear resize128",
        "openvino_backward": "OpenCV nearest index-map gather plus Torch float bilinear resize from carrier ROI; crop90 offset; fixed-geometry BPDA, not JPEG derivative",
        "source_only_prep_sha256": digest(args.openvino_prep),
        "openvino_gradient_diagnostic_sha256": GRADIENT_DIAGNOSTIC_SHA,
        "openvino_crop_diagnostic_sha256": CROP_DIAGNOSTIC_SHA,
        "manifest_sha256": digest(args.manifest),
        "calibration_sha256": {"SFace": digest(args.sface_calibration),
                               "GhostFaceNet": digest(args.ghostface_calibration),
                               "OpenVINO0095": CAL_SHA},
        "model_sha256": {name: digest(path) for name, path in pinned.items()},
        "clone_diagnostic_sha256": digest(args.clone_diagnostic),
        "source_sha256": source_hashes, "thresholds": thresholds,
        "source_photos_sha256": {name: row[1] for name, row in sources.items()},
        "forward_comparisons": "B-A isolates balanced loss; C-B isolates compact feature basis",
        "scientific_gate": "all seven development scores valid; paired condition face-RMS difference <=.25; each tested branch lowers ArcFace worst own-gallery cosine >=.05 on both people",
        "candidate_expansion_gate": "all seven nonmatch on SFace/GhostFaceNet/0095 for both people; independent human appearance and reserved final panel remain separate",
        "gallery_barrier": "all six JPEGs freeze, then root full/crop appearance review, then separate model scorers",
    }
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "openvino-residual-pullback.json", pullback)
    native = {"SFace": SFaceModel(pinned["yunet"], pinned["sface"]),
              "GhostFaceNet": GhostFaceModel(pinned["yunet"], pinned["ghostface"])}
    clone = load_float32_clone(native["GhostFaceNet"].model)
    ghost_gradient, ghost_report = tf_value_gradient_functions(clone)
    write_json(args.out / "ghost-gradient-preflight.json", ghost_report)
    if not ghost_report["compiled_accepted"]:
        raise RuntimeError("Ghost gradient parity failed")
    torch_sface = convert(onnx.load(str(pinned["sface"]))).eval()
    torch_0095 = convert(onnx.load(str(pinned["openvino_onnx"]))).eval()
    for model in (torch_sface, torch_0095):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    torch_models = {"SFace": torch_sface, "GhostFaceNet": ghost_gradient,
                    "OpenVINO0095": torch_0095}
    frozen, reports, cases = [], [], []
    started = perf_counter()
    try:
        for identity in IDS:
            path, source_sha = sources[identity]
            if digest(path) != source_sha:
                raise ValueError(f"Source changed before optimization: {identity}")
            source = decode_image(path)
            reference = {name: model.embed(source) for name, model in native.items()}
            if any(value.status != "valid" for value in reference.values()):
                raise ValueError(f"Invalid native clean source: {identity}")
            box = reference["SFace"].selected_box
            if any(np.max(np.abs(np.asarray(value.selected_box)-box)) > .01
                   for value in reference.values()):
                raise ValueError("SFace/Ghost source boxes differ")
            prep_case = next(row for row in prep["cases"] if row["identity"] == identity)
            if (prep_case["source_sha256"] != source_sha or
                    np.max(np.abs(np.asarray(prep_case["source_box"])-box)) > .01):
                raise ValueError("Frozen native 0095 source differs")
            feature = np.asarray(prep_case["source_feature"], np.float32)
            if (feature.shape != (256,) or
                    hashlib.sha256(feature.tobytes()).hexdigest() != prep_case["source_feature_sha256"] or
                    not np.isfinite(feature).all() or abs(np.linalg.norm(feature)-1) > 1e-4):
                raise ValueError("Frozen native 0095 clean embedding invalid")
            native_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if (native_bgr is None or native_bgr.shape != source.shape or
                    hashlib.sha256(native_bgr.tobytes()).hexdigest() != prep_case["source_cv2_bgr_sha256"]):
                raise ValueError("Clean native 0095 source decode changed")
            source_input = exact_input(cv2.cvtColor(native_bgr, cv2.COLOR_BGR2RGB),
                                       np.asarray(prep_case["source_row"], np.float32))
            if hashlib.sha256(source_input.tobytes()).hexdigest() != prep_case["source_input_sha256"]:
                raise ValueError("Clean 0095 source input differs from official native prep")
            clean = {name: value.feature for name, value in reference.items()}
            clean["OpenVINO0095"] = feature
            variants = make_variants(export_jpeg(source), box)
            points = {condition: detect_points(native["SFace"].detector, variants[condition])
                      for condition in OBJECTIVE}
            for condition in OBJECTIVE:
                variant = variants[condition]
                saved = prep_case["conditions"][condition]
                if (variant.jpeg is None or
                        hashlib.sha256(variant.jpeg).hexdigest() != saved["jpeg_sha256"] or
                        hashlib.sha256(exact_input(variant.image, np.asarray(saved["row"], np.float32)).tobytes()).hexdigest() != saved["input_sha256"]):
                    raise ValueError(f"Frozen 0095 clean condition differs: {identity}/{condition}")
            case = {"identity": identity, "source_sha256": source_sha, "box": box,
                    "clean_points_by_condition": {name: row.tolist() for name, row in points.items()},
                    "arms": {}}
            cases.append({"identity": identity, "source": source, "box": box})
            for arm in ARMS:
                jpeg, logs, seconds = optimize_arm(source, box, points, prep_case,
                                                   torch_models, clean, thresholds, arm)
                directory = args.out / identity / arm
                directory.mkdir(parents=True)
                target = directory / "selected.jpg"
                target.write_bytes(jpeg)
                write_json(directory / "surrogate-progress.json", logs)
                record = {"identity": identity, "method": "H14_balanced_basis", "arm": arm,
                          "path": str(target), "sha256": digest(target),
                          "source_view": "neutral_front"}
                frozen.append(record)
                case["arms"][arm] = {"path": str(target), "sha256": record["sha256"],
                                     "forward_step": 18, "effective_updates": 17,
                                     "native_checkpoint_queries": 0, "seconds": seconds,
                                     "pre_jpeg_face_rms": logs[-1]["pre_jpeg_face_rms"]}
                write_json(args.out / "freeze-progress.json", {"cases": reports+[case],
                                                                  "frozen": frozen})
            reports.append(case)
        if len(frozen) != 6:
            raise AssertionError("Six fixed H14 JPEGs not frozen")
        write_json(args.out / "frozen-inputs.json", {"schema_version": 1, "items": frozen})
        write_json(args.out / "freeze-results.json", {
            "status": "six_exports_frozen_before_gallery", "protocol_sha256": digest(args.out / "protocol.json"),
            "frozen_manifest_sha256": digest(args.out / "frozen-inputs.json"),
            "optimization_seconds_excluding_load": perf_counter()-started, "cases": reports})
        distortion_after_freeze(cases, frozen, args.out)
        contact_sheets(cases, frozen, args.out)
    except Exception as exc:
        write_json(args.out / "failure.json", {"type": type(exc).__name__,
                                               "message": str(exc), "frozen_count": len(frozen)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "sface-calibration", "ghostface-calibration",
                 "openvino-calibration", "openvino-pipeline", "openvino-erratum",
                 "openvino-artifacts", "openvino-onnx", "openvino-gradient-diagnostic",
                 "openvino-crop-diagnostic",
                 "openvino-prep", "yunet", "sface", "ghostface", "clone-diagnostic", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
