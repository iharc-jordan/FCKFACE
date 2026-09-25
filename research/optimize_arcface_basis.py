"""H15 source-only SFace+ArcFace balanced-dot baseline.

Two explicit phases: `preflight` freezes inputs and verifies the converted
ArcFace component on two original sources plus eight clean processed inputs;
`optimize` uses that frozen source-only record and exports two step-18 JPEGs.
Neither phase opens another same-person view or a reserved final recognizer.
The pretrained ArcFace weights are research-only and stay outside Git.
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
from h15_arcface_alignment import exact_blob, matrix, residual_blob, synthetic_pullback_check

IDS = ("frll-001", "frll-003")
OBJECTIVE = ("export", "jpeg75_420", "blur", "crop90")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960",
              "half_restore", "crop90", "blur")
STEPS, ADAM_LR = 18, .15
FRLL_SHA = "2509c48bd7852251f7a3778cc7e1e320e17003d1bb56cca6dbf84998b4931600"
SFACE_CAL_SHA = "03f93365189dae853e306bdbed7a96536926628dfa1302d363d2779db98c8f04"
ARC_CAL_SHA = "a186f3d285f4cff65d8c187a65f5929c19b13b86f8170583dee34dccab9bb56a"
ARC_ONNX_SHA = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
ARC_DET_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
ARC_ADAPTER_SHA = "7238216b83174131f39b5817ad24f56c8c546a3eab401175a2a3d536145ecaa0"
ARC_COMPONENT_SHA = "ac5a8a994551fd1befbe573f601e1205ab4699a3d3d72a00b4bd5f460a243eb7"
ARC_COMPONENT_SOURCE_SHA = "ff2adad7e5f3e2eeb46a34ea355fb6c344ff089424236a688eacd4c9465ae2df"
ARC_COMPONENT_PROTOCOL_SHA = "c2776514e095fa0e891f9104b9843f681c4aaa3465a97aece3c52cd0df449088"
H14_HASHES = {
    "protocol.json": "bb78d04424f8b099dc080cfe4509c427266fdc73c99d41f8b642d5cc9abe3dab",
    "frozen-inputs.json": "512f8eb8976b3859b0eb236d36f10b17c092c3c76399e573b21edb3d9bf15773",
    "freeze-results.json": "69f8124816731c086e777359eb4e1bdbfee1455c4ee66be8b3454c0e69418df7",
    "distortion.json": "82c1728b6fec65a2707487b3ae5646a402c5fe32a80b1357848373357b1f9aff",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checked_metadata(args):
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(repo):
        raise ValueError("Private output must be outside Git")
    for path, expected in ((args.manifest, FRLL_SHA),
                           (args.sface_calibration, SFACE_CAL_SHA),
                           (args.arc_calibration, ARC_CAL_SHA),
                           (args.arc_dir / "w600k_r50.onnx", ARC_ONNX_SHA),
                           (args.arc_dir / "det_10g.onnx", ARC_DET_SHA),
                           (Path(__file__).with_name("arcface_development.py"), ARC_ADAPTER_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen H15 dependency changed: {path}")
    for name, expected in H14_HASHES.items():
        if digest(args.h14 / name) != expected:
            raise ValueError(f"Frozen H14 control changed: {name}")
    h14 = json.loads((args.h14 / "frozen-inputs.json").read_text(encoding="utf-8"))
    controls = {(row["identity"], row["arm"]): row for row in h14["items"]}
    for identity in IDS:
        for arm in ("sg_balanced", "s0095_balanced"):
            row = controls[(identity, arm)]
            if digest(Path(row["path"])) != row["sha256"]:
                raise ValueError("Frozen H14 control JPEG changed")
    s_cal = json.loads(args.sface_calibration.read_text(encoding="utf-8"))
    a_cal = json.loads(args.arc_calibration.read_text(encoding="utf-8"))
    if (s_cal.get("status") != "complete" or a_cal.get("status") != "complete" or
            s_cal["dataset"]["manifest_sha256"] != FRLL_SHA or
            a_cal["dataset"]["manifest_sha256"] != FRLL_SHA or
            a_cal["pipeline"]["provenance"]["commit"] != "1480e705287bc5d59f923b46c260ec6e3e4150f6" or
            a_cal["pipeline"]["adapter_sha256"] != ARC_ADAPTER_SHA):
        raise ValueError("Frozen SFace/ArcFace calibration metadata differs")
    for name, path in (("yunet", args.yunet), ("sface", args.sface)):
        artifact = s_cal["pipeline"]["model_artifacts"][name]
        if digest(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"SFace calibration model bytes differ: {name}")
    for name, expected in s_cal["pipeline"]["source_sha256"].items():
        if name.startswith("fckface_lab/") and digest(repo / "research" / name) != expected:
            raise ValueError(f"Calibrated SFace source changed: {name}")
    for name, expected in (("w600k_r50.onnx", ARC_ONNX_SHA), ("det_10g.onnx", ARC_DET_SHA)):
        if a_cal["pipeline"]["provenance"]["assets"][name]["sha256"] != expected:
            raise ValueError(f"ArcFace calibration model differs: {name}")
    upstream = {"scrfd.py": "model_zoo/scrfd.py",
                "arcface_onnx.py": "model_zoo/arcface_onnx.py",
                "face_align.py": "utils/face_align.py",
                "coreml_cache.py": "model_zoo/coreml_cache.py",
                "onnxruntime_utils.py": "model_zoo/onnxruntime_utils.py"}
    for name, relative in upstream.items():
        expected = a_cal["pipeline"]["provenance"]["assets"][name]["sha256"]
        if digest(args.arc_dir / "insightface_official" / relative) != expected:
            raise ValueError(f"Pinned official ArcFace source changed: {name}")
    groups = grouped_images(args.manifest, "development")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = {}
    for identity in IDS:
        entries = [row for row in manifest["images"] if row["identity"] == identity and
                   row["view"] == "neutral_front" and row["split"] == "development"]
        if len(entries) != 1 or manifest["identity_splits"][identity] != "development":
            raise ValueError(f"No unique development source: {identity}")
        path = groups[identity]["neutral_front"]
        if digest(path) != entries[0]["sha256"]:
            raise ValueError(f"Source bytes changed: {identity}")
        sources[identity] = (path, entries[0]["sha256"])
    return s_cal, a_cal, sources, controls


def checked_component(path: Path) -> dict:
    """Component gate is external and must pass before loading a Torch model."""
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = report.get("cases", {})
    post = report.get("postflight", {})
    expected_cases = {"range", "flat", "random", "frll-024", "frll-036"}
    if (digest(path) != ARC_COMPONENT_SHA or report.get("status") != "pass" or
            report.get("failure") is not None or post.get("unchanged") is not True or
            post.get("script_sha256") != ARC_COMPONENT_SOURCE_SHA or
            report.get("protocol_sha256") != ARC_COMPONENT_PROTOCOL_SHA or
            post.get("protocol_sha256") != ARC_COMPONENT_PROTOCOL_SHA or
            post.get("original_model_sha256") != ARC_ONNX_SHA or
            post.get("private_model_sha256") != ARC_ONNX_SHA or
            set(cases) != expected_cases or
            set(report.get("completed_cases", [])) != expected_cases or
            set(report.get("attempted_cases", [])) != expected_cases):
        raise ValueError("Frozen native ArcFace Torch component diagnostic did not pass")
    for case in cases.values():
        if (case.get("raw_max_abs_vs_ort", float("inf")) > 1e-3 or
                case.get("unit_cosine_vs_ort", 0) < .99999 or
                case.get("gradient_finite") is not True or
                case.get("gradient_pass") is not True or
                not np.isfinite(case.get("gradient_l2", np.nan)) or
                case.get("gradient_l2", 0) <= 0 or
                len(case.get("coordinates", [])) != 3 or
                not all(row.get("nontrivial") is True and row.get("pass_relative") is True
                        for row in case["coordinates"])):
            raise ValueError("ArcFace Torch component case failed")
    return report


def source_snapshot(args, models_dir: Path) -> dict:
    snapshot = args.out / "source-snapshot"
    snapshot.mkdir()
    hashes = {}
    for name in ("optimize_arcface_basis.py", "h15_arcface_alignment.py",
                 "arcface_development.py", "optimize_balanced_basis.py",
                 "optimize_alignment_dots.py", "optimize_gradient_art.py",
                 "optimize_ensemble_dots.py", "arcface_onnx_gradient.py"):
        source = Path(__file__).with_name(name)
        if name == "arcface_onnx_gradient.py" and digest(source) != ARC_COMPONENT_SOURCE_SHA:
            raise ValueError("Accepted peak-memory helper source changed")
        shutil.copyfile(source, snapshot / name)
        hashes[name] = digest(source)
        if digest(snapshot / name) != hashes[name]:
            raise ValueError("Source snapshot differs")
    package = snapshot / "fckface_lab"
    package.mkdir()
    for source in sorted((Path(__file__).parent / "fckface_lab").glob("*.py")):
        shutil.copyfile(source, package / source.name)
        hashes[f"fckface_lab/{source.name}"] = digest(source)
    shutil.copytree(args.arc_dir / "insightface_official",
                    models_dir / "insightface_official",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    official = json.loads(args.arc_calibration.read_text(encoding="utf-8"))["pipeline"]["provenance"]["assets"]
    for name, relative in (("scrfd.py", "model_zoo/scrfd.py"),
                           ("arcface_onnx.py", "model_zoo/arcface_onnx.py"),
                           ("face_align.py", "utils/face_align.py"),
                           ("coreml_cache.py", "model_zoo/coreml_cache.py"),
                           ("onnxruntime_utils.py", "model_zoo/onnxruntime_utils.py")):
        if digest(models_dir / "insightface_official" / relative) != official[name]["sha256"]:
            raise ValueError(f"Private official source copy changed: {name}")
    for name, source in (("yunet.onnx", args.yunet), ("sface.onnx", args.sface),
                         ("det_10g.onnx", args.arc_dir / "det_10g.onnx"),
                         ("w600k_r50.onnx", args.arc_dir / "w600k_r50.onnx")):
        shutil.copyfile(source, models_dir / name)
        if digest(models_dir / name) != digest(source):
            raise ValueError(f"Private model copy changed: {name}")
    return hashes


def native_source_preflight(args, s_cal, a_cal, sources) -> dict:
    """Runs only after root separately authorizes official source-only parity."""
    import onnx
    import torch
    from onnx2torch import convert
    import arcface_development as arc_module
    from fckface_lab.imaging import decode_image, export_jpeg, make_variants
    from fckface_lab.recognition import SFaceModel
    from optimize_alignment_dots import detect_points

    model_dir = args.out / "model-inputs"
    arc_module.ARC = model_dir
    cv2.setNumThreads(2); torch.set_num_threads(2)
    native_arc = arc_module.ArcFaceModel()
    native_sface = SFaceModel(model_dir / "yunet.onnx", model_dir / "sface.onnx")
    torch_arc = convert(onnx.load(str(model_dir / "w600k_r50.onnx"))).eval()
    for parameter in torch_arc.parameters():
        parameter.requires_grad_(False)
    cases = []
    for identity in IDS:
        source_path, source_sha = sources[identity]
        source = decode_image(source_path)
        sref = native_sface.embed(source)
        if sref.status != "valid":
            raise ValueError(f"Invalid clean SFace source: {identity}/{sref.reason}")
        box = sref.selected_box
        original = native_arc.embed(source, expected_box=box)
        if original.status != "valid":
            raise ValueError(f"Invalid clean ArcFace original: {identity}/{original.reason}")
        variants = make_variants(export_jpeg(source), box)
        rows = {}
        inputs = [("original", source, original)]
        for condition in OBJECTIVE:
            variant = variants[condition]
            if variant.status != "valid" or variant.jpeg is None:
                raise ValueError(f"Invalid clean condition: {identity}/{condition}")
            embedded = native_arc.embed(variant.image, expected_box=variant.target_box)
            if embedded.status != "valid":
                raise ValueError(f"Invalid clean ArcFace condition: {identity}/{condition}/{embedded.reason}")
            inputs.append((condition, variant.image, embedded))
        for name, image, embedded in inputs:
            affine = matrix(embedded.landmarks)
            blob = exact_blob(image, affine)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            crop = cv2.warpAffine(bgr, affine, (112, 112), borderValue=0.0)
            official_crop = native_arc.face_align.norm_crop(
                bgr, landmark=embedded.landmarks, image_size=112)
            official_blob = cv2.dnn.blobFromImages(
                [official_crop], 1.0/127.5, (112, 112), (127.5,)*3,
                swapRB=True).astype(np.float32, copy=False)
            crop_equal = bool(np.array_equal(crop, official_crop))
            blob_equal = bool(np.array_equal(blob, official_blob))
            if not crop_equal or not blob_equal:
                raise ValueError(f"H15 official crop/blob pixels differ: {identity}/{name}")
            native_raw = native_arc.recognizer.get_feat(official_crop).reshape(-1)
            with torch.no_grad():
                converted = torch_arc(torch.from_numpy(blob)).reshape(-1).numpy()
            native_unit = native_raw / np.linalg.norm(native_raw)
            torch_unit = converted / np.linalg.norm(converted)
            raw_error = float(np.max(np.abs(native_raw-converted)))
            unit_cosine = float(np.dot(native_unit, torch_unit))
            if (not np.isfinite(converted).all() or raw_error > 1e-3 or
                    unit_cosine < .99999 or
                    np.max(np.abs(native_unit-embedded.feature)) > 1e-5):
                raise ValueError(f"Official/native converted ArcFace parity failed: {identity}/{name}")
            rows[name] = {"arc_landmarks": embedded.landmarks.tolist(),
                          "official_crop_sha256": hashlib.sha256(official_crop.tobytes()).hexdigest(),
                          "constructed_crop_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                          "official_blob_sha256": hashlib.sha256(official_blob.tobytes()).hexdigest(),
                          "arc_crop_blob_sha256": hashlib.sha256(blob.tobytes()).hexdigest(),
                          "uint8_crop_exact_equal": crop_equal, "normalized_blob_exact_equal": blob_equal,
                          "raw_max_abs_vs_official": raw_error,
                          "unit_cosine_vs_official": unit_cosine,
                          "native_feature": embedded.feature.tolist()}
            if name != "original":
                variant = variants[name]
                rows[name]["clean_jpeg_sha256"] = hashlib.sha256(variant.jpeg).hexdigest()
                rows[name]["sface_landmarks"] = detect_points(native_sface.detector, variant).tolist()
        cases.append({"identity": identity, "source_sha256": source_sha,
                      "box": box, "sface_source_feature": sref.feature.tolist(),
                      "arc_source_feature": original.feature.tolist(), "conditions": rows})
    return {"status": "pass", "comparison_count": sum(len(row["conditions"]) for row in cases),
            "scope": "two original sources plus eight clean objective-condition inputs",
            "cases": cases, "sface_threshold": s_cal["calibration"]["threshold"],
            "arcface_threshold": a_cal["calibration"]["threshold"]}


def end_integrity(args, protocol: dict, sources: dict) -> dict:
    """Rehash source originals, executed code, snapshots and private model copies."""
    mismatches = []
    for identity, (path, expected) in sources.items():
        if digest(path) != expected:
            mismatches.append(f"source:{identity}")
    for name, expected in protocol["source_sha256"].items():
        if (digest(Path(__file__).parent / name) != expected or
                digest(args.out / "source-snapshot" / name) != expected):
            mismatches.append(f"code:{name}")
    for name, expected in protocol["model_sha256"].items():
        filename = {"yunet": "yunet.onnx", "sface": "sface.onnx",
                    "arcface": "w600k_r50.onnx", "scrfd": "det_10g.onnx"}[name]
        if digest(args.out / "model-inputs" / filename) != expected:
            mismatches.append(f"private-model:{name}")
    for name, original in (("yunet", args.yunet), ("sface", args.sface),
                           ("arcface", args.arc_dir / "w600k_r50.onnx"),
                           ("scrfd", args.arc_dir / "det_10g.onnx")):
        if digest(original) != protocol["model_sha256"][name]:
            mismatches.append(f"original-model:{name}")
    for name, relative in (("scrfd.py", "model_zoo/scrfd.py"),
                           ("arcface_onnx.py", "model_zoo/arcface_onnx.py"),
                           ("face_align.py", "utils/face_align.py"),
                           ("coreml_cache.py", "model_zoo/coreml_cache.py"),
                           ("onnxruntime_utils.py", "model_zoo/onnxruntime_utils.py")):
        expected = protocol["official_source_sha256"][name]
        if (digest(args.arc_dir / "insightface_official" / relative) != expected or
                digest(args.out / "model-inputs" / "insightface_official" / relative) != expected):
            mismatches.append(f"official-code:{name}")
    if digest(args.arc_component_report) != ARC_COMPONENT_SHA:
        mismatches.append("arc-component-report")
    result = {"unchanged": not mismatches, "mismatches": mismatches,
              "source_photos_sha256": {key: digest(row[0]) for key, row in sources.items()},
              "private_model_sha256": {name: digest(args.out / "model-inputs" / filename)
                                       for name, filename in (("yunet", "yunet.onnx"),
                                           ("sface", "sface.onnx"),
                                           ("arcface", "w600k_r50.onnx"),
                                           ("scrfd", "det_10g.onnx"))}}
    if mismatches:
        raise ValueError(f"H15 end-of-phase input integrity failed: {mismatches}")
    return result


def preflight(args) -> None:
    started = perf_counter()
    from arcface_onnx_gradient import peak_bytes
    if args.out.exists():
        raise FileExistsError("New external output directory required")
    s_cal, a_cal, sources, _ = checked_metadata(args)
    checked_component(args.arc_component_report)
    synthetic = synthetic_pullback_check()
    if not synthetic["passed"]:
        raise ValueError("H15 model-free pullback failed")
    args.out.mkdir(parents=True, exist_ok=False)
    models_dir = args.out / "model-inputs"
    models_dir.mkdir()
    source_hashes = source_snapshot(args, models_dir)
    protocol = {
        "hypothesis": "H15 conventional omitted ArcFace gradient basis",
        "identities": IDS, "source_view": "neutral_front", "arms": ["sarc_balanced"],
        "controls": "frozen H14 sg_balanced and s0095_balanced; sg_global_max descriptive",
        "steps": STEPS, "effective_updates": 17, "gradient_model_condition_forwards_per_person": 144,
        "objective_conditions": OBJECTIVE, "final_conditions": CONDITIONS,
        "carrier": "H5 literal 14x14 radius-.3 RGB dots, 588 coefficients, seed0, RMS16, cap64",
        "optimizer": "Adam .15; same clean source references, fixed clean condition geometry, exact edited JPEG forward, BPDA residual",
        "loss": "mean of per-model four-condition max normalized source cosine margins for SFace and ArcFace",
        "normalized_margin": "(cosine - model threshold)/(1 - model threshold)",
        "selection": "unconditional forward-step18 JPEG, 17 updates; zero native checkpoint queries",
        "arc_forward": "official SCRFD 640/.5 source-only landmarks; skimage five-point 112 similarity; OpenCV linear BGR warp zero border; swapRB RGB; (RGB-127.5)/127.5",
        "arc_backward": "fixed clean-condition affine grid_sample bilinear ROI residual /127.5; crop90 offset; no detector/JPEG/rounding derivative",
        "component_diagnostic_sha256": digest(args.arc_component_report),
        "manifest_sha256": FRLL_SHA,
        "calibration_sha256": {"SFace": SFACE_CAL_SHA, "ArcFace": ARC_CAL_SHA},
        "model_sha256": {name: digest(path) for name, path in (("yunet", models_dir / "yunet.onnx"),
            ("sface", models_dir / "sface.onnx"), ("arcface", models_dir / "w600k_r50.onnx"),
            ("scrfd", models_dir / "det_10g.onnx"))},
        "official_source_sha256": {name: a_cal["pipeline"]["provenance"]["assets"][name]["sha256"]
                                   for name in ("scrfd.py", "arcface_onnx.py", "face_align.py",
                                                "coreml_cache.py", "onnxruntime_utils.py")},
        "h14_control_sha256": H14_HASHES, "source_sha256": source_hashes,
        "source_photos_sha256": {name: row[1] for name, row in sources.items()},
        "development_expansion_gate": "all four models valid in all seven conditions; SFace+ArcFace all-seven nonmatch and at least one omitted model (GhostFaceNet or 0095) all-seven nonmatch for each person; paired H14 B/C same-condition face-RMS gap <=.25",
        "gallery_barrier": "both JPEGs freeze; root full/crop appearance screen; separate seven-condition four-model scoring",
        "limitations": "ArcFace already development-influenced, not independent transfer; official weights research-only and excluded from public bundle",
    }
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "model-free-pullback.json", synthetic)
    try:
        result = native_source_preflight(args, s_cal, a_cal, sources)
        result["protocol_sha256"] = digest(args.out / "protocol.json")
        result["component_diagnostic_sha256"] = digest(args.arc_component_report)
        result["elapsed_seconds"] = perf_counter()-started
        result["peak_working_set_bytes"] = peak_bytes()
        result["postflight"] = end_integrity(args, protocol, sources)
        write_json(args.out / "source-only-parity.json", result)
    except Exception as exc:
        write_json(args.out / "failure.json", {"phase": "preflight", "type": type(exc).__name__,
                                              "message": str(exc)})
        raise


def surrogate_loss(source, box, carrier, delta, conditions, models, clean, thresholds):
    import torch
    import torch.nn.functional as F
    from fckface_lab.imaging import export_jpeg, make_variants
    from optimize_alignment_dots import crop_offset, warp_grid
    from optimize_gradient_art import affine, aligned_rgb

    jpeg = export_jpeg(carrier.render(delta))
    variants = make_variants(jpeg, box)
    scores = {"SFace": [], "ArcFace": []}
    cosines = {}
    source_s = torch.from_numpy(np.asarray(clean["SFace"], np.float32))
    source_a = torch.from_numpy(np.asarray(clean["ArcFace"], np.float32))
    for condition in OBJECTIVE:
        variant = variants[condition]
        if variant.status != "valid" or variant.image is None:
            raise RuntimeError(f"Invalid surrogate condition: {condition}/{variant.reason}")
        offset = crop_offset(condition, source.shape)
        points = conditions[condition]
        s_affine = affine(np.asarray(points["sface_landmarks"], np.float32))
        s_grid = warp_grid(carrier, np.vstack((s_affine, [0., 0., 1.])), offset)
        s_residual = F.grid_sample(delta.unsqueeze(0), s_grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=True)
        s_crop = aligned_rgb(variant.image, s_affine)
        s_exact = torch.from_numpy(np.ascontiguousarray(
            s_crop.transpose(2, 0, 1)[None].astype(np.float32)))
        s_raw = models["SFace"](s_exact + s_residual - s_residual.detach()).reshape(-1)
        s_cos = F.cosine_similarity(s_raw, source_s, dim=0)
        scores["SFace"].append((s_cos-thresholds["SFace"])/(1-thresholds["SFace"]))
        cosines[f"SFace:{condition}"] = float(s_cos.detach())
        a_affine = matrix(np.asarray(points["arc_landmarks"], np.float32))
        a_exact = torch.from_numpy(exact_blob(variant.image, a_affine))
        a_residual = residual_blob(delta, carrier, a_affine, offset)
        a_raw = models["ArcFace"](a_exact + a_residual - a_residual.detach()).reshape(-1)
        a_cos = F.cosine_similarity(a_raw, source_a, dim=0)
        scores["ArcFace"].append((a_cos-thresholds["ArcFace"])/(1-thresholds["ArcFace"]))
        cosines[f"ArcFace:{condition}"] = float(a_cos.detach())
    loss = (torch.stack(scores["SFace"]).max() + torch.stack(scores["ArcFace"]).max()) / 2
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("Nonfinite H15 balanced margin")
    return loss, jpeg, cosines


def optimize_one(source, box, conditions, models, clean, thresholds):
    import torch
    from optimize_gradient_art import Carrier, affine

    carrier = Carrier(source, box,
                      affine(np.asarray(conditions["export"]["sface_landmarks"], np.float32)),
                      "dots_14x14")
    coefficients = torch.nn.Parameter(torch.from_numpy(
        (np.random.default_rng(0).standard_normal((3, 14, 14))*.1).astype(np.float32)))
    optimizer = torch.optim.Adam([coefficients], lr=ADAM_LR)
    logs = []
    chosen = None
    t0 = perf_counter()
    for step in range(1, STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        delta = carrier.project(coefficients)
        loss, jpeg, cosines = surrogate_loss(source, box, carrier, delta, conditions,
                                             models, clean, thresholds)
        if step in (1, 6, 12, 18):
            logs.append({"step": step, "surrogate_loss": float(loss.detach()),
                         "source_cosines": cosines,
                         "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                         "pre_jpeg_face_rms": float(torch.sqrt(torch.mean(delta.detach()[:, carrier.face]**2))),
                         "max_abs_channel_delta": float(delta.detach().abs().max())})
        if step == STEPS:
            chosen = jpeg
        loss.backward()
        gradient = coefficients.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().sum() > 0):
            raise RuntimeError(f"Missing/zero/nonfinite H15 gradient at step {step}")
        if step < STEPS:
            optimizer.step()
    return chosen, logs, perf_counter()-t0


def distortion_and_sheets(args, frozen, sources, parity):
    from PIL import Image, ImageDraw
    from fckface_lab.imaging import decode_image, export_jpeg, make_variants
    from fckface_lab.patterns import actual_distortion

    h14 = json.loads((args.h14 / "distortion.json").read_text(encoding="utf-8"))
    controls = json.loads((args.h14 / "frozen-inputs.json").read_text(encoding="utf-8"))["items"]
    result, pairing = {}, {}
    for case in parity["cases"]:
        identity, box = case["identity"], case["box"]
        source = decode_image(sources[identity][0])
        selected = next(row for row in frozen if row["identity"] == identity)
        if digest(Path(selected["path"])) != selected["sha256"]:
            raise ValueError("H15 frozen JPEG changed")
        clean = make_variants(export_jpeg(source), box)
        altered = make_variants(Path(selected["path"]).read_bytes(), box)
        result[identity], pairing[identity] = {}, {}
        for condition in CONDITIONS:
            a, b = clean[condition], altered[condition]
            if a.status != "valid" or b.status != "valid":
                result[identity][condition] = {"status": "inconclusive", "reason": a.reason or b.reason}
                continue
            x0, y0, x1, y1 = b.target_box
            result[identity][condition] = {"status": "valid", **actual_distortion(
                a.image, b.image, (x0, y0, x1-x0, y1-y0))}
        for arm in ("sg_balanced", "s0095_balanced"):
            gaps = {}
            for condition in CONDITIONS:
                current, prior = result[identity][condition], h14[identity][arm][condition]
                gaps[condition] = (abs(current["face_rms"]-prior["face_rms"])
                                   if current["status"] == prior["status"] == "valid" else None)
            pairing[identity][arm] = {"absolute_face_rms_gap_by_condition": gaps,
                "all_valid_and_within_0_25": all(value is not None and value <= .25 for value in gaps.values())}
        x0, y0, x1, y1 = box
        pad = .15*(x1-x0)
        rect = (max(0, int(x0-pad)), max(0, int(y0-pad)),
                min(source.shape[1], int(x1+pad)), min(source.shape[0], int(y1+pad)))
        tiles = [("Clean", Image.fromarray(source))]
        for arm in ("sg_balanced", "s0095_balanced"):
            row = next(row for row in controls if row["identity"] == identity and row["arm"] == arm)
            tiles.append((f"H14 {arm}", Image.open(row["path"]).convert("RGB")))
        tiles.append(("H15 SFace+ArcFace", Image.open(selected["path"]).convert("RGB")))
        for face in (False, True):
            canvas = Image.new("RGB", (450*len(tiles), 560), "#f5f5f5")
            draw = ImageDraw.Draw(canvas)
            for index, (label, image) in enumerate(tiles):
                tile = image.crop(rect) if face else image.copy()
                tile.thumbnail((438, 510), Image.Resampling.LANCZOS)
                canvas.paste(tile, (450*index+(450-tile.width)//2, 40+(510-tile.height)//2))
                draw.text((450*index+12, 12), label, fill="#111111")
            canvas.save(args.out / f"{identity}-{'face' if face else 'full'}.png")
    write_json(args.out / "distortion.json", result)
    write_json(args.out / "distortion-pairing.json", pairing)


def optimize(args) -> None:
    phase_started = perf_counter()
    from arcface_onnx_gradient import peak_bytes
    import onnx
    import torch
    from onnx2torch import convert
    from fckface_lab.imaging import decode_image, export_jpeg, make_variants

    s_cal, a_cal, sources, _ = checked_metadata(args)
    protocol_path = args.out / "protocol.json"
    parity_path = args.out / "source-only-parity.json"
    if not args.expected_source_parity_sha256 or digest(parity_path) != args.expected_source_parity_sha256:
        raise ValueError("Root-reviewed source-only ArcFace parity SHA is required")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (parity.get("status") != "pass" or parity.get("comparison_count") != 10 or
            tuple(row["identity"] for row in parity["cases"]) != IDS or
            parity["protocol_sha256"] != digest(protocol_path) or
            parity["component_diagnostic_sha256"] != digest(args.arc_component_report) or
            protocol["source_sha256"]["optimize_arcface_basis.py"] != digest(Path(__file__)) or
            protocol["source_sha256"]["h15_arcface_alignment.py"] != digest(Path(__file__).with_name("h15_arcface_alignment.py")) or
            (args.out / "frozen-inputs.json").exists()):
        raise ValueError("H15 preflight/source changed or output already frozen")
    checked_component(args.arc_component_report)
    for name, expected in protocol["model_sha256"].items():
        filename = {"yunet": "yunet.onnx", "sface": "sface.onnx",
                    "arcface": "w600k_r50.onnx", "scrfd": "det_10g.onnx"}[name]
        if digest(args.out / "model-inputs" / filename) != expected:
            raise ValueError(f"Private H15 model copy changed: {name}")
    for name, expected in protocol["source_sha256"].items():
        if digest(Path(__file__).parent / name) != expected or digest(args.out / "source-snapshot" / name) != expected:
            raise ValueError(f"H15 source dependency changed: {name}")
    cv2.setNumThreads(2); torch.set_num_threads(2)
    models = {"SFace": convert(onnx.load(str(args.out / "model-inputs" / "sface.onnx"))).eval(),
              "ArcFace": convert(onnx.load(str(args.out / "model-inputs" / "w600k_r50.onnx"))).eval()}
    for model in models.values():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    thresholds = {"SFace": s_cal["calibration"]["threshold"],
                  "ArcFace": a_cal["calibration"]["threshold"]}
    frozen, rows = [], []
    started = perf_counter()
    try:
        for case in parity["cases"]:
            identity = case["identity"]
            source_path, source_sha = sources[identity]
            if digest(source_path) != source_sha or case["source_sha256"] != source_sha:
                raise ValueError(f"H15 source changed: {identity}")
            source = decode_image(source_path)
            box = case["box"]
            variants = make_variants(export_jpeg(source), box)
            for condition in OBJECTIVE:
                if hashlib.sha256(variants[condition].jpeg).hexdigest() != case["conditions"][condition]["clean_jpeg_sha256"]:
                    raise ValueError("Frozen clean JPEG changed")
            clean = {"SFace": case["sface_source_feature"],
                     "ArcFace": case["arc_source_feature"]}
            jpeg, logs, seconds = optimize_one(source, box, case["conditions"],
                                                models, clean, thresholds)
            directory = args.out / identity
            directory.mkdir(exist_ok=False)
            target = directory / "selected.jpg"
            target.write_bytes(jpeg)
            write_json(directory / "surrogate-progress.json", logs)
            record = {"identity": identity, "method": "H15_arcface_basis",
                      "arm": "sarc_balanced", "path": str(target), "sha256": digest(target),
                      "source_view": "neutral_front"}
            frozen.append(record)
            rows.append({"identity": identity, "selected_sha256": record["sha256"],
                         "forward_step": 18, "effective_updates": 17,
                         "native_checkpoint_queries": 0, "seconds_excluding_load": seconds,
                         "pre_jpeg_face_rms": logs[-1]["pre_jpeg_face_rms"]})
            write_json(args.out / "freeze-progress.json", {"frozen": frozen, "cases": rows})
        if len(frozen) != 2:
            raise AssertionError("Two H15 exports not frozen")
        write_json(args.out / "frozen-inputs.json", {"schema_version": 1, "items": frozen})
        distortion_and_sheets(args, frozen, sources, parity)
        write_json(args.out / "freeze-results.json", {
            "status": "two_exports_frozen_before_gallery", "protocol_sha256": digest(protocol_path),
            "source_parity_sha256": digest(parity_path),
            "frozen_manifest_sha256": digest(args.out / "frozen-inputs.json"),
            "optimization_seconds_excluding_load": perf_counter()-started,
            "phase_seconds_including_model_load": perf_counter()-phase_started,
            "peak_working_set_bytes": peak_bytes(),
            "postflight": end_integrity(args, protocol, sources), "cases": rows})
    except Exception as exc:
        write_json(args.out / "failure.json", {"phase": "optimize", "type": type(exc).__name__,
                                              "message": str(exc), "frozen_count": len(frozen)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "optimize"))
    for name in ("manifest", "sface-calibration", "arc-calibration", "arc-dir",
                 "arc-component-report", "h14", "yunet", "sface", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-source-parity-sha256",
                        help="Required only for optimize, after review of preflight result")
    args = parser.parse_args()
    {"preflight": preflight, "optimize": optimize}[args.phase](args)
