"""H13 fixed substrate ablation: optimize common dots on original or H12 chimera.

Development only. ``--preflight`` reads frozen H12 geometry/source and performs
no recognition inference. The optimizer intentionally stops after four JPEGs
freeze; gallery scoring requires a separate, later action after visual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

IDS = ("frll-032", "frll-037")
ARMS = ("stack", "joint")
STEPS = 18
OBJECTIVE = ("export", "jpeg75_420", "blur", "crop90")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960", "half_restore", "crop90", "blur")
RMS_INCREMENT = 16.0
RMS_TOLERANCE = .25
MAX_SCALE = 65536.0
BISECTION_STEPS = 16
ADAM_LR = .15


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def frozen_h12_cases(manifest_path: Path, frozen_path: Path):
    """Use only frozen H12 clean geometry, with exact source/JPEG integrity."""
    from fckface_lab.imaging import decode_image, export_jpeg
    from fckface_lab.patterns import actual_distortion
    from render_contrast_chimera import masks, render

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = json.loads(frozen_path.read_text(encoding="utf-8"))
    if prior["source_sha256"] != sha(Path(__file__).with_name("render_contrast_chimera.py")):
        raise ValueError("Frozen H12 renderer source changed")
    if prior["protocol_sha256"] != sha(frozen_path.with_name("protocol.json")):
        raise ValueError("Frozen H12 protocol changed")
    if sha(manifest_path) != json.loads(frozen_path.with_name("protocol.json").read_text())["manifest_sha256"]:
        raise ValueError("Frozen H12 manifest changed")
    if tuple(row["identity"] for row in prior["cases"]) != IDS:
        raise ValueError("Frozen H12 IDs/order changed")
    cases = []
    for row in prior["cases"]:
        identity = row["identity"]
        matches = [item for item in manifest["images"] if item["identity"] == identity
                   and item["view"] == "neutral_front" and item["split"] == "development"]
        if len(matches) != 1:
            raise ValueError(f"Missing unique development source: {identity}")
        item = matches[0]
        path = (manifest_path.parent / item["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Source escaped dataset root")
        if sha(path) != item["sha256"] or sha(path) != row["source_sha256"]:
            raise ValueError(f"H12 source SHA mismatch: {identity}")
        source = decode_image(path)
        face = np.array(row["clean_yunet_box_xywh"] + row["clean_yunet_landmarks"] + [1.], np.float32)
        face_support, hard_support, xywh = masks(source, face)
        chimera, _ = render(source, hard_support, "negative")
        expected = row["arms"]["eye_positive_chimera"]["sha256"]
        if sha(Path(row["arms"]["eye_positive_chimera"]["path"])) != expected:
            raise ValueError(f"Saved H12 base JPEG changed: {identity}")
        if hashlib.sha256(export_jpeg(chimera)).hexdigest() != expected:
            raise ValueError(f"H12 clean-landmark replay differs: {identity}")
        clean_export = decode_image(export_jpeg(source))
        base_rms = actual_distortion(clean_export, decode_image(export_jpeg(chimera)), xywh)["face_rms"]
        base_prejpeg_rms = actual_distortion(source, chimera, xywh)["face_rms"]
        if abs(base_rms - row["chimera_metrics"]["face_rms"]) > 1e-5:
            raise ValueError("H12 RMS replay differs")
        box = (xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3])
        cases.append(dict(identity=identity, path=path, source=source, chimera=chimera,
                          hard_support=hard_support, face_support=face_support,
                          box=box, xywh=xywh, landmarks=face[4:14].reshape(5, 2),
                          clean_export=clean_export, base_rms=base_rms,
                          base_prejpeg_rms=base_prejpeg_rms,
                          training_target_rms=float(np.hypot(base_prejpeg_rms,RMS_INCREMENT)),
                          target_rms=float(np.hypot(base_rms, RMS_INCREMENT)),
                          h12_sha256=expected))
    return cases


class SharedCarrier:
    """H5 literal cells/dots, gated by H12 hard treatment support.

    ``project`` is the same function for both arms; ``source`` changes only
    the substrate JPEG supplied to the H10 surrogate. The scalar is detached
    (BPDA), while gradients pass through cell values, feathering and clamp.
    """

    def __init__(self, case: dict, matrix: np.ndarray, substrate: str):
        import torch
        from optimize_gradient_art import Carrier

        self._carrier = Carrier(case["source"], case["box"], matrix, "dots_14x14")
        c = self._carrier
        self.left, self.top, self.right, self.bottom = c.left, c.top, c.right, c.bottom
        self.roi, self.face, self.cell_index, self.grid = c.roi, c.face, c.cell_index, c.grid
        self.mask = c.mask
        self.dot_mask = c.dot_mask * torch.from_numpy(
            case["hard_support"][self.top:self.bottom, self.left:self.right].astype(np.float32)
        ).unsqueeze(0)
        self.original = case["source"]
        self.chimera = case["chimera"]
        self.source = case["source"] if substrate == "stack" else case["chimera"]
        self.base = torch.from_numpy(self.source[self.top:self.bottom, self.left:self.right]
                                     .astype(np.float32).transpose(2, 0, 1).copy())
        original = torch.from_numpy(self.original[self.top:self.bottom, self.left:self.right]
                                    .astype(np.float32).transpose(2, 0, 1).copy())
        chimera = torch.from_numpy(self.chimera[self.top:self.bottom, self.left:self.right]
                                   .astype(np.float32).transpose(2, 0, 1).copy())
        self.low = torch.maximum(torch.maximum(-original, -chimera), torch.tensor(-64.))
        self.high = torch.minimum(torch.minimum(255.-original, 255.-chimera), torch.tensor(64.))
        if not bool((self.low <= self.high).all()):
            raise ValueError("Common headroom is empty")
        self.case = case
        self.last_projection = None
        from fckface_lab.patterns import face_mask
        self.full_face = face_mask(case["source"].shape,case["xywh"])>0
        self.local_face = self.full_face[self.top:self.bottom,self.left:self.right]
        error = (self.chimera.astype(np.float32)-self.original.astype(np.float32))
        local_error = error[self.top:self.bottom,self.left:self.right]
        self.error_local = torch.from_numpy(local_error.transpose(2,0,1).copy())
        self.base_error_ss = float(np.sum(error[self.full_face]**2,dtype=np.float64))
        self.local_error_ss = float(np.sum(local_error[self.local_face]**2,dtype=np.float64))
        self.face_denominator = int(self.full_face.sum())*3
        if abs(np.sqrt(self.base_error_ss/self.face_denominator)-case["base_prejpeg_rms"])>.0002:
            raise ValueError("Pre-JPEG RMS support differs from actual_distortion")

    def field(self, coefficients):
        return (coefficients.reshape(3, -1)[:, self.cell_index].reshape_as(self.base)
                * self.dot_mask * self.mask)

    def bounded(self, raw, scale: float):
        import torch
        return torch.minimum(torch.maximum(raw * scale, self.low), self.high)

    def render(self, delta):
        image = self.source.copy()
        patch = (self.base + delta.detach()).permute(1, 2, 0).numpy()
        image[self.top:self.bottom, self.left:self.right] = np.rint(patch).clip(0, 255).astype(np.uint8)
        return image

    def final_image(self, delta):
        image = self.chimera.copy()
        patch = (self.chimera[self.top:self.bottom, self.left:self.right].astype(np.float32)
                 + delta.detach().permute(1, 2, 0).numpy())
        image[self.top:self.bottom, self.left:self.right] = np.rint(patch).clip(0, 255).astype(np.uint8)
        return image

    def achieved_jpeg(self, raw, scale: float) -> float:
        from fckface_lab.imaging import decode_image, export_jpeg
        from fckface_lab.patterns import actual_distortion
        delta = self.bounded(raw, scale)
        jpeg = export_jpeg(self.final_image(delta))
        return actual_distortion(self.case["clean_export"], decode_image(jpeg),
                                 self.case["xywh"])["face_rms"]

    def achieved_prejpeg(self, raw, scale: float) -> float:
        delta = self.bounded(raw,scale)
        error = self.error_local+delta
        changed_ss = float((error[:,self.local_face]**2).double().sum())
        return float(np.sqrt((self.base_error_ss-self.local_error_ss+changed_ss)
                             /self.face_denominator))

    def _fit_scale(self, raw, target:float, exact_jpeg:bool, tolerance:float):
        """Deterministic first observed tolerance crossing and scalar bisection."""
        import torch
        with torch.no_grad():
            started = perf_counter()
            samples = []
            def measure(scale):
                value = self.achieved_jpeg(raw,scale) if exact_jpeg else self.achieved_prejpeg(raw,scale)
                samples.append((scale,value))
                return value
            threshold=target-tolerance
            zero = measure(0.)
            if zero >= threshold:
                if abs(zero-target)>tolerance:
                    raise ValueError("Base already exceeds scalar target")
                scale,found=0.,zero
            else:
                lo,lo_rms,hi = 0.,zero,1.
                bracket=False
                while hi <= MAX_SCALE:
                    hi_rms=measure(hi)
                    if hi_rms>=threshold:
                        bracket=True
                        break
                    lo,lo_rms,hi=hi,hi_rms,hi*2
                if not bracket:
                    raise ValueError(f"{'JPEG' if exact_jpeg else 'pre-JPEG'} RMS target {target:.4f} unattainable by scale {MAX_SCALE:g}")
                for _ in range(24 if exact_jpeg else 18):
                    mid=(lo+hi)/2
                    mid_rms=measure(mid)
                    if mid_rms>=threshold:
                        hi,hi_rms=mid,mid_rms
                    else:
                        lo,lo_rms=mid,mid_rms
                feasible=[(s,r) for s,r in samples if abs(r-target)<=tolerance]
                if not feasible:
                    raise ValueError(f"Quantized RMS mismatch; no sample within {tolerance:g} of {target:.4f}")
                scale,found=min(feasible,key=lambda pair:pair[0])
            return {"scale":scale,"face_rms":found,"target_rms":target,
                    "residual_rms":found-target,"evaluations":len(samples),
                    "seconds":perf_counter()-started,
                    "observed_first_crossing_only":True,
                    "metric":"decoded JPEG vs clean decoded JPEG" if exact_jpeg else "float pre-JPEG vs original RGB"}

    def project(self, coefficients):
        """Fast common pre-JPEG scalar fit; no model score or arm enters it."""
        raw = self.field(coefficients)
        self.last_projection=self._fit_scale(raw.detach(),self.case["training_target_rms"],False,.02)
        return self.bounded(raw,self.last_projection["scale"])

    def final_correct(self,coefficients):
        """One no-gradient exact-JPEG scalar correction of frozen coefficients."""
        raw=self.field(coefficients).detach()
        fit=self._fit_scale(raw,self.case["target_rms"],True,.02)
        delta=self.bounded(raw,fit["scale"])
        fitted=self.final_image(delta)
        if np.any(fitted[~self.case["hard_support"]]!=self.original[~self.case["hard_support"]]):
            raise AssertionError("Final correction escaped hard treatment support")
        fit["training_scale"]=self.last_projection["scale"]
        fit["scale_change"]=fit["scale"]-self.last_projection["scale"]
        fit["saturated_channel_values"]=int(((raw*fit["scale"]<self.low)|
                                             (raw*fit["scale"]>self.high)).sum())
        fit["max_abs_channel_delta"]=float(delta.abs().max())
        return delta,fit


def preflight(cases: list[dict]) -> dict:
    """Model-free exact-byte, support, common-bounds and JPEG timing check."""
    import torch
    from optimize_gradient_art import affine
    torch.set_num_threads(2)
    rows = {}
    for case in cases:
        carrier = SharedCarrier(case, affine(case["landmarks"]), "stack")
        twin = SharedCarrier(case, affine(case["landmarks"]), "joint")
        rng = np.random.default_rng(0)
        param = torch.from_numpy((rng.standard_normal((3, 14, 14))*.1).astype(np.float32))
        if not torch.equal(carrier.dot_mask, twin.dot_mask):
            raise AssertionError("Arm supports differ")
        raw = carrier.field(param)
        if not torch.equal(raw, twin.field(param)):
            raise AssertionError("Common coefficient field differs")
        if not torch.equal(carrier.low, twin.low) or not torch.equal(carrier.high, twin.high):
            raise AssertionError("Common headroom differs")
        restricted = np.zeros(case["source"].shape[:2], bool)
        restricted[carrier.top:carrier.bottom,carrier.left:carrier.right] = (
            carrier.dot_mask[0].numpy() > 0)
        if np.any(restricted & ~case["hard_support"]):
            raise AssertionError("Dots escaped H12 treatment support")
        result = {"h12_base_sha256":case["h12_sha256"],"base_rms":case["base_rms"],
                  "target_rms":case["target_rms"],
                  "base_prejpeg_rms":case["base_prejpeg_rms"],
                  "training_target_rms":case["training_target_rms"],
                  "dot_pixels":int(np.count_nonzero(restricted)),
                  "hard_support_pixels":int(case["hard_support"].sum()),
                  "same_field_and_bounds":True}
        try:
            delta = carrier.project(param)
            final = carrier.final_image(delta)
            if np.any(final[~case["hard_support"]] != case["source"][~case["hard_support"]]):
                raise AssertionError("Final pre-JPEG image modified eye/outside treatment support")
            if not np.array_equal(final,twin.final_image(twin.bounded(raw,carrier.last_projection["scale"]))):
                raise AssertionError("Final feasible JPEG differs across arm substrates")
            result["initial_projection"] = carrier.last_projection
            differentiable=param.clone().requires_grad_(True)
            proxy=carrier.bounded(carrier.field(differentiable),carrier.last_projection["scale"])
            weights=torch.linspace(-1.,1.,proxy.numel()).reshape_as(proxy)
            (proxy*weights).sum().backward()
            if differentiable.grad is None or not bool(torch.isfinite(differentiable.grad).all()) or not bool(differentiable.grad.abs().sum()>0):
                raise AssertionError("Shared bounded carrier lacks a finite gradient")
            result["bounded_field_gradient_finite_nonzero"]=True
            corrected,fit=carrier.final_correct(param)
            result["initial_final_correction"]=fit
            if np.any(carrier.final_image(corrected)[~case["hard_support"]] != case["source"][~case["hard_support"]]):
                raise AssertionError("Final corrected field escaped H12 support")
        except ValueError as exc:
            result["initial_projection_invalid"] = str(exc)
        rows[case["identity"]] = result
    return rows


def verify_models(args):
    """Pin the exact development calibration/pipeline before model loading."""
    from fckface_lab.calibration import Calibration
    from fckface_lab.datasets import digest
    calibrations = {
        "SFace": json.loads(args.sface_calibration.read_text(encoding="utf-8")),
        "GhostFaceNet": json.loads(args.ghostface_calibration.read_text(encoding="utf-8")),
    }
    paths = {"yunet": args.yunet, "sface": args.sface, "ghostface": args.ghostface}
    for name, artifact in calibrations.items():
        expected = ("yunet", "sface") if name == "SFace" else ("yunet", "ghostface")
        if digest(args.manifest) != artifact["dataset"]["manifest_sha256"]:
            raise ValueError(f"{name} calibration manifest differs")
        for model in expected:
            record = artifact["pipeline"]["model_artifacts"][model]
            if sha(paths[model]) != record["sha256"] or paths[model].stat().st_size != record["bytes"]:
                raise ValueError(f"Calibrated artifact mismatch: {name}/{model}")
        for source, expected_sha in artifact["pipeline"]["source_sha256"].items():
            if source.startswith("fckface_lab/") and sha(Path(__file__).parent/source) != expected_sha:
                raise ValueError(f"Calibrated pipeline mismatch: {source}")
    diagnostic = json.loads(args.clone_diagnostic.read_text(encoding="utf-8"))
    if (not diagnostic["gradient_finite"] or
        any(row["unit_cosine"] < .999 or row["unit_max_abs_difference"] > .02
            for row in diagnostic["parity"].values()) or
        not any(row["relative_error"] <= .1 or row["absolute_error"] <= 1e-6
                for row in diagnostic["unit_l2_finite_differences"])):
        raise ValueError("Frozen Ghost float32 clone diagnostic did not pass")
    thresholds = {name: Calibration(**cal["calibration"]).threshold
                  for name, cal in calibrations.items()}
    return paths, calibrations, thresholds


def optimize_arm(case, arm, points, torch_sface, ghost_gradient,
                 native_clean, thresholds, output: Path):
    import torch
    from fckface_lab.imaging import export_jpeg
    from optimize_gradient_art import affine
    from optimize_transfer_dots import surrogate_loss

    carrier = SharedCarrier(case, affine(points["export"]), arm)
    rng = np.random.default_rng(0)
    coefficients = torch.nn.Parameter(torch.from_numpy(
        (rng.standard_normal((3, 14, 14))*.1).astype(np.float32)))
    optimizer = torch.optim.Adam([coefficients], lr=ADAM_LR)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    checkpoints = []
    selected = None
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = carrier.project(coefficients)
        loss, substrate_jpeg, scores = surrogate_loss(
            carrier.source, case["box"], carrier, delta, points, torch_sface,
            ghost_gradient, native_clean, thresholds, {}, "adam")
        final_fit = None
        if step == STEPS:
            delta,final_fit=carrier.final_correct(coefficients)
        final_image = carrier.final_image(delta)
        if np.any(final_image[~case["hard_support"]] != case["source"][~case["hard_support"]]):
            raise AssertionError("Final pre-JPEG image changed original eye/outside")
        final_jpeg = export_jpeg(final_image)
        if step % 3 == 0:
            row = {"step": step, "arm": arm, "selected_only_if_step18": step == 18,
                   "substrate_jpeg_sha256": hashlib.sha256(substrate_jpeg).hexdigest(),
                   "final_jpeg_sha256": hashlib.sha256(final_jpeg).hexdigest(),
                   "projection": carrier.last_projection,
                   "final_jpeg_scalar_correction":final_fit,
                   "surrogate_worst_normalized_margin": float(loss.detach()),
                   "surrogate_source_cosines": scores}
            checkpoints.append(row)
            (output/f"step{step:02d}.jpg").write_bytes(final_jpeg)
            write_json(output/f"step{step:02d}.json", row)
            if step == STEPS:
                selected = {"jpeg": final_jpeg, "row": row}
        loss.backward()
        gradient = coefficients.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().sum() > 0):
            raise RuntimeError(f"Missing/zero/nonfinite coefficient gradient at step {step}")
        if step < STEPS:
            optimizer.step()
    if selected is None:
        raise AssertionError("No unconditional forward-step18 export")
    selected["seconds"] = perf_counter()-started
    write_json(output/"checkpoints.json", checkpoints)
    return selected


def distortion_report(cases, frozen, destination: Path):
    """Measure all seven exact JPEG conditions, with no model/gallery reads."""
    from fckface_lab.imaging import decode_image, export_jpeg, make_variants
    from fckface_lab.patterns import actual_distortion
    report = {}
    for case in cases:
        clean = make_variants(export_jpeg(case["source"]), case["box"])
        person = {}
        for arm in ARMS:
            selected = next(row for row in frozen if row["identity"] == case["identity"] and row["arm"] == arm)
            jpeg = Path(selected["path"]).read_bytes()
            if hashlib.sha256(jpeg).hexdigest() != selected["sha256"]:
                raise ValueError("Frozen JPEG changed before distortion analysis")
            variants = make_variants(jpeg, case["box"])
            metrics = {}
            for condition in CONDITIONS:
                a, b = clean[condition], variants[condition]
                if a.status != "valid" or b.status != "valid":
                    metrics[condition] = {"status": "inconclusive", "reason": a.reason or b.reason}
                    continue
                x0, y0, x1, y1 = b.target_box
                metrics[condition] = {"status": "valid", **actual_distortion(
                    a.image, b.image, (x0,y0,x1-x0,y1-y0))}
            person[arm] = metrics
        report[case["identity"]] = person
    write_json(destination, report)


def contact_sheets(cases, frozen, output: Path):
    from PIL import Image, ImageDraw
    for case in cases:
        items = [("Clean", Image.fromarray(case["source"])),
                 ("H12 base", Image.fromarray(case["chimera"]))]
        for arm in ARMS:
            path = next(Path(row["path"]) for row in frozen if row["identity"] == case["identity"] and row["arm"] == arm)
            items.append((arm, Image.open(path).convert("RGB")))
        for crop in (False, True):
            canvas = Image.new("RGB", (4*450, 560), "#f5f5f5")
            draw = ImageDraw.Draw(canvas)
            x0,y0,x1,y1 = case["box"]
            pad = .15*(x1-x0)
            rect = (max(0,int(x0-pad)), max(0,int(y0-pad)),
                    min(case["source"].shape[1],int(x1+pad)),
                    min(case["source"].shape[0],int(y1+pad)))
            for index,(name,img) in enumerate(items):
                tile = img.crop(rect) if crop else img.copy()
                tile.thumbnail((438,510),Image.Resampling.LANCZOS)
                canvas.paste(tile,(450*index+(450-tile.width)//2,40+(510-tile.height)//2))
                draw.text((450*index+12,12),name,fill="#111111")
            canvas.save(output/f"{case['identity']}-{'face' if crop else 'full'}.png")


def run(args):
    """Frozen source-only optimization; this function intentionally never reads gallery views."""
    import cv2
    import onnx
    import torch
    from onnx2torch import convert
    from fckface_lab.datasets import digest
    from fckface_lab.imaging import export_jpeg, make_variants
    from fckface_lab.recognition import SFaceModel
    from fckface_lab.ghostface import GhostFaceModel
    from optimize_alignment_dots import detect_points, tf_value_gradient_functions
    from optimize_ensemble_dots import load_float32_clone

    repo = Path(__file__).resolve().parents[1]
    if args.out.exists() or args.out.resolve().is_relative_to(repo):
        raise ValueError("Use a new external output directory")
    cases = frozen_h12_cases(args.manifest,args.h12_frozen)
    model_paths, calibrations, thresholds = verify_models(args)
    cv2.setNumThreads(2); torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=False)
    snapshot = args.out/"source-snapshot"; snapshot.mkdir()
    source_hashes = {}
    for name in ("optimize_chimera_dots.py","render_contrast_chimera.py",
                 "optimize_transfer_dots.py","optimize_alignment_dots.py",
                 "optimize_gradient_art.py","optimize_ensemble_dots.py"):
        path = Path(__file__).parent/name
        shutil.copyfile(path,snapshot/name)
        source_hashes[name] = sha(path)
        if sha(snapshot/name) != source_hashes[name]:
            raise ValueError(f"Source changed while archiving: {name}")
    package_snapshot=snapshot/"fckface_lab";package_snapshot.mkdir()
    for path in sorted((Path(__file__).parent/"fckface_lab").glob("*.py")):
        shutil.copyfile(path,package_snapshot/path.name)
        source_hashes[f"fckface_lab/{path.name}"]=sha(path)
        if sha(package_snapshot/path.name)!=source_hashes[f"fckface_lab/{path.name}"]:
            raise ValueError(f"Pipeline source changed while archiving: {path.name}")
    pinned_dir = args.out/"model-inputs"; pinned_dir.mkdir()
    pinned = {}
    for name,path in model_paths.items():
        target = pinned_dir/path.name
        shutil.copyfile(path,target)
        reference = "GhostFaceNet" if name == "ghostface" else "SFace"
        expected = calibrations[reference]["pipeline"]["model_artifacts"][name]
        if sha(target) != expected["sha256"] or target.stat().st_size != expected["bytes"]:
            raise ValueError(f"Private model copy differs from frozen calibration: {name}")
        pinned[name] = target
    protocol = {"hypothesis":"H13 substrate-aware versus stacked dot optimization",
        "scope":"development only, IDs 032/037; no independent test or privacy claim",
        "identities":IDS,"arms":ARMS,"steps":STEPS,"effective_updates":STEPS-1,
        "objective_conditions":OBJECTIVE,"final_conditions":CONDITIONS,
        "gradient_model_condition_forwards_per_arm":STEPS*len(OBJECTIVE)*2,
        "native_diagnostic_queries_per_arm":0,
        "progress":"surrogate source values and projection metadata every 3 steps; no native checkpoint inference",
        "selection":"unconditional final chimera+field at forward step18; progress never selects",
        "seed":0,"adam_lr":ADAM_LR,"coefficients":588,"cap":64,
        "support":"H5 literal 14x14 radius .3 dots intersect H12 hard treatment support",
        "shared_headroom":"per-channel low=max(-64,-original,-chimera), high=min(64,255-original,255-chimera)",
        "training_target":"sqrt(H12 pre-JPEG chimera-versus-original face RMS squared + 16 squared), same support; detached bounded scalar projection",
        "final_target":"sqrt(H12 decoded Q95 base-versus-clean Q95 face RMS squared + 16 squared), final scalar-only Q95 correction tolerance .02; per-condition pairing gate .25",
        "projector":"same common bounded field, geometric first observed crossing and bisection; pre-JPEG during gradient, exact Q95 once at step18; no model-based selection",
        "substrate":{"stack":"original+shared field in surrogate", "joint":"chimera+shared field in surrogate"},
        "final_export":"chimera+shared field in both arms",
        "gradient":"H10 fixed clean-condition landmarks and two-model max normalized margin; exact JPEG forward, fixed-grid BPDA backward",
        "scientific_gate":"both arms plausible, all 7 detections valid and paired processed face RMS within .25; joint ArcFace worst-gallery cosine >=.05 lower than both stack and unchanged H12 base for each ID; joint SFace/Ghost all7 nonmatch each",
        "h12_frozen_sha256":sha(args.h12_frozen),"manifest_sha256":sha(args.manifest),
        "calibration_sha256":{"SFace":sha(args.sface_calibration),"GhostFaceNet":sha(args.ghostface_calibration)},
        "clone_diagnostic_sha256":sha(args.clone_diagnostic),
        "source_sha256":source_hashes,"model_sha256":{name:sha(path) for name,path in pinned.items()},
        "thresholds":thresholds,
        "cases":{c["identity"]:{"h12_base_sha256":c["h12_sha256"],"base_rms":c["base_rms"],
                  "target_rms":c["target_rms"],"base_prejpeg_rms":c["base_prejpeg_rms"],
                  "training_target_rms":c["training_target_rms"],
                  "source_sha256":sha(c["path"]),
                  "frozen_clean_landmarks":c["landmarks"].tolist(),"frozen_clean_box_xyxy":c["box"]}
                 for c in cases}}
    write_json(args.out/"protocol.json",protocol)
    preflight_report=preflight(cases)
    write_json(args.out/"model-free-preflight.json",preflight_report)
    if any("initial_projection_invalid" in row for row in preflight_report.values()):
        raise ValueError("Initial shared field could not reach the fixed RMS target")
    models = {"SFace":SFaceModel(pinned["yunet"],pinned["sface"]),
              "GhostFaceNet":GhostFaceModel(pinned["yunet"],pinned["ghostface"])}
    clone = load_float32_clone(models["GhostFaceNet"].model)
    ghost_gradient, tf_report = tf_value_gradient_functions(clone)
    write_json(args.out/"ghost-gradient-preflight.json",tf_report)
    if not tf_report["compiled_accepted"]:
        raise RuntimeError("Compiled Ghost gradient parity failed")
    torch_sface = convert(onnx.load(str(pinned["sface"]))).eval()
    for parameter in torch_sface.parameters(): parameter.requires_grad_(False)
    frozen = []
    results = []
    try:
        for case in cases:
            source = case["source"]
            native = {name:model.embed(source) for name,model in models.items()}
            if any(row.status != "valid" for row in native.values()):
                raise RuntimeError(f"Invalid source: {case['identity']}")
            if any(np.max(np.abs(np.asarray(row.selected_box)-case["box"])) > .01
                   for row in native.values()):
                raise ValueError("Native clean box differs from frozen H12 box")
            native_clean = {name:row.feature for name,row in native.items()}
            variants = make_variants(export_jpeg(source),case["box"])
            points = {name:detect_points(models["SFace"].detector,variants[name]) for name in OBJECTIVE}
            person = {"identity":case["identity"],"source_sha256":sha(case["path"]),
                      "clean_points_by_condition":{k:v.tolist() for k,v in points.items()},"arms":{}}
            for arm in ARMS:
                directory = args.out/case["identity"]/arm
                selected = optimize_arm(case,arm,points,torch_sface,
                                        ghost_gradient,native_clean,thresholds,directory/"checkpoints")
                path = directory/"selected.jpg"; path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(selected["jpeg"])
                frozen.append({"identity":case["identity"],"method":"H13", "arm":arm,
                               "path":str(path),"sha256":sha(path),"source_view":"neutral_front"})
                person["arms"][arm] = {"path":str(path),"sha256":sha(path),
                    "forward_step":18,"effective_updates":17,"seconds":selected["seconds"],
                    "training_projection":selected["row"]["projection"],
                    "final_jpeg_scalar_correction":selected["row"]["final_jpeg_scalar_correction"]}
                write_json(args.out/"freeze-progress.json",{"cases":results+[person],"frozen":frozen})
            results.append(person)
        if len(frozen) != 4:
            raise AssertionError("All four exports must freeze before gallery access")
        write_json(args.out/"frozen-inputs.json",{"schema_version":1,"items":frozen})
        write_json(args.out/"freeze-results.json",{"status":"four_exports_frozen_before_gallery",
            "protocol_sha256":sha(args.out/"protocol.json"),"cases":results,
            "frozen_manifest_sha256":sha(args.out/"frozen-inputs.json")})
        distortion_report(cases,frozen,args.out/"distortion.json")
        contact_sheets(cases,frozen,args.out)
    except Exception as exc:
        write_json(args.out/"failure.json",{"type":type(exc).__name__,"message":str(exc),
                                              "frozen_count":len(frozen)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--h12-frozen", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="No recognition inference")
    for name in ("sface_calibration","ghostface_calibration","yunet","sface",
                 "ghostface","clone_diagnostic","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(frozen_h12_cases(args.manifest,args.h12_frozen)), indent=2, allow_nan=False))
    else:
        required=("sface_calibration","ghostface_calibration","yunet","sface",
                  "ghostface","clone_diagnostic","out")
        missing=[name for name in required if getattr(args,name) is None]
        if missing: parser.error("Missing: "+", ".join(missing))
        run(args)
