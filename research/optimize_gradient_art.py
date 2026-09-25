"""H5 fixed development-identity, source-only SFace gradient experiment.

This is development research, not a validated privacy processor. Selected
JPEGs are frozen before any gallery reference is computed. Exact JPEG/blur
pixels are used in the forward objective. The fixed-affine identity BPDA does
not propagate the true rounding, JPEG, or blur derivative. Every eighth step
is selected by a fresh native detector/alignment/source-score check, not by
the differentiable surrogate. The dot carrier is a fixed circular 14x14 grid;
its individual RGB cell weights are optimized, with no geometry edits.
"""
from __future__ import annotations

import argparse

import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx2torch import convert

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.evaluation import Reference, evaluate_gallery
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.patterns import face_mask, actual_distortion
from fckface_lab.recognition import SFaceModel

DATA = Path.home() / "Downloads" / "FCKFACE-data"
WEIGHTS = Path.home() / "Downloads" / "Face Privacy Filter" / "cache" / ".deepface" / "weights"
OUT = DATA / "runs" / "gradient-art-v2"
FROZEN_IDS = ("frll-001", "frll-003", "frll-004", "frll-007",
              "frll-011", "frll-013", "frll-018", "frll-019")
IDS = FROZEN_IDS[:2]
VIEWS = ("neutral_front", "smiling_front", "neutral_left_3quarter", "neutral_right_3quarter")
OBJECTIVE = ("export", "jpeg75_420", "blur")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960", "half_restore", "crop90", "blur")
ARMS = ("unrestricted_rgb", "smooth_14x14", "dots_14x14")
STEPS, RMS, MAX_DELTA = 48, 16.0, 64.0
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_input_images(manifest: dict, groups: dict, identities: tuple[str, ...]) -> None:
    expected = {(row["identity"], row["view"]): row["sha256"]
                for row in manifest["images"]
                if row["identity"] in identities and row["view"] in VIEWS}
    for identity in identities:
        for view in VIEWS:
            if ((identity, view) not in expected or view not in groups[identity]
                    or digest(groups[identity][view]) != expected[(identity, view)]):
                raise ValueError(f"FRLL image differs from manifest: {identity}:{view}")


def affine(landmarks: np.ndarray) -> np.ndarray:
    """OpenCV FaceRecognizerSF five-point similarity, source to 112 target."""
    source = np.asarray(landmarks, np.float64).reshape(5, 2)
    a, b = source - source.mean(0), TEMPLATE - TEMPLATE.mean(0)
    dot = float(np.sum(a * b))
    cross = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    den = float(np.sum(a * a))
    if den <= 0:
        raise ValueError("Degenerate landmarks")
    c, s = dot / den, cross / den
    tx = TEMPLATE[:, 0].mean() - c * source[:, 0].mean() + s * source[:, 1].mean()
    ty = TEMPLATE[:, 1].mean() - s * source[:, 0].mean() - c * source[:, 1].mean()
    return np.array([[c, -s, tx], [s, c, ty]], np.float64)


def aligned_rgb(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    crop = cv2.warpAffine(bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def raw_feature(model, rgb_crop: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(rgb_crop.transpose(2, 0, 1)[None].astype(np.float32))
    return model(torch.from_numpy(rgb)).reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).reshape(-1), np.asarray(b, np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Carrier:
    def __init__(self, source: np.ndarray, box, matrix: np.ndarray, arm: str):
        self.source, self.box, self.arm = source, box, arm
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        self.left = max(0, int(np.floor(x0 - .1 * bw)))
        self.top = max(0, int(np.floor(y0 - .1 * bh)))
        self.right = min(source.shape[1], int(np.ceil(x1 + .1 * bw)))
        self.bottom = min(source.shape[0], int(np.ceil(y1 + .1 * bh)))
        self.roi = source[self.top:self.bottom, self.left:self.right]
        h, w = self.roi.shape[:2]
        local = (x0-self.left, y0-self.top, bw, bh)
        mask = face_mask(self.roi.shape, local)
        self.face = torch.from_numpy(mask > 0)
        self.mask = torch.from_numpy(mask).unsqueeze(0)
        self.base = torch.from_numpy(self.roi.astype(np.float32).transpose(2, 0, 1).copy())
        self.low = torch.maximum(-self.base, torch.tensor(-MAX_DELTA))
        self.high = torch.minimum(255-self.base, torch.tensor(MAX_DELTA))
        if arm == "dots_14x14":
            yy, xx = np.mgrid[:h, :w]
            u, v = (xx + self.left - x0) / bw * 14, (yy + self.top - y0) / bh * 14
            cx, cy = np.floor(u).astype(int), np.floor(v).astype(int)
            inside = (cx >= 0) & (cx < 14) & (cy >= 0) & (cy < 14)
            dots = inside & ((u-cx-.5)**2 + (v-cy-.5)**2 <= .3**2)
            self.dot_mask = torch.from_numpy(dots.astype(np.float32)).unsqueeze(0)
            self.cell_index = torch.from_numpy((np.clip(cy,0,13)*14+np.clip(cx,0,13)).reshape(-1).astype(np.int64))
        inverse = cv2.invertAffineTransform(matrix)
        ys, xs = np.mgrid[:112, :112].astype(np.float64)
        source_x = inverse[0,0]*xs + inverse[0,1]*ys + inverse[0,2] - self.left
        source_y = inverse[1,0]*xs + inverse[1,1]*ys + inverse[1,2] - self.top
        grid = np.stack((2*source_x/(w-1)-1, 2*source_y/(h-1)-1), axis=-1).astype(np.float32)
        self.grid = torch.from_numpy(grid).unsqueeze(0)

    def field(self, param: torch.Tensor) -> torch.Tensor:
        if self.arm == "unrestricted_rgb":
            raw = param
        elif self.arm == "smooth_14x14":
            raw = F.interpolate(param.unsqueeze(0), size=self.roi.shape[:2], mode="bilinear", align_corners=True)[0]
        else:
            raw = param.reshape(3, -1)[:, self.cell_index].reshape_as(self.base) * self.dot_mask
        return raw * self.mask

    def project(self, param: torch.Tensor) -> torch.Tensor:
        raw = self.field(param)
        # Fixed-scale BPDA: solve an exact capped/headroom RMS projection on
        # detached values, then differentiate the rendered field at that scale.
        with torch.no_grad():
            hi = 1.0
            def achieved(scale):
                bounded = torch.minimum(torch.maximum(raw.detach()*scale, self.low), self.high)
                return float(torch.sqrt(torch.mean(bounded[:, self.face]**2)))
            while achieved(hi) < RMS and hi < 65536:
                hi *= 2
            if achieved(hi) < RMS - .25:
                raise ValueError("RMS 16 unattainable under this carrier and channel cap")
            lo = 0.0
            for _ in range(13):
                middle = (lo + hi)/2
                if achieved(middle) < RMS:
                    lo = middle
                else:
                    hi = middle
        return torch.minimum(torch.maximum(raw*hi, self.low), self.high)

    def render(self, delta: torch.Tensor) -> np.ndarray:
        image = self.source.copy()
        patch = (self.base + delta.detach()).permute(1,2,0).numpy()
        image[self.top:self.bottom, self.left:self.right] = np.rint(patch).clip(0,255).astype(np.uint8)
        return image

    def warp_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(delta.unsqueeze(0), self.grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]


def actual_source_scores(jpeg: bytes, box, native: SFaceModel, clean_feature: np.ndarray) -> dict:
    values = {}
    variants = make_variants(jpeg, box)
    for name in OBJECTIVE:
        variant = variants[name]
        if variant.status != "valid":
            values[name] = {"status": "inconclusive", "reason": variant.reason}
            continue
        result = native.embed(variant.image, expected_box=variant.target_box)
        values[name] = {"status": result.status, "reason": result.reason,
                        "source_cosine": cosine(result.feature, clean_feature) if result.status == "valid" else None}
    return values


def gallery_fields(gallery) -> dict:
    """Keep each condition's authoritative status alongside its scores."""
    return {"status": gallery.status, "reason": gallery.reason,
            "gallery_status": gallery.status, "gallery_reason": gallery.reason,
            "maximum_cosine": max(gallery.per_reference.values()) if gallery.per_reference else None,
            "own_source_cosine": gallery.own_source_score,
            "matching_references": gallery.matching_references,
            "nonmatch": gallery.nonmatch}


def optimize(source, box, matrix, native, torch_model, clean_feature, arm, log):
    carrier = Carrier(source, box, matrix, arm)
    rng = np.random.default_rng(0)
    if arm == "unrestricted_rgb":
        shape = tuple(carrier.base.shape)
    else:
        shape = (3, 14, 14)
    param = torch.nn.Parameter(torch.from_numpy(rng.standard_normal(shape).astype(np.float32)*.1))
    optimizer = torch.optim.Adam([param], lr=.15)
    clean_unit = torch.from_numpy(clean_feature.astype(np.float32))
    best = None
    started = perf_counter()
    for step in range(1, STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        delta = carrier.project(param)
        image = carrier.render(delta)
        jpeg = export_jpeg(image)
        variants = make_variants(jpeg, box)
        surrogate = carrier.warp_delta(delta)
        losses = []
        for name in OBJECTIVE:
            variant = variants[name]
            if variant.status != "valid":
                raise RuntimeError(f"{name}: {variant.reason}")
            exact_crop = aligned_rgb(variant.image, matrix)
            exact = torch.from_numpy(np.ascontiguousarray(exact_crop.transpose(2,0,1).astype(np.float32)))
            bpda_crop = exact + surrogate - surrogate.detach()
            feature = torch_model(bpda_crop.unsqueeze(0)).reshape(-1)
            losses.append(F.cosine_similarity(feature, clean_unit, dim=0))
        loss = torch.stack(losses).max()
        loss.backward()
        optimizer.step()
        if step % 8 == 0:
            actual = actual_source_scores(jpeg, box, native, clean_feature)
            vals = [actual[k]["source_cosine"] for k in OBJECTIVE]
            score = max(vals) if all(v is not None for v in vals) else 1.0
            row = {"arm": arm, "step": step, "surrogate_worst_cosine": float(loss.detach()),
                   "native_worst_cosine_or_failure_one": score, "native_conditions": actual,
                   "pre_jpeg_face_rms": float(torch.sqrt(torch.mean(delta.detach()[:,carrier.face]**2))),
                   "pre_jpeg_max_channel_delta": float(delta.detach().abs().max()),
                   "jpeg_sha256": hashlib.sha256(jpeg).hexdigest()}
            log.append(row)
            if best is None or score < best["score"]:
                best = {"score": score, "step": step, "jpeg": jpeg,
                        "image": image, "row": row}
    best["seconds"] = perf_counter()-started
    return best


def main(args: argparse.Namespace) -> None:
    global OUT
    OUT = args.output.resolve()
    identities = tuple(args.identities)
    arms = tuple(arm for arm in ARMS if arm in args.arms)
    if identities not in (FROZEN_IDS[:2], FROZEN_IDS[2:], FROZEN_IDS):
        raise ValueError("Identities must be the frozen first two, remaining six, or all eight in order")
    if len(arms) != len(args.arms) or len(set(args.arms)) != len(args.arms):
        raise ValueError("Arms must be unique names from the three fixed controls")
    if OUT.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Experiment output must be outside Git")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = DATA / "frll" / "manifest.json"
    groups = grouped_images(manifest, "development")
    calibration_path = DATA / "runs" / "calibration-sface-v1" / "calibration.json"
    artifact = json.loads(calibration_path.read_text())
    if digest(manifest) != artifact["dataset"]["manifest_sha256"]:
        raise ValueError("FRLL manifest differs from frozen calibration")
    verify_input_images(json.loads(manifest.read_text(encoding="utf-8")), groups, identities)
    calibration = Calibration(**artifact["calibration"])
    yunet, sface = WEIGHTS / "face_detection_yunet_2023mar.onnx", WEIGHTS / "face_recognition_sface_2021dec.onnx"
    for name, path in (("yunet", yunet), ("sface", sface)):
        if digest(path) != artifact["pipeline"]["model_artifacts"][name]["sha256"]:
            raise ValueError(f"Frozen {name} model hash differs")
    for name in ("recognition", "imaging", "evaluation"):
        relative = f"fckface_lab/{name}.py"
        pinned = artifact["pipeline"]["source_sha256"][relative]
        if digest(Path(__file__).parent / relative) != pinned:
            raise ValueError(f"Frozen calibration pipeline changed: {relative}")
    native = SFaceModel(yunet, sface)
    t0 = perf_counter()
    model = convert(onnx.load(str(sface))).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    record = {"hypothesis": "H5 independently optimized regular dots versus conventional controls",
              "scope": "Frozen FRLL development identities only; SFace only; no held-out models/data",
              "identities": identities, "arms": arms, "steps_each": STEPS, "checkpoint_every": 8,
              "target_face_rms": RMS, "max_channel_delta": MAX_DELTA,
              "source_only_objective": OBJECTIVE, "final_conditions": CONDITIONS,
              "gradient": "actual JPEG Q95 and condition pixels in forward; fixed clean-landmark affine, identity BPDA for rounding/JPEG/blur in backward; no gallery optimization",
              "source_sha256": {"manifest": digest(manifest), "calibration": digest(calibration_path),
                                "yunet": digest(yunet), "sface": digest(sface), "script": digest(Path(__file__))},
              "cases": [], "conversion_seconds": perf_counter()-t0}
    save_json(OUT/"header.json", {k:v for k,v in record.items() if k != "cases"})
    frozen = []
    try:
        for identity in identities:
            path = groups[identity]["neutral_front"]
            source = decode_image(path)
            clean = native.embed(source)
            if clean.status != "valid":
                raise RuntimeError(f"{identity} clean source: {clean.reason}")
            matrix = affine(clean.landmarks)
            native_crop = native.recognizer.alignCrop(cv2.cvtColor(source,cv2.COLOR_RGB2BGR),
                np.r_[np.array([clean.selected_box[0],clean.selected_box[1],
                    clean.selected_box[2]-clean.selected_box[0],clean.selected_box[3]-clean.selected_box[1]]),
                    clean.landmarks.reshape(-1), [1.0]].astype(np.float32))
            our_crop = cv2.cvtColor(aligned_rgb(source,matrix),cv2.COLOR_RGB2BGR)
            crop_mae = float(np.mean(np.abs(native_crop.astype(np.int16)-our_crop.astype(np.int16))))
            crop_max = int(np.max(np.abs(native_crop.astype(np.int16)-our_crop.astype(np.int16))))
            with torch.no_grad():
                model_feature = raw_feature(model, cv2.cvtColor(native_crop,cv2.COLOR_BGR2RGB)).numpy()
            native_feature = native.recognizer.feature(native_crop).reshape(-1)
            parity = {"crop_mae": crop_mae, "crop_max": crop_max,
                      "raw_feature_max_abs": float(np.max(np.abs(model_feature-native_feature))),
                      "l2_feature_cosine": cosine(model_feature,native_feature)}
            if crop_mae > .1 or parity["l2_feature_cosine"] < .99999:
                raise RuntimeError(f"{identity} clean alignment/feature parity failed: {parity}")
            case = {"identity": identity, "source_sha256": digest(path), "source_path": str(path),
                    "clean_parity": parity, "arms": {}}
            print(identity, "parity", parity, flush=True)
            for arm in arms:
                case_dir = OUT / identity / arm
                jpg = case_dir / "selected.jpg"
                log = []
                best = optimize(source, clean.selected_box, matrix, native, model, clean.feature, arm, log)
                case_dir.mkdir(parents=True)
                jpg.write_bytes(best["jpeg"])
                save_json(case_dir/"checkpoints.json", log)
                case["arms"][arm] = {"selected_step": best["step"], "native_objective_worst_cosine": best["score"],
                    "selected_jpeg_sha256": digest(jpg), "seconds": best["seconds"],
                    "gradient_steps": STEPS, "native_source_queries": len(log)*len(OBJECTIVE),
                    "checkpoints": len(log), "pre_jpeg_face_rms": best["row"]["pre_jpeg_face_rms"],
                    "pre_jpeg_max_channel_delta": best["row"]["pre_jpeg_max_channel_delta"]}
                frozen.append((identity, arm, jpg, clean.selected_box, source))
                print(identity, arm, "frozen", best["step"], best["score"], flush=True)
            record["cases"].append(case)
        # All source-only selections are immutable before the first gallery
        # reference is loaded or scored.
        save_json(OUT/"frozen.json", record)
        references = {}
        controls = {}
        for identity in identities:
            references[identity] = []
            for view in VIEWS:
                embedded = native.embed(decode_image(groups[identity][view]))
                if embedded.status == "valid":
                    references[identity].append(Reference(f"{identity}:{view}", identity, embedded.feature))
            source = decode_image(groups[identity]["neutral_front"])
            box = next(box for who, arm, jpg, box, original in frozen if who == identity)
            controls[identity] = {}
            for name, variant in make_variants(export_jpeg(source), box).items():
                if name not in CONDITIONS:
                    continue
                if variant.status != "valid":
                    controls[identity][name] = {"status": "inconclusive", "reason": variant.reason}
                    continue
                embedded = native.embed(variant.image, expected_box=variant.target_box)
                gallery = evaluate_gallery(embedded, references[identity], calibration,
                    own_identity=identity, own_source_name=f"{identity}:neutral_front",
                    expected_reference_names=[f"{identity}:{view}" for view in VIEWS])
                controls[identity][name] = {"status": gallery.status,
                    "own_identity_matched": gallery.own_identity_matched, "reason":gallery.reason}
            if not all(row["status"] == "valid" and row["own_identity_matched"] is True
                       for row in controls[identity].values()):
                raise RuntimeError(f"{identity} clean controls not eligible: {controls[identity]}")
        record["clean_controls"] = controls
        for identity, arm, jpg, box, source in frozen:
            result = {}
            clean_variants = make_variants(export_jpeg(source), box)
            for name in CONDITIONS:
                variant = make_variants(jpg.read_bytes(),box)[name]
                row = {"status": variant.status, "reason": variant.reason}
                if variant.status == "valid":
                    embedded = native.embed(variant.image, expected_box=variant.target_box)
                    gallery = evaluate_gallery(embedded, references[identity], calibration,
                        own_identity=identity, own_source_name=f"{identity}:neutral_front",
                        expected_reference_names=[f"{identity}:{view}" for view in VIEWS])
                    row.update(gallery_fields(gallery))
                    control = clean_variants[name]
                    if control.status == "valid" and control.image.shape == variant.image.shape:
                        row["post_jpeg_distortion"] = actual_distortion(control.image,variant.image,
                            (variant.target_box[0],variant.target_box[1],
                             variant.target_box[2]-variant.target_box[0],variant.target_box[3]-variant.target_box[1]))
                result[name] = row
            save_json(jpg.parent/"evaluation.json",result)
            next(c for c in record["cases"] if c["identity"]==identity)["arms"][arm]["conditions"] = result
        record["status"] = "complete"
        record["wall_seconds"] = perf_counter()-t0
        save_json(OUT/"results.json",record)
        print("complete", OUT, record["wall_seconds"], flush=True)
    except Exception as exc:
        save_json(OUT/"failure.json", {"type":type(exc).__name__,"message":str(exc), "completed_arms":len(frozen)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="New outside-Git output directory (default: gradient-art-v2)")
    parser.add_argument("--identities", nargs="+", default=IDS,
                        help="Exact frozen first two, remaining six, or all eight development IDs in order")
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS,
                        help="Subset of the three fixed arms; default all three")
    main(parser.parse_args())
