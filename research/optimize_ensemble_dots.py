"""H6 development-only dot ensemble pilot with a separate gradient surrogate.

No held-out model or identity is used. A standard float32 Keras clone of the
author mixed-float16 H5 supplies differentiable gradients only after synthetic
and clean-development parity checks. Fresh native author-H5 and OpenCV SFace
scores select checkpoints and score final frozen JPEGs. JPEG, blur, rounding,
and fixed-affine interpolation use an identity BPDA backward approximation.
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
import onnx
import torch
import torch.nn.functional as F
from onnx2torch import convert
from skimage import transform

from fckface_lab.calibration import Calibration
from fckface_lab.datasets import digest, grouped_images
from fckface_lab.evaluation import Reference, evaluate_gallery
from fckface_lab.ghostface import GhostFaceModel, TEMPLATE, author_align, author_normalize, unit_feature
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.recognition import SFaceModel
from optimize_gradient_art import Carrier, affine, aligned_rgb, cosine, RMS, MAX_DELTA

DATA = Path.home() / "Downloads" / "FCKFACE-data"
WEIGHTS = Path.home() / "Downloads" / "Face Privacy Filter" / "cache" / ".deepface" / "weights"
OUT = DATA / "runs" / "ensemble-dots-v1"
PILOT = DATA / "runs" / "ensemble-dots-pilot-v1"
IDS = ("frll-001", "frll-003")
VIEWS = ("neutral_front", "smiling_front", "neutral_left_3quarter", "neutral_right_3quarter")
OBJECTIVE = ("export", "jpeg75_420", "blur")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960", "half_restore", "crop90", "blur")
ARMS = (("ghost_only",48,8),("sface_ghost_ensemble",24,4))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def load_float32_clone(author):
    import tensorflow as tf
    def float32_layer(layer):
        config = layer.get_config()
        config["dtype"] = "float32"
        return layer.__class__.from_config(config)
    clone = tf.keras.models.clone_model(author,clone_function=float32_layer)
    clone.set_weights(author.get_weights())
    clone.trainable=False
    original_bn=[(x.name,x.epsilon) for x in author.layers if isinstance(x,tf.keras.layers.BatchNormalization)]
    cloned_bn=[(x.name,x.epsilon) for x in clone.layers if isinstance(x,tf.keras.layers.BatchNormalization)]
    if original_bn != cloned_bn or not all(np.array_equal(a,b) for a,b in zip(author.get_weights(),clone.get_weights())):
        raise ValueError("Float32 clone changed serialized BN epsilon or weights")
    return clone


def author_grid(carrier: Carrier, points: np.ndarray) -> torch.Tensor:
    fitted=transform.SimilarityTransform()
    if not fitted.estimate(points,TEMPLATE):
        raise ValueError("Author similarity fit failed")
    inverse=np.linalg.inv(fitted.params)
    yy,xx=np.mgrid[:112,:112].astype(np.float64)
    sx=inverse[0,0]*xx+inverse[0,1]*yy+inverse[0,2]-carrier.left
    sy=inverse[1,0]*xx+inverse[1,1]*yy+inverse[1,2]-carrier.top
    h,w=carrier.roi.shape[:2]
    grid=np.stack((2*sx/(w-1)-1,2*sy/(h-1)-1),axis=-1).astype(np.float32)
    return torch.from_numpy(grid).unsqueeze(0)


def ghost_value_gradient(clone, image_crop: np.ndarray, clean_unit: np.ndarray):
    import tensorflow as tf
    pixel=tf.constant(image_crop.astype(np.float32))
    source=tf.constant(clean_unit.astype(np.float32))
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(pixel)
        raw=clone((pixel-127.5)*.0078125,training=False)
        score=tf.reduce_sum(tf.math.l2_normalize(raw,axis=1)[0]*source)
    gradient=tape.gradient(score,pixel)
    if gradient is None or not np.isfinite(gradient.numpy()).all():
        raise ValueError("Float32 Ghost crop gradient invalid")
    return float(score.numpy()),torch.from_numpy(np.ascontiguousarray(gradient.numpy()[0].transpose(2,0,1)))


def margins(jpeg: bytes,box,native,clean_features,thresholds,models):
    variants=make_variants(jpeg,box)
    rows={};values=[]
    for model_name in models:
        rows[model_name]={}
        for name in OBJECTIVE:
            variant=variants[name]
            embedded=native[model_name].embed(variant.image,expected_box=variant.target_box) if variant.status=="valid" else None
            score=cosine(embedded.feature,clean_features[model_name]) if embedded is not None and embedded.status=="valid" else None
            margin=(score-thresholds[model_name])/(1-thresholds[model_name]) if score is not None else None
            rows[model_name][name]={"status":embedded.status if embedded is not None else "inconclusive",
                                    "reason":embedded.reason if embedded is not None else variant.reason,
                                    "source_cosine":score,"normalized_margin":margin}
            values.append(margin if margin is not None else 1.0)
    return max(values),rows


def optimize(source,box,sface_points,ghost_points,models,clone,clean_features,thresholds,arm,steps,checkpoint_every):
    carrier=Carrier(source,box,affine(sface_points),"dots_14x14")
    ghost_warp_grid=author_grid(carrier,ghost_points)
    rng=np.random.default_rng(0)
    param=torch.nn.Parameter(torch.from_numpy(rng.standard_normal((3,14,14)).astype(np.float32)*.1))
    optimizer=torch.optim.Adam([param],lr=.15)
    source_sface=torch.from_numpy(clean_features["SFace"].astype(np.float32))
    t0=perf_counter();log=[];best=None
    active=("GhostFaceNet",) if arm=="ghost_only" else ("SFace","GhostFaceNet")
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        delta=carrier.project(param)
        image=carrier.render(delta)
        jpeg=export_jpeg(image)
        variants=make_variants(jpeg,box)
        sface_delta=carrier.warp_delta(delta)
        ghost_delta=F.grid_sample(delta.unsqueeze(0),ghost_warp_grid,mode="bilinear",padding_mode="zeros",align_corners=True)[0]
        terms=[];surrogate={}
        for name in OBJECTIVE:
            variant=variants[name]
            if variant.status!="valid":
                raise RuntimeError(f"Condition {name}: {variant.reason}")
            if "SFace" in active:
                aligned=aligned_rgb(variant.image,affine(sface_points))
                exact=torch.from_numpy(np.ascontiguousarray(aligned.transpose(2,0,1).astype(np.float32)))
                crop=exact+sface_delta-sface_delta.detach()
                raw=models["SFaceGradient"](crop.unsqueeze(0)).reshape(-1)
                value=F.cosine_similarity(raw,source_sface,dim=0)
                term=(value-thresholds["SFace"])/(1-thresholds["SFace"])
                terms.append(term);surrogate[f"SFace:{name}"]=float(value.detach())
            ghost_crop=author_align(variant.image,ghost_points)
            ghost_score,ghost_grad=ghost_value_gradient(clone,ghost_crop[None],clean_features["GhostFaceNet"])
            ghost_term=(torch.tensor(ghost_score)+((ghost_delta-ghost_delta.detach())*ghost_grad).sum()-thresholds["GhostFaceNet"])/(1-thresholds["GhostFaceNet"])
            terms.append(ghost_term);surrogate[f"GhostFaceNet:{name}"]=ghost_score
        loss=torch.stack(terms).max()
        loss.backward()
        optimizer.step()
        if step%checkpoint_every==0:
            score,native_rows=margins(jpeg,box,models,clean_features,thresholds,active)
            row={"arm":arm,"step":step,"native_worst_normalized_margin_or_failure_one":score,
                 "native":native_rows,"surrogate":surrogate,
                 "surrogate_worst_margin":float(loss.detach()),
                 "pre_jpeg_face_rms":float(torch.sqrt(torch.mean(delta.detach()[:,carrier.face]**2))),
                 "pre_jpeg_max_channel_delta":float(delta.detach().abs().max()),
                 "jpeg_sha256":hashlib.sha256(jpeg).hexdigest()}
            log.append(row)
            if best is None or score<best["score"]:
                best={"score":score,"step":step,"jpeg":jpeg,"row":row}
    best["seconds"]=perf_counter()-t0
    return best,log


def pilot():
    import tensorflow as tf
    torch.set_num_threads(2);cv2.setNumThreads(2)
    if PILOT.exists():
        raise FileExistsError(PILOT)
    report=json.loads((OUT/"ghost-float32-clone-diagnostic.json").read_text())
    if not report["gradient_finite"] or any(v["unit_cosine"]<.999 or v["unit_max_abs_difference"]>.02 for v in report["parity"].values()):
        raise RuntimeError("Float32 clone forward parity preflight failed")
    if not any(v["relative_error"]<=.1 or v["absolute_error"]<=1e-6 for v in report["unit_l2_finite_differences"]):
        raise RuntimeError("Float32 clone directional derivative preflight failed")
    manifest=DATA/"frll"/"manifest.json"
    groups=grouped_images(manifest,"development")
    s_cal_path=DATA/"runs"/"calibration-sface-v1"/"calibration.json"
    g_cal_path=DATA/"runs"/"calibration-ghostface-v1"/"calibration.json"
    s_cal=json.loads(s_cal_path.read_text());g_cal=json.loads(g_cal_path.read_text())
    yunet=WEIGHTS/"face_detection_yunet_2023mar.onnx"
    sface=WEIGHTS/"face_recognition_sface_2021dec.onnx"
    ghost_path=WEIGHTS/"ghostfacenet_v1.h5"
    for cal,name,path in ((s_cal,"yunet",yunet),(s_cal,"sface",sface),(g_cal,"yunet",yunet),(g_cal,"ghostface",ghost_path)):
        if digest(path)!=cal["pipeline"]["model_artifacts"][name]["sha256"]:
            raise ValueError(f"{name} differs from calibrated model")
    for cal in (s_cal,g_cal):
        for name,sha in cal["pipeline"]["source_sha256"].items():
            if name.startswith("fckface_lab/") and digest(Path(__file__).parent/name)!=sha:
                raise ValueError(f"Calibrated pipeline changed: {name}")
    if digest(manifest)!=s_cal["dataset"]["manifest_sha256"] or digest(manifest)!=g_cal["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from calibration")
    thresholds={"SFace":Calibration(**s_cal["calibration"]).threshold,
                "GhostFaceNet":Calibration(**g_cal["calibration"]).threshold}
    PILOT.mkdir(parents=True)
    shutil.copyfile(Path(__file__),PILOT/"executed-source.py")
    models={"SFace":SFaceModel(yunet,sface),"GhostFaceNet":GhostFaceModel(yunet,ghost_path)}
    clone=load_float32_clone(models["GhostFaceNet"].model)
    models["SFaceGradient"]=convert(onnx.load(str(sface))).eval()
    for parameter in models["SFaceGradient"].parameters():
        parameter.requires_grad_(False)
    header={"hypothesis":"H6 fixed regular dots jointly optimized for two development recognizers",
        "identities":IDS,"arms":ARMS,"seed":0,"target_face_rms":RMS,"max_channel_delta":MAX_DELTA,
        "objective_conditions":OBJECTIVE,"final_conditions":CONDITIONS,"thresholds":thresholds,
        "gradient":"float32 same-weight Ghost clone; author H5 only for native checkpoint/final scores; fixed clean landmarks and identity BPDA for interpolation/JPEG/blur",
        "source_sha256":{"script":digest(Path(__file__)),"carrier":digest(Path(__file__).parent/"optimize_gradient_art.py"),
            "manifest":digest(manifest),"sface_calibration":digest(s_cal_path),"ghost_calibration":digest(g_cal_path),
            "yunet":digest(yunet),"sface":digest(sface),"ghostface":digest(ghost_path),
            "clone_diagnostic":digest(OUT/"ghost-float32-clone-diagnostic.json")}}
    write_json(PILOT/"header.json",header)
    records=[];frozen=[];t0=perf_counter()
    try:
        for identity in IDS:
            source_path=groups[identity]["neutral_front"]
            source=decode_image(source_path)
            clean={name:models[name].embed(source) for name in ("SFace","GhostFaceNet")}
            if any(v.status!="valid" for v in clean.values()):
                raise RuntimeError(f"{identity} clean detection failure: {clean}")
            box=clean["SFace"].selected_box
            if clean["GhostFaceNet"].selected_box!=box:
                raise ValueError("Clean model face boxes differ")
            features={name:clean[name].feature for name in ("SFace","GhostFaceNet")}
            case={"identity":identity,"source_sha256":digest(source_path),"arms":{}}
            for arm,steps,interval in ARMS:
                best,log=optimize(source,box,clean["SFace"].landmarks,clean["GhostFaceNet"].landmarks,
                                  models,clone,features,thresholds,arm,steps,interval)
                directory=PILOT/identity/arm
                directory.mkdir(parents=True)
                path=directory/"selected.jpg"
                path.write_bytes(best["jpeg"])
                write_json(directory/"checkpoints.json",log)
                case["arms"][arm]={"selected_step":best["step"],"native_objective_worst_margin":best["score"],
                    "selected_jpeg_sha256":digest(path),"seconds":best["seconds"],"steps":steps,
                    "native_checkpoints":len(log),"native_model_condition_queries":len(log)*len(OBJECTIVE)*(1 if arm=="ghost_only" else 2),
                    "pre_jpeg_face_rms":best["row"]["pre_jpeg_face_rms"],
                    "pre_jpeg_max_channel_delta":best["row"]["pre_jpeg_max_channel_delta"]}
                frozen.append((identity,arm,path,box,source))
                print(identity,arm,"frozen",best["step"],best["score"],flush=True)
            records.append(case)
        write_json(PILOT/"frozen.json",{"header":header,"cases":records,
            "all_outputs_frozen_before_gallery":True})
        # Only after four JPEGs are fixed: clean gallery/control and all-seven scoring.
        for identity in IDS:
            refs={name:[] for name in ("SFace","GhostFaceNet")}
            for view in VIEWS:
                image=decode_image(groups[identity][view])
                for name in refs:
                    result=models[name].embed(image)
                    if result.status=="valid":
                        refs[name].append(Reference(f"{identity}:{view}",identity,result.feature))
            case=next(x for x in records if x["identity"]==identity)
            case["gallery_reference_counts"]={name:len(rows) for name,rows in refs.items()}
            for name,rows in refs.items():
                if len(rows)!=len(VIEWS):
                    raise RuntimeError(f"{identity}/{name} incomplete gallery")
            box=next(x[3] for x in frozen if x[0]==identity)
            controls=make_variants(export_jpeg(decode_image(groups[identity]["neutral_front"])),box)
            case["clean_controls"]={}
            for name in refs:
                case["clean_controls"][name]={}
                for condition in CONDITIONS:
                    variant=controls[condition]
                    probe=models[name].embed(variant.image,expected_box=variant.target_box)
                    evaluation=evaluate_gallery(probe,refs[name],Calibration(**(s_cal if name=="SFace" else g_cal)["calibration"]),
                        own_identity=identity,own_source_name=f"{identity}:neutral_front",
                        expected_reference_names=[f"{identity}:{v}" for v in VIEWS])
                    case["clean_controls"][name][condition]={"status":evaluation.status,
                        "own_identity_matched":evaluation.own_identity_matched}
                    if evaluation.status!="valid" or evaluation.own_identity_matched is not True:
                        raise RuntimeError(f"{identity}/{name}/{condition} ineligible clean control")
            for who,arm,path,box,source in frozen:
                if who!=identity:
                    continue
                variants=make_variants(path.read_bytes(),box)
                final={}
                for model_name in refs:
                    final[model_name]={}
                    for condition in CONDITIONS:
                        variant=variants[condition]
                        if variant.status!="valid":
                            final[model_name][condition]={"status":"inconclusive","reason":variant.reason}
                            continue
                        probe=models[model_name].embed(variant.image,expected_box=variant.target_box)
                        evaluation=evaluate_gallery(probe,refs[model_name],Calibration(**(s_cal if model_name=="SFace" else g_cal)["calibration"]),
                            own_identity=identity,own_source_name=f"{identity}:neutral_front",
                            expected_reference_names=[f"{identity}:{v}" for v in VIEWS])
                        final[model_name][condition]={"status":evaluation.status,"reason":evaluation.reason,
                            "maximum_cosine":max(evaluation.per_reference.values()) if evaluation.per_reference else None,
                            "own_source_cosine":evaluation.own_source_score,
                            "matching_references":evaluation.matching_references,"nonmatch":evaluation.nonmatch}
                write_json(path.parent/"evaluation.json",final)
                case["arms"][arm]["evaluation"]=final
        write_json(PILOT/"results.json",{"status":"complete","header":header,
            "wall_seconds":perf_counter()-t0,"cases":records})
        print("H6 complete",PILOT,flush=True)
    except Exception as exc:
        write_json(PILOT/"failure.json",{"type":type(exc).__name__,"message":str(exc),
            "frozen_count":len(frozen)})
        raise


def diagnostic() -> dict:
    import tensorflow as tf
    ghost = GhostFaceModel(WEIGHTS / "face_detection_yunet_2023mar.onnx",
                           WEIGHTS / "ghostfacenet_v1.h5")
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(1, 112, 112, 3), dtype=np.uint8).astype(np.float32)
    direction = rng.standard_normal(crop.shape).astype(np.float32)
    direction /= np.sqrt(np.mean(direction**2))
    target = rng.standard_normal(512).astype(np.float32)
    target /= np.linalg.norm(target)
    target_tf = tf.constant(target)

    def loss(pixel):
        normalized = (pixel - 127.5) * .0078125
        raw = ghost.model(normalized, training=False)[0]
        return tf.reduce_sum(tf.math.l2_normalize(tf.cast(raw, tf.float32), axis=0) * target_tf)

    x = tf.Variable(crop)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        score = loss(x)
    gradient = tape.gradient(score, x)
    if gradient is None:
        raise RuntimeError("Author GhostFaceNet input gradient is None")
    grad = gradient.numpy()
    analytic = float(np.sum(grad * direction))
    finite = []
    for step in (.03, .1, .25, .5, 1.0):
        plus = float(loss(tf.constant(crop + step*direction)).numpy())
        minus = float(loss(tf.constant(crop - step*direction)).numpy())
        fd = (plus - minus)/(2*step)
        finite.append({"rms_pixel_step":step, "plus":plus, "minus":minus,
                       "directional_fd":fd, "absolute_error":abs(fd-analytic)})
    repeat = float(loss(tf.constant(crop)).numpy())
    file = WEIGHTS / "ghostfacenet_v1.h5"
    with file.open("rb") as stream:
        sha = hashlib.file_digest(stream,"sha256").hexdigest()
    result = {"author_h5_sha256":sha, "input":"seed0 synthetic RGB112 float32 pixels, (RGB-127.5)/128",
              "saved_graph_output_dtype":str(ghost.model.output.dtype),
              "loss":"cosine of raw author feature to deterministic random unit target",
              "value":float(score.numpy()), "repeat_value":repeat,
              "gradient_finite":bool(np.isfinite(grad).all()),
              "gradient_l2":float(np.linalg.norm(grad)),
              "analytic_directional_derivative":analytic,
              "finite_differences":finite}
    return result


def clone_diagnostic() -> dict:
    """Standard Keras float32 clone, kept distinct from the saved author graph."""
    import tensorflow as tf
    from fckface_lab.datasets import grouped_images
    from fckface_lab.ghostface import author_align, author_normalize, unit_feature
    from fckface_lab.imaging import decode_image
    ghost = GhostFaceModel(WEIGHTS / "face_detection_yunet_2023mar.onnx",
                           WEIGHTS / "ghostfacenet_v1.h5")

    def float32_layer(layer):
        config = layer.get_config()
        config["dtype"] = "float32"
        return layer.__class__.from_config(config)

    clone = tf.keras.models.clone_model(ghost.model, clone_function=float32_layer)
    clone.set_weights(ghost.model.get_weights())
    clone.trainable = False
    original_bn = [(layer.name, layer.epsilon) for layer in ghost.model.layers
                   if isinstance(layer, tf.keras.layers.BatchNormalization)]
    clone_bn = [(layer.name, layer.epsilon) for layer in clone.layers
                if isinstance(layer, tf.keras.layers.BatchNormalization)]
    if original_bn != clone_bn:
        raise ValueError("Float32 clone changed serialized BN epsilon")
    if not all(np.array_equal(a,b) for a,b in zip(ghost.model.get_weights(),clone.get_weights())):
        raise ValueError("Float32 clone weights differ from author H5")
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    cases = [("synthetic", crop)]
    groups = grouped_images(DATA / "frll" / "manifest.json", "development")
    for identity in ("frll-001", "frll-003"):
        source = decode_image(groups[identity]["neutral_front"])
        detection = ghost.embed(source)
        if detection.status != "valid":
            raise RuntimeError(f"{identity}: {detection.reason}")
        cases.append((identity, author_align(source, detection.landmarks)))
    parity = {}
    for name, image in cases:
        data = author_normalize(image)[None]
        author = np.asarray(ghost.model(data, training=False).numpy(), np.float32).reshape(-1)
        substitute = np.asarray(clone(data, training=False).numpy(), np.float32).reshape(-1)
        parity[name] = {"raw_max_abs_difference":float(np.max(np.abs(author-substitute))),
                        "unit_max_abs_difference":float(np.max(np.abs(unit_feature(author)-unit_feature(substitute)))),
                        "unit_cosine":float(np.dot(unit_feature(author).astype(np.float64),unit_feature(substitute).astype(np.float64)))}
    direction = rng.standard_normal((1,112,112,3)).astype(np.float32)
    direction /= np.sqrt(np.mean(direction**2))
    target = rng.standard_normal(512).astype(np.float32)
    target /= np.linalg.norm(target)
    target_tf = tf.constant(target)

    def loss(pixel):
        raw = clone((pixel-127.5)*.0078125,training=False)[0]
        return tf.reduce_sum(tf.math.l2_normalize(tf.cast(raw,tf.float32),axis=0)*target_tf)

    pixels = tf.Variable(crop[None].astype(np.float32))
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(pixels)
        value = loss(pixels)
    gradient = tape.gradient(value,pixels)
    if gradient is None:
        raise RuntimeError("Float32 clone gradient missing")
    grad = gradient.numpy()
    analytic = float(np.sum(grad*direction))
    finite=[]
    for step in (.03,.1,.25,.5,1.0):
        plus=float(loss(tf.constant(crop[None].astype(np.float32)+step*direction)).numpy())
        minus=float(loss(tf.constant(crop[None].astype(np.float32)-step*direction)).numpy())
        estimate=(plus-minus)/(2*step)
        finite.append({"rms_pixel_step":step,"directional_fd":estimate,
                       "absolute_error":abs(estimate-analytic)})
    unit_direction = direction / np.linalg.norm(direction)
    unit_analytic = float(np.sum(grad*unit_direction))
    unit_finite = []
    for step in (.25,.5,1.0,2.0,4.0):
        plus=float(loss(tf.constant(crop[None].astype(np.float32)+step*unit_direction)).numpy())
        minus=float(loss(tf.constant(crop[None].astype(np.float32)-step*unit_direction)).numpy())
        estimate=(plus-minus)/(2*step)
        unit_finite.append({"l2_pixel_step":step,"directional_fd":estimate,
            "absolute_error":abs(estimate-unit_analytic),
            "relative_error":abs(estimate-unit_analytic)/max(abs(unit_analytic),1e-12)})
    return {"kind":"separate_float32_clone_surrogate", "parity":parity,
            "batch_norm_count":len(original_bn), "batch_norm_epsilon_identical":True,
            "all_weights_identical":True,
            "gradient_finite":bool(np.isfinite(grad).all()),
            "gradient_l2":float(np.linalg.norm(grad)),
            "analytic_directional_derivative":analytic,
            "finite_differences":finite,
            "unit_l2_analytic_directional_derivative":unit_analytic,
            "unit_l2_finite_differences":unit_finite}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-only",action="store_true")
    parser.add_argument("--clone-diagnostic",action="store_true")
    parser.add_argument("--pilot",action="store_true")
    args = parser.parse_args()
    if sum((args.diagnostic_only,args.clone_diagnostic,args.pilot)) != 1:
        parser.error("Choose exactly one of --diagnostic-only, --clone-diagnostic, --pilot")
    if args.pilot:
        pilot()
        return
    OUT.mkdir(parents=True,exist_ok=True)
    if args.clone_diagnostic:
        result = clone_diagnostic()
        path = OUT / "ghost-float32-clone-diagnostic.json"
        path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2),flush=True)
        return
    result = diagnostic()
    path = OUT / "ghost-gradient-diagnostic.json"
    path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
