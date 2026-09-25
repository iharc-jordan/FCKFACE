"""H7: fixed versus refreshed alignment for two-model regular dots.

Development only. Both arms keep H5's 14x14 circle carrier, seed 0, RMS 16,
and channel cap 64. JPEG/blur/rounding and landmark refresh have no derivative;
the float32 Ghost clone and fixed-affine grid provide a BPDA approximation.
Fresh author H5 and OpenCV SFace select checkpoints. Four JPEGs freeze before
any gallery is loaded. Detection/selection failures remain inconclusive.
"""
from __future__ import annotations

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
from fckface_lab.ghostface import GhostFaceModel, TEMPLATE, author_align
from fckface_lab.imaging import decode_image, export_jpeg, make_variants
from fckface_lab.patterns import actual_distortion
from fckface_lab.recognition import (SFaceModel, detector_frame,
    restore_face_coordinates, select_face)
from optimize_gradient_art import Carrier, affine, aligned_rgb, cosine, RMS, MAX_DELTA
from optimize_ensemble_dots import load_float32_clone

DATA=Path.home()/"Downloads"/"FCKFACE-data"
WEIGHTS=Path.home()/"Downloads"/"Face Privacy Filter"/"cache"/".deepface"/"weights"
OUT=DATA/"runs"/"alignment-dots-v1"
IDS=("frll-001","frll-003")
VIEWS=("neutral_front","smiling_front","neutral_left_3quarter","neutral_right_3quarter")
OBJECTIVE=("export","jpeg75_420","blur","crop90")
CONDITIONS=("export","jpeg85_420","jpeg75_420","resize960","half_restore","crop90","blur")
ARMS=("fixed_condition_landmarks","refresh_edited_landmarks")
STEPS=18
CHECKPOINT_EVERY=3


def write_json(path:Path,value:object)->None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def crop_offset(name:str,full_shape:tuple[int,...])->tuple[int,int]:
    if name!="crop90":
        return (0,0)
    h,w=full_shape[:2]
    return (round(w*.05),round(h*.05))


def warp_grid(carrier:Carrier,forward:np.ndarray,offset:tuple[int,int])->torch.Tensor:
    inverse=np.linalg.inv(forward)
    yy,xx=np.mgrid[:112,:112].astype(np.float64)
    sx=inverse[0,0]*xx+inverse[0,1]*yy+inverse[0,2]+offset[0]-carrier.left
    sy=inverse[1,0]*xx+inverse[1,1]*yy+inverse[1,2]+offset[1]-carrier.top
    h,w=carrier.roi.shape[:2]
    grid=np.stack((2*sx/(w-1)-1,2*sy/(h-1)-1),axis=-1).astype(np.float32)
    return torch.from_numpy(grid).unsqueeze(0)


def ghost_matrix(points:np.ndarray)->np.ndarray:
    fitted=transform.SimilarityTransform()
    if not fitted.estimate(points,TEMPLATE):
        raise ValueError("Ghost author five-point similarity fit failed")
    return fitted.params


def detect_points(detector,variant)->np.ndarray:
    if variant.status!="valid" or variant.image is None:
        raise ValueError(f"Invalid condition: {variant.reason}")
    image=variant.image
    h,w=image.shape[:2]
    frame=detector_frame(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    fh,fw=frame.shape[:2]
    detector.setInputSize((fw,fh))
    _,faces=detector.detect(frame)
    restored=restore_face_coordinates(faces,(w,h),(fw,fh))
    face,reason=select_face(restored,variant.target_box)
    if face is None:
        raise RuntimeError(f"Fresh YuNet target detection: {reason}")
    return np.asarray(face[4:14].reshape(5,2),np.float32)


def tf_value_gradient_functions(clone):
    import tensorflow as tf
    def eager(pixel,source):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(pixel)
            raw=clone((pixel-127.5)*.0078125,training=False)
            score=tf.reduce_sum(tf.math.l2_normalize(raw,axis=1)[0]*source)
        return score,tape.gradient(score,pixel)
    compiled=tf.function(eager,input_signature=[
        tf.TensorSpec((1,112,112,3),tf.float32),tf.TensorSpec((512,),tf.float32)])
    rng=np.random.default_rng(0)
    pixels=tf.constant(rng.integers(0,256,size=(1,112,112,3),dtype=np.uint8).astype(np.float32))
    source=rng.standard_normal(512).astype(np.float32)
    source/=np.linalg.norm(source)
    source=tf.constant(source)
    start=perf_counter();ev,eg=eager(pixels,source);eager_seconds=perf_counter()-start
    try:
        start=perf_counter();cv,cg=compiled(pixels,source);first_compiled_seconds=perf_counter()-start
        start=perf_counter();cv2,cg2=compiled(pixels,source);warm_compiled_seconds=perf_counter()-start
    except Exception as exc:
        return eager,{"synthetic_seed":0,"compiled_accepted":False,
                      "reason":f"{type(exc).__name__}: {exc}",
                      "eager_value":float(ev.numpy()),"eager_seconds":eager_seconds}
    report={"synthetic_seed":0,"eager_value":float(ev.numpy()),
        "compiled_value":float(cv.numpy()),
        "value_abs_difference":abs(float(ev.numpy())-float(cv.numpy())),
        "gradient_max_abs_difference":float(np.max(np.abs(eg.numpy()-cg.numpy()))),
        "compiled_repeat_value_difference":abs(float(cv.numpy())-float(cv2.numpy())),
        "compiled_repeat_gradient_max_abs":float(np.max(np.abs(cg.numpy()-cg2.numpy()))),
        "eager_seconds":eager_seconds,"first_compiled_seconds":first_compiled_seconds,
        "warm_compiled_seconds":warm_compiled_seconds}
    passed=(np.isfinite(eg.numpy()).all() and np.isfinite(cg.numpy()).all()
            and report["value_abs_difference"]<=1e-5
            and report["gradient_max_abs_difference"]<=1e-6)
    report["compiled_accepted"]=bool(passed)
    return (compiled if passed else eager),report


def ghost_term(function,clone_crop,clean_feature,threshold,proxy):
    import tensorflow as tf
    crop=tf.constant(clone_crop[None].astype(np.float32))
    source=tf.constant(clean_feature.astype(np.float32))
    score,grad=function(crop,source)
    if grad is None or not np.isfinite(grad.numpy()).all():
        raise RuntimeError("Ghost float32 crop gradient invalid")
    gradient=torch.from_numpy(np.ascontiguousarray(grad.numpy()[0].transpose(2,0,1)))
    term=(torch.tensor(float(score.numpy()))+((proxy-proxy.detach())*gradient).sum()-threshold)/(1-threshold)
    return term,float(score.numpy())


def native_objective(jpeg,box,models,clean,thresholds):
    variants=make_variants(jpeg,box)
    rows={};values=[]
    for model_name in ("SFace","GhostFaceNet"):
        rows[model_name]={}
        for name in OBJECTIVE:
            variant=variants[name]
            if variant.status!="valid":
                rows[model_name][name]={"status":"inconclusive","reason":variant.reason,"source_cosine":None}
                values.append(1.0)
                continue
            embedded=models[model_name].embed(variant.image,expected_box=variant.target_box)
            score=cosine(embedded.feature,clean[model_name]) if embedded.status=="valid" else None
            margin=(score-thresholds[model_name])/(1-thresholds[model_name]) if score is not None else None
            rows[model_name][name]={"status":embedded.status,"reason":embedded.reason,
                                    "source_cosine":score,"normalized_margin":margin}
            values.append(margin if margin is not None else 1.0)
    return max(values),rows


def optimize(source,box,clean_landmarks,models,torch_sface,tf_gradient,thresholds,clean_features,arm):
    carrier=Carrier(source,box,affine(clean_landmarks["export"]),"dots_14x14")
    rng=np.random.default_rng(0)
    param=torch.nn.Parameter(torch.from_numpy(rng.standard_normal((3,14,14)).astype(np.float32)*.1))
    optimizer=torch.optim.Adam([param],lr=.15)
    source_sface=torch.from_numpy(clean_features["SFace"].astype(np.float32))
    current_points=clean_landmarks.copy()
    log=[];best=None;t0=perf_counter()
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        delta=carrier.project(param)
        image=carrier.render(delta)
        jpeg=export_jpeg(image)
        variants=make_variants(jpeg,box)
        if arm=="refresh_edited_landmarks" and (step-1)%CHECKPOINT_EVERY==0:
            current_points={name:detect_points(models["SFace"].detector,variants[name]) for name in OBJECTIVE}
        terms=[];surrogate={}
        for name in OBJECTIVE:
            variant=variants[name]
            if variant.status!="valid":
                raise RuntimeError(f"Condition {name}: {variant.reason}")
            points=current_points[name]
            offset=crop_offset(name,source.shape)
            s_matrix=affine(points)
            g_matrix=ghost_matrix(points)
            s_grid=warp_grid(carrier,np.vstack((s_matrix,[0,0,1])),offset)
            g_grid=warp_grid(carrier,g_matrix,offset)
            s_delta=F.grid_sample(delta.unsqueeze(0),s_grid,mode="bilinear",padding_mode="zeros",align_corners=True)[0]
            g_delta=F.grid_sample(delta.unsqueeze(0),g_grid,mode="bilinear",padding_mode="zeros",align_corners=True)[0]
            s_crop=aligned_rgb(variant.image,s_matrix)
            s_exact=torch.from_numpy(np.ascontiguousarray(s_crop.transpose(2,0,1).astype(np.float32)))
            s_proxy=s_exact+s_delta-s_delta.detach()
            s_raw=torch_sface(s_proxy.unsqueeze(0)).reshape(-1)
            s_score=F.cosine_similarity(s_raw,source_sface,dim=0)
            terms.append((s_score-thresholds["SFace"])/(1-thresholds["SFace"]))
            surrogate[f"SFace:{name}"]=float(s_score.detach())
            g_crop=author_align(variant.image,points)
            g_exact=torch.from_numpy(np.ascontiguousarray(g_crop.transpose(2,0,1).astype(np.float32)))
            g_proxy=g_exact+g_delta-g_delta.detach()
            g_value,g_score=ghost_term(tf_gradient,g_crop,clean_features["GhostFaceNet"],
                                       thresholds["GhostFaceNet"],g_proxy)
            terms.append(g_value)
            surrogate[f"GhostFaceNet:{name}"]=g_score
        loss=torch.stack(terms).max()
        loss.backward()
        optimizer.step()
        if step%CHECKPOINT_EVERY==0:
            actual,rows=native_objective(jpeg,box,models,clean_features,thresholds)
            row={"step":step,"arm":arm,"native_worst_normalized_margin_or_failure_one":actual,
                "native":rows,"surrogate":surrogate,"surrogate_worst_margin":float(loss.detach()),
                "pre_jpeg_face_rms":float(torch.sqrt(torch.mean(delta.detach()[:,carrier.face]**2))),
                "pre_jpeg_max_channel_delta":float(delta.detach().abs().max()),
                "jpeg_sha256":hashlib.sha256(jpeg).hexdigest()}
            log.append(row)
            if best is None or actual<best["score"]:
                best={"step":step,"score":actual,"jpeg":jpeg,"row":row}
    best["seconds"]=perf_counter()-t0
    return best,log


def main():
    import tensorflow as tf
    torch.set_num_threads(2);cv2.setNumThreads(2)
    if OUT.exists():
        raise FileExistsError(OUT)
    prior=DATA/"runs"/"ensemble-dots-v1"/"ghost-float32-clone-diagnostic.json"
    diag=json.loads(prior.read_text())
    if (not diag["gradient_finite"] or any(row["unit_cosine"]<.999 or row["unit_max_abs_difference"]>.02
        for row in diag["parity"].values()) or not any(row["relative_error"]<=.1 or row["absolute_error"]<=1e-6
        for row in diag["unit_l2_finite_differences"])):
        raise ValueError("H6 clone gradient/parity preflight not accepted")
    manifest=DATA/"frll"/"manifest.json"
    groups=grouped_images(manifest,"development")
    s_path=DATA/"runs"/"calibration-sface-v1"/"calibration.json"
    g_path=DATA/"runs"/"calibration-ghostface-v1"/"calibration.json"
    s_cal=json.loads(s_path.read_text());g_cal=json.loads(g_path.read_text())
    yunet=WEIGHTS/"face_detection_yunet_2023mar.onnx"
    sface=WEIGHTS/"face_recognition_sface_2021dec.onnx"
    ghost=WEIGHTS/"ghostfacenet_v1.h5"
    for cal,name,path in ((s_cal,"yunet",yunet),(s_cal,"sface",sface),(g_cal,"yunet",yunet),(g_cal,"ghostface",ghost)):
        if digest(path)!=cal["pipeline"]["model_artifacts"][name]["sha256"]:
            raise ValueError(f"{name} calibrated model hash differs")
    for cal in (s_cal,g_cal):
        for name,sha in cal["pipeline"]["source_sha256"].items():
            if name.startswith("fckface_lab/") and digest(Path(__file__).parent/name)!=sha:
                raise ValueError(f"Calibrated pipeline changed: {name}")
    thresholds={"SFace":Calibration(**s_cal["calibration"]).threshold,
                "GhostFaceNet":Calibration(**g_cal["calibration"]).threshold}
    models={"SFace":SFaceModel(yunet,sface),"GhostFaceNet":GhostFaceModel(yunet,ghost)}
    clone=load_float32_clone(models["GhostFaceNet"].model)
    tf_gradient,tf_report=tf_value_gradient_functions(clone)
    torch_sface=convert(onnx.load(str(sface))).eval()
    for parameter in torch_sface.parameters():parameter.requires_grad_(False)
    OUT.mkdir(parents=True)
    shutil.copyfile(Path(__file__),OUT/"executed-source.py")
    header={"hypothesis":"H7 refreshed edited-condition landmarks improve dot robustness",
        "identities":IDS,"arms":ARMS,"steps_each":STEPS,"checkpoint_every":CHECKPOINT_EVERY,
        "gradient_model_condition_evaluations_each":STEPS*len(OBJECTIVE)*2,
        "seed":0,"target_face_rms":RMS,"max_channel_delta":MAX_DELTA,
        "objective_conditions":OBJECTIVE,"final_conditions":CONDITIONS,"thresholds":thresholds,
        "refresh_schedule":"B refreshes edited YuNet landmarks at steps 1,4,7,10,13,16; stop-gradient; A keeps per-condition clean landmarks",
        "crop90_proxy":"inverse alignment coordinates in crop image + round(original W*.05), round(original H*.05) to original-image ROI",
        "tf_function_preflight":tf_report,"tf_gradient_mode":"compiled" if tf_report["compiled_accepted"] else "eager",
        "source_sha256":{"script":digest(Path(__file__)),"carrier":digest(Path(__file__).parent/"optimize_gradient_art.py"),
            "h6_clone":digest(Path(__file__).parent/"optimize_ensemble_dots.py"),
            "manifest":digest(manifest),"sface_calibration":digest(s_path),"ghost_calibration":digest(g_path),
            "yunet":digest(yunet),"sface":digest(sface),"ghostface":digest(ghost),
            "clone_diagnostic":digest(prior)}}
    write_json(OUT/"header.json",header)
    records=[];frozen=[];t0=perf_counter()
    try:
        for identity in IDS:
            source_path=groups[identity]["neutral_front"]
            source=decode_image(source_path)
            clean={name:models[name].embed(source) for name in ("SFace","GhostFaceNet")}
            if any(value.status!="valid" for value in clean.values()):
                raise RuntimeError(f"{identity} clean source invalid")
            box=clean["SFace"].selected_box
            if box!=clean["GhostFaceNet"].selected_box:
                raise ValueError("Clean model boxes disagree")
            clean_features={name:clean[name].feature for name in clean}
            clean_variants=make_variants(export_jpeg(source),box)
            clean_landmarks={name:detect_points(models["SFace"].detector,clean_variants[name]) for name in OBJECTIVE}
            case={"identity":identity,"source_sha256":digest(source_path),"arms":{}}
            for arm in ARMS:
                best,log=optimize(source,box,clean_landmarks,models,torch_sface,tf_gradient,thresholds,clean_features,arm)
                directory=OUT/identity/arm
                directory.mkdir(parents=True)
                selected=directory/"selected.jpg"
                selected.write_bytes(best["jpeg"])
                write_json(directory/"checkpoints.json",log)
                case["arms"][arm]={"selected_step":best["step"],"native_worst_margin":best["score"],
                    "selected_jpeg_sha256":digest(selected),"seconds":best["seconds"],
                    "gradient_model_condition_evaluations":STEPS*len(OBJECTIVE)*2,
                    "native_checkpoints":len(log),"native_model_condition_queries":len(log)*len(OBJECTIVE)*2,
                    "pre_jpeg_face_rms":best["row"]["pre_jpeg_face_rms"],
                    "pre_jpeg_max_channel_delta":best["row"]["pre_jpeg_max_channel_delta"]}
                frozen.append((identity,arm,selected,box,source))
                print(identity,arm,"frozen",best["step"],best["score"],flush=True)
            records.append(case)
        write_json(OUT/"frozen.json",{"header":header,"cases":records,
            "all_outputs_frozen_before_gallery":True})
        # Gallery is first read after every selected JPEG and its hash are fixed.
        for identity in IDS:
            refs={name:[] for name in ("SFace","GhostFaceNet")}
            for view in VIEWS:
                image=decode_image(groups[identity][view])
                for name in refs:
                    result=models[name].embed(image)
                    if result.status=="valid":
                        refs[name].append(Reference(f"{identity}:{view}",identity,result.feature))
            case=next(row for row in records if row["identity"]==identity)
            if any(len(rows)!=len(VIEWS) for rows in refs.values()):
                raise RuntimeError(f"{identity} incomplete gallery")
            box=next(row[3] for row in frozen if row[0]==identity)
            source=decode_image(groups[identity]["neutral_front"])
            clean_variants=make_variants(export_jpeg(source),box)
            controls={name:{} for name in refs}
            for name in refs:
                calibration=Calibration(**(s_cal if name=="SFace" else g_cal)["calibration"])
                for condition in CONDITIONS:
                    variant=clean_variants[condition]
                    embedded=models[name].embed(variant.image,expected_box=variant.target_box)
                    gallery=evaluate_gallery(embedded,refs[name],calibration,own_identity=identity,
                        own_source_name=f"{identity}:neutral_front",
                        expected_reference_names=[f"{identity}:{view}" for view in VIEWS])
                    controls[name][condition]={"status":gallery.status,
                        "own_identity_matched":gallery.own_identity_matched}
                    if gallery.status!="valid" or gallery.own_identity_matched is not True:
                        raise RuntimeError(f"{identity}/{name}/{condition} clean control ineligible")
            case["clean_controls"]=controls
            for who,arm,path,box,source in frozen:
                if who!=identity:continue
                variants=make_variants(path.read_bytes(),box)
                final={name:{} for name in refs}
                for name in refs:
                    calibration=Calibration(**(s_cal if name=="SFace" else g_cal)["calibration"])
                    for condition in CONDITIONS:
                        variant=variants[condition]
                        if variant.status!="valid":
                            final[name][condition]={"status":"inconclusive","reason":variant.reason,"nonmatch":False}
                            continue
                        embedded=models[name].embed(variant.image,expected_box=variant.target_box)
                        gallery=evaluate_gallery(embedded,refs[name],calibration,own_identity=identity,
                            own_source_name=f"{identity}:neutral_front",
                            expected_reference_names=[f"{identity}:{view}" for view in VIEWS])
                        final[name][condition]={"status":gallery.status,"reason":gallery.reason,
                            "maximum_cosine":max(gallery.per_reference.values()) if gallery.per_reference else None,
                            "own_source_cosine":gallery.own_source_score,
                            "matching_references":gallery.matching_references,"nonmatch":gallery.nonmatch}
                write_json(path.parent/"evaluation.json",final)
                case["arms"][arm]["evaluation"]=final
        manifest_rows={(row["identity"],row["view"]):row["sha256"]
                       for row in json.loads(manifest.read_text())["images"]}
        artifact_audit={"runner_sha256":digest(OUT/"executed-source.py"),"identities":{}}
        for case in records:
            identity=case["identity"]
            refs={view:{"path":str(groups[identity][view]),"sha256":digest(groups[identity][view])}
                  for view in VIEWS}
            if any(row["sha256"]!=manifest_rows[(identity,view)] for view,row in refs.items()):
                raise ValueError(f"{identity} source/gallery image hash changed")
            source=decode_image(groups[identity]["neutral_front"])
            box=next(row[3] for row in frozen if row[0]==identity)
            clean_variants=make_variants(export_jpeg(source),box)
            artifact_audit["identities"][identity]={"source":refs["neutral_front"],
                "gallery":refs,"arms":{}}
            for who,arm,path,_,_ in frozen:
                if who!=identity:continue
                if digest(path)!=case["arms"][arm]["selected_jpeg_sha256"]:
                    raise ValueError("Frozen selected JPEG changed")
                variants=make_variants(path.read_bytes(),box)
                directory=OUT/"condition-jpegs"/identity/arm
                directory.mkdir(parents=True)
                artifacts={};rms=[]
                for condition in CONDITIONS:
                    variant=variants[condition]
                    if variant.status!="valid" or variant.jpeg is None:
                        raise RuntimeError(f"{identity}/{arm}/{condition}: {variant.reason}")
                    target=directory/f"{condition}.jpg"
                    target.write_bytes(variant.jpeg)
                    clean=clean_variants[condition]
                    distortion=actual_distortion(clean.image,variant.image,
                        (variant.target_box[0],variant.target_box[1],
                         variant.target_box[2]-variant.target_box[0],
                         variant.target_box[3]-variant.target_box[1]))
                    rms.append(distortion["face_rms"])
                    artifacts[condition]={"path":str(target),"sha256":digest(target),
                        "bytes":len(variant.jpeg),"post_jpeg_face_rms":distortion["face_rms"]}
                if artifacts["export"]["sha256"]!=case["arms"][arm]["selected_jpeg_sha256"]:
                    raise ValueError("Export condition does not equal frozen selected JPEG")
                artifact_audit["identities"][identity]["arms"][arm]=artifacts
                case["arms"][arm]["post_jpeg_export_face_rms"]=artifacts["export"]["post_jpeg_face_rms"]
                case["arms"][arm]["post_jpeg_face_rms_range"]=[min(rms),max(rms)]
        write_json(OUT/"artifact-audit.json",artifact_audit)
        write_json(OUT/"results.json",{"status":"complete","header":header,
            "wall_seconds":perf_counter()-t0,"cases":records})
        print("H7 complete",OUT,flush=True)
    except Exception as exc:
        write_json(OUT/"failure.json",{"type":type(exc).__name__,"message":str(exc),
            "frozen_count":len(frozen)})
        raise


if __name__=="__main__":
    main()
