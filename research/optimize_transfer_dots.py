"""H10: bounded optimizer/input-diversity transfer ablation on regular dots.

Development only. Three arms share one untransformed native clean source
embedding per model, one dot carrier, 18 steps, and 144 edited model-condition
gradient forwards. DI changes only the edited crop seen during a gradient step.
Step-18 exports freeze unconditionally before any other same-person view is read.
Native checkpoints diagnose validity; they never select an image. ArcFace is
scored separately from the frozen six-JPEG manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

IDS=("frll-029","frll-030")
VIEWS=("neutral_front","smiling_front","neutral_left_3quarter","neutral_right_3quarter")
OBJECTIVE=("export","jpeg75_420","blur","crop90")
CONDITIONS=("export","jpeg85_420","jpeg75_420","resize960","half_restore","crop90","blur")
ARMS=("adam","momentum","momentum_di")
STEPS=18
CHECKPOINT_EVERY=3
DI_SEED=20260925
COEFFICIENT_STEP=.15
MOMENTUM=1.0


def write_json(path:Path,value:object)->None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def diversity_schedule()->dict[str,dict[str,dict[str,int|bool]]]:
    """Freeze all random choices once, before loading or scoring a photo."""
    rng=np.random.default_rng(DI_SEED)
    rows={}
    for step in range(1,STEPS+1):
        rows[str(step)]={}
        for condition in OBJECTIVE:
            if rng.random()<.5:
                entry={"identity":True,"side":112,"left":0,"top":0}
            else:
                side=int(rng.integers(101,112))
                entry={"identity":False,"side":side,
                       "left":int(rng.integers(0,113-side)),
                       "top":int(rng.integers(0,113-side))}
            rows[str(step)][condition]=entry
    return rows


def transform_crop(chw,entry:dict):
    """Differentiable bilinear resize and zero-RGB pad; output remains 112²."""
    import torch.nn.functional as F
    if tuple(chw.shape)!=(3,112,112):
        raise ValueError("Expected 3x112x112 aligned RGB crop")
    if entry["identity"]:
        return chw
    side,left,top=(entry[key] for key in ("side","left","top"))
    if not (101<=side<=111 and 0<=left<=112-side and 0<=top<=112-side):
        raise ValueError("Invalid frozen DI geometry")
    resized=F.interpolate(chw.unsqueeze(0),size=(side,side),mode="bilinear",
                          align_corners=False)[0]
    return F.pad(resized,(left,112-side-left,top,112-side-top),mode="constant",value=0.)


def verify_inputs(args):
    from fckface_lab.calibration import Calibration
    from fckface_lab.datasets import digest,grouped_images
    repo=Path(__file__).resolve().parents[1]
    if args.out.exists() or args.out.resolve().is_relative_to(repo):
        raise ValueError("Output must be a new external directory")
    manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    groups=grouped_images(args.manifest,"development")
    if any(identity not in groups or not set(VIEWS).issubset(groups[identity])
           for identity in IDS):
        raise ValueError("Both frozen H10 identities need four development views")
    expected={(row["identity"],row["view"]):row["sha256"]
              for row in manifest["images"] if row["split"]=="development"}
    calibrations={"SFace":json.loads(args.sface_calibration.read_text(encoding="utf-8")),
                  "GhostFaceNet":json.loads(args.ghostface_calibration.read_text(encoding="utf-8"))}
    model_paths={"yunet":args.yunet,"sface":args.sface,"ghostface":args.ghostface}
    for name,cal in calibrations.items():
        for model_name in (("yunet","sface") if name=="SFace" else ("yunet","ghostface")):
            record=cal["pipeline"]["model_artifacts"][model_name]
            if digest(model_paths[model_name])!=record["sha256"] or model_paths[model_name].stat().st_size!=record["bytes"]:
                raise ValueError(f"Calibrated model mismatch: {name}/{model_name}")
        if digest(args.manifest)!=cal["dataset"]["manifest_sha256"]:
            raise ValueError(f"Manifest mismatch with {name} calibration")
        for source,sha in cal["pipeline"]["source_sha256"].items():
            if source.startswith("fckface_lab/") and digest(Path(__file__).parent/source)!=sha:
                raise ValueError(f"Calibrated pipeline changed: {source}")
    diagnostic=json.loads(args.clone_diagnostic.read_text(encoding="utf-8"))
    if (not diagnostic["gradient_finite"] or
        any(row["unit_cosine"]<.999 or row["unit_max_abs_difference"]>.02
            for row in diagnostic["parity"].values()) or
        not any(row["relative_error"]<=.1 or row["absolute_error"]<=1e-6
                for row in diagnostic["unit_l2_finite_differences"])):
        raise ValueError("H6 float32 Ghost clone diagnostic failed")
    thresholds={name:Calibration(**cal["calibration"]).threshold
                for name,cal in calibrations.items()}
    return groups,expected,model_paths,thresholds


def surrogate_loss(source,box,carrier,delta,points_by_condition,
                   torch_sface,ghost_gradient,native_clean,thresholds,
                   di_schedule:dict,arm:str):
    import torch
    import torch.nn.functional as F
    from fckface_lab.ghostface import author_align
    from fckface_lab.imaging import export_jpeg,make_variants
    from optimize_alignment_dots import crop_offset,ghost_matrix,warp_grid,ghost_term
    from optimize_gradient_art import affine,aligned_rgb

    jpeg=export_jpeg(carrier.render(delta))
    variants=make_variants(jpeg,box)
    source_s=torch.from_numpy(native_clean["SFace"].astype(np.float32))
    terms=[];cosines={}
    for condition in OBJECTIVE:
        variant=variants[condition]
        if variant.status!="valid" or variant.image is None:
            raise RuntimeError(f"Invalid surrogate condition {condition}: {variant.reason}")
        points=points_by_condition[condition]
        offset=crop_offset(condition,source.shape)
        sm=affine(points)
        sg=warp_grid(carrier,np.vstack((sm,[0,0,1])),offset)
        gg=warp_grid(carrier,ghost_matrix(points),offset)
        sd=F.grid_sample(delta.unsqueeze(0),sg,mode="bilinear",padding_mode="zeros",align_corners=True)[0]
        gd=F.grid_sample(delta.unsqueeze(0),gg,mode="bilinear",padding_mode="zeros",align_corners=True)[0]
        # Exact exported condition pixels in the forward pass, fixed-affine
        # delta in the backward pass. DI transforms the edited crop only.
        sc=aligned_rgb(variant.image,sm)
        se=torch.from_numpy(np.ascontiguousarray(sc.transpose(2,0,1).astype(np.float32)))
        sp=se+sd-sd.detach()
        geom=(di_schedule[condition] if arm=="momentum_di"
              else {"identity":True,"side":112,"left":0,"top":0})
        sp=transform_crop(sp,geom)
        s_raw=torch_sface(sp.unsqueeze(0)).reshape(-1)
        s_score=F.cosine_similarity(s_raw,source_s,dim=0)
        terms.append((s_score-thresholds["SFace"])/(1-thresholds["SFace"]))
        cosines[f"SFace:{condition}"]=float(s_score.detach())
        gc=author_align(variant.image,points)
        ge=torch.from_numpy(np.ascontiguousarray(gc.transpose(2,0,1).astype(np.float32)))
        gp=transform_crop(ge+gd-gd.detach(),geom)
        # The TF VJP is with respect to the transformed crop. Its injection at
        # gp lets Torch differentiate resize/pad and the Ghost alignment grid.
        exact=gp.detach().permute(1,2,0).numpy()
        g_term,g_score=ghost_term(ghost_gradient,exact,native_clean["GhostFaceNet"],
                                  thresholds["GhostFaceNet"],gp)
        terms.append(g_term)
        cosines[f"GhostFaceNet:{condition}"]=g_score
    loss=torch.stack(terms).max()
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("Nonfinite surrogate margin")
    return loss,jpeg,cosines


def optimize(source,box,points_by_condition,models,torch_sface,ghost_gradient,
             native_clean,thresholds,arm,schedule,checkpoint_dir:Path):
    import torch
    from optimize_gradient_art import Carrier,affine
    from optimize_alignment_dots import native_objective

    carrier=Carrier(source,box,affine(points_by_condition["export"]),"dots_14x14")
    checkpoint_dir.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(0)
    param=torch.nn.Parameter(torch.from_numpy(rng.standard_normal((3,14,14)).astype(np.float32)*.1))
    optimizer=torch.optim.Adam([param],lr=COEFFICIENT_STEP) if arm=="adam" else None
    velocity=torch.zeros_like(param)
    rows=[];chosen=None;start=perf_counter()
    for step in range(1,STEPS+1):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            param.grad=None
        delta=carrier.project(param)
        loss,jpeg,cosines=surrogate_loss(source,box,carrier,delta,points_by_condition,
            torch_sface,ghost_gradient,native_clean,thresholds,schedule[str(step)],arm)
        loss.backward()
        gradient=param.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"Missing or nonfinite coefficient gradient at step {step}")
        mean_abs=float(gradient.detach().abs().mean())
        if not np.isfinite(mean_abs) or mean_abs<=0:
            raise RuntimeError(f"Zero coefficient gradient at step {step}")
        if optimizer is not None:
            optimizer.step()
        else:
            with torch.no_grad():
                velocity.mul_(MOMENTUM).add_(gradient/mean_abs)
                if not bool(torch.isfinite(velocity).all()):
                    raise RuntimeError(f"Nonfinite momentum at step {step}")
                param.sub_(COEFFICIENT_STEP*velocity.sign())
        if step%CHECKPOINT_EVERY==0:
            native_score,native=native_objective(jpeg,box,models,native_clean,thresholds)
            valid=all(native[name][condition]["status"]=="valid"
                      for name in ("SFace","GhostFaceNet") for condition in OBJECTIVE)
            row={"step":step,"arm":arm,"selected_only_if_step18":step==STEPS,
                 "native_all_conditions_valid":valid,
                 "native_worst_normalized_margin_or_failure_one":native_score,
                 "native":native,"surrogate_worst_margin":float(loss.detach()),
                 "surrogate_source_cosines":cosines,
                 "pre_jpeg_face_rms":float(torch.sqrt(torch.mean(delta.detach()[:,carrier.face]**2))),
                 "pre_jpeg_max_channel_delta":float(delta.detach().abs().max()),
                 "jpeg_sha256":hashlib.sha256(jpeg).hexdigest()}
            rows.append(row)
            (checkpoint_dir/f"step{step:02d}.jpg").write_bytes(jpeg)
            write_json(checkpoint_dir/f"step{step:02d}.json",row)
            if step==STEPS:
                chosen={"jpeg":jpeg,"row":row,"step":step}
    if chosen is None:
        raise RuntimeError("Fixed forward step18 was not exported")
    chosen["seconds"]=perf_counter()-start
    return chosen,rows


def summarize(rows:dict)->dict:
    valid=len(rows)==len(CONDITIONS) and all(row["status"]=="valid" for row in rows.values())
    return {"valid_all":valid,
            "nonmatch_all":valid and all(row.get("own_identity_matched") is False for row in rows.values()),
            "valid_conditions":sum(row["status"]=="valid" for row in rows.values()),
            "invalid_or_inconclusive":[name for name,row in rows.items() if row["status"]!="valid"],
            "worst_gallery_cosine":max((row["maximum_cosine"] for row in rows.values()
                                        if row.get("maximum_cosine") is not None),default=None)}


def score_native_after_freeze(output:Path,cases:list,groups:dict,expected:dict,
                              models:dict,calibrations:dict)->None:
    """Read extra same-person views only after verifying all six frozen bytes."""
    from fckface_lab.calibration import Calibration
    from fckface_lab.datasets import digest
    from fckface_lab.evaluation import Reference
    from fckface_lab.imaging import decode_image,export_jpeg
    from optimize_regions import evaluate_conditions

    manifest=output/"frozen-inputs.json"
    frozen=json.loads(manifest.read_text(encoding="utf-8"))["items"]
    if len(frozen)!=len(IDS)*len(ARMS):
        raise ValueError("Incomplete frozen manifest; refusing gallery access")
    for row in frozen:
        if digest(Path(row["path"]))!=row["sha256"]:
            raise ValueError("Frozen JPEG hash differs before gallery access")
    thresholds={name:Calibration(**cal["calibration"]) for name,cal in calibrations.items()}
    report={"status":"in_progress","purpose":"H10 frozen development native scoring",
            "frozen_manifest_sha256":digest(manifest),"conditions":CONDITIONS,"views":VIEWS,
            "identities":{}}
    for case in cases:
        identity=case["identity"]
        photos={}
        hashes={}
        for view in VIEWS:
            path=groups[identity][view]
            sha=digest(path)
            if sha!=expected[(identity,view)]:
                raise ValueError(f"Gallery image differs from manifest: {identity}/{view}")
            photos[view]=decode_image(path)
            hashes[view]=sha
        person={"gallery_sha256":hashes,"models":{}}
        for name,model in models.items():
            refs=[];reference_status={}
            for view,image in photos.items():
                embedded=model.embed(image)
                reference_status[view]={"status":embedded.status,"reason":embedded.reason}
                if embedded.status=="valid":
                    refs.append(Reference(f"{identity}:{view}",identity,embedded.feature))
            root=output/"native-seven-condition"/identity/name
            control=evaluate_conditions(export_jpeg(photos["neutral_front"]),case["box"],
                model,refs,thresholds[name],identity,CONDITIONS,save_dir=root/"control")
            eligible=len(refs)==len(VIEWS) and all(
                row["status"]=="valid" and row.get("own_identity_matched") is True
                for row in control.values())
            record={"references":reference_status,"clean_eligible":eligible,
                    "control":control,"control_summary":summarize(control),"arms":{}}
            for arm in ARMS:
                chosen=next(row for row in frozen if row["identity"]==identity and row["arm"]==arm)
                candidate=evaluate_conditions(Path(chosen["path"]).read_bytes(),case["box"],
                    model,refs,thresholds[name],identity,CONDITIONS,save_dir=root/arm)
                record["arms"][arm]={"candidate":candidate,"summary":summarize(candidate),
                    "clean_eligible":eligible,"selected_sha256":chosen["sha256"]}
            person["models"][name]=record
        report["identities"][identity]=person
        write_json(output/"native-seven-condition-progress.json",report)
        print(identity,"native seven-condition scored",flush=True)
    report["status"]="complete"
    write_json(output/"native-seven-condition.json",report)


def distortion_after_freeze(output:Path,cases:list)->None:
    from fckface_lab.datasets import digest
    from fckface_lab.imaging import decode_image,export_jpeg,make_variants
    from fckface_lab.patterns import actual_distortion
    rows={}
    for case in cases:
        identity=case["identity"]
        source=decode_image(Path(case["source_path"]))
        clean=make_variants(export_jpeg(source),case["box"])
        rows[identity]={}
        for arm in ARMS:
            path=Path(case["arms"][arm]["path"])
            if digest(path)!=case["arms"][arm]["selected_sha256"]:
                raise ValueError("Selected JPEG changed before distortion measurement")
            edited=make_variants(path.read_bytes(),case["box"])
            metrics={}
            for condition in CONDITIONS:
                a,b=clean[condition],edited[condition]
                if a.status!="valid" or b.status!="valid":
                    metrics[condition]={"status":"inconclusive","reason":a.reason or b.reason}
                else:
                    x0,y0,x1,y1=b.target_box
                    metrics[condition]={"status":"valid",**actual_distortion(
                        a.image,b.image,(x0,y0,x1-x0,y1-y0)),
                        "candidate_jpeg_sha256":hashlib.sha256(b.jpeg).hexdigest()}
            rows[identity][arm]=metrics
    write_json(output/"distortion.json",{"basis":"clean and edited exact JPEG condition pixels",
                                         "identities":rows})


def run(args)->None:
    import cv2
    import onnx
    import torch
    from onnx2torch import convert
    from fckface_lab.datasets import digest
    from fckface_lab.imaging import decode_image,export_jpeg,make_variants
    from fckface_lab.recognition import SFaceModel
    from fckface_lab.ghostface import GhostFaceModel
    from optimize_alignment_dots import detect_points,tf_value_gradient_functions
    from optimize_ensemble_dots import load_float32_clone

    groups,expected,model_paths,thresholds=verify_inputs(args)
    cv2.setNumThreads(2);torch.set_num_threads(2)
    schedule=diversity_schedule()
    args.out.mkdir(parents=True,exist_ok=False)
    snapshots=args.out/"source-snapshot";snapshots.mkdir()
    source_hashes={}
    for name in ("optimize_transfer_dots.py","optimize_alignment_dots.py",
                 "optimize_gradient_art.py","optimize_ensemble_dots.py","optimize_regions.py"):
        path=Path(__file__).parent/name
        shutil.copyfile(path,snapshots/name)
        source_hashes[name]=digest(path)
        if digest(snapshots/name)!=source_hashes[name]:
            raise ValueError(f"Source changed during snapshot: {name}")
    write_json(args.out/"diversity-schedule.json",schedule)
    calibration_records={"SFace":json.loads(args.sface_calibration.read_text(encoding="utf-8")),
                         "GhostFaceNet":json.loads(args.ghostface_calibration.read_text(encoding="utf-8"))}
    pinned_dir=args.out/"model-inputs";pinned_dir.mkdir()
    pinned={}
    for name,path in model_paths.items():
        target=pinned_dir/path.name
        shutil.copyfile(path,target)
        reference=("GhostFaceNet" if name=="ghostface" else "SFace")
        artifact=calibration_records[reference]["pipeline"]["model_artifacts"][name]
        if digest(target)!=artifact["sha256"] or target.stat().st_size!=artifact["bytes"]:
            raise ValueError(f"Private model copy differs from calibrated artifact: {name}")
        pinned[name]=target
    calibrations=calibration_records
    protocol={"hypothesis":"H10 optimizer/input-diversity transfer on fixed artwork",
              "development_only":True,"identities":IDS,"arms":ARMS,"steps":STEPS,
              "checkpoint_every":CHECKPOINT_EVERY,"seed":0,"diversity_seed":DI_SEED,
              "diversity_schedule_sha256":digest(args.out/"diversity-schedule.json"),
              "diversity":"p=.5 identity, else integer resize 101..111 and random zero-RGB pad to 112; identical geometry for both edited-model crops; clean native source embeddings untransformed in ALL arms",
              "momentum":MOMENTUM,"coefficient_sign_step":COEFFICIENT_STEP,
              "objective_conditions":OBJECTIVE,"final_conditions":CONDITIONS,
              "gradient_model_condition_forwards_per_arm":STEPS*len(OBJECTIVE)*2,
              "native_checkpoint_model_condition_queries_per_arm":(STEPS//CHECKPOINT_EVERY)*len(OBJECTIVE)*2,
              "selection":"unconditional forward step18 JPEG; native checkpoints diagnostics only",
              "carrier":"H5 dots_14x14, face RMS16, channel cap64",
              "prediction":"DI median ArcFace worst-gallery cosine reduction >=.02 versus each control, positive improvement for both people, all 7 conditions valid; otherwise retire unchanged",
              "gallery_barrier":"six selected JPEGs and SHA-256 manifest fixed before additional same-person views",
              "manifest_sha256":digest(args.manifest),
              "calibration_sha256":{"SFace":digest(args.sface_calibration),
                                    "GhostFaceNet":digest(args.ghostface_calibration)},
              "model_sha256":{name:digest(path) for name,path in pinned.items()},
              "clone_diagnostic_sha256":digest(args.clone_diagnostic),
              "source_sha256":source_hashes,"thresholds":thresholds}
    write_json(args.out/"protocol.json",protocol)
    models={"SFace":SFaceModel(pinned["yunet"],pinned["sface"]),
            "GhostFaceNet":GhostFaceModel(pinned["yunet"],pinned["ghostface"])}
    clone=load_float32_clone(models["GhostFaceNet"].model)
    gradient,tf_report=tf_value_gradient_functions(clone)
    write_json(args.out/"ghost-gradient-preflight.json",tf_report)
    if not tf_report["compiled_accepted"]:
        raise RuntimeError("H7 compiled Ghost gradient did not pass parity preflight")
    torch_sface=convert(onnx.load(str(pinned["sface"]))).eval()
    for parameter in torch_sface.parameters():parameter.requires_grad_(False)
    cases=[];frozen=[];start=perf_counter()
    try:
        for identity in IDS:
            path=groups[identity]["neutral_front"]
            if digest(path)!=expected[(identity,"neutral_front")]:
                raise ValueError(f"Source differs from manifest: {identity}")
            source=decode_image(path)
            native={name:model.embed(source) for name,model in models.items()}
            if any(item.status!="valid" for item in native.values()):
                raise RuntimeError(f"Clean neutral_front invalid: {identity}")
            box=native["SFace"].selected_box
            if box!=native["GhostFaceNet"].selected_box:
                raise ValueError("Clean model target boxes disagree")
            native_clean={name:item.feature for name,item in native.items()}
            clean_conditions=make_variants(export_jpeg(source),box)
            points={condition:detect_points(models["SFace"].detector,clean_conditions[condition])
                    for condition in OBJECTIVE}
            case={"identity":identity,"source_path":str(path),"source_sha256":digest(path),
                  "box":box,"arms":{}}
            for arm in ARMS:
                directory=args.out/identity/arm;directory.mkdir(parents=True)
                chosen,checkpoints=optimize(source,box,points,models,torch_sface,gradient,
                    native_clean,thresholds,arm,schedule,directory/"checkpoints")
                selected=directory/"selected.jpg";selected.write_bytes(chosen["jpeg"])
                write_json(directory/"checkpoints.json",checkpoints)
                case["arms"][arm]={"selected_step":18,"path":str(selected),
                    "selected_sha256":digest(selected),"seconds":chosen["seconds"],
                    "native_final_all_conditions_valid":chosen["row"]["native_all_conditions_valid"],
                    "pre_jpeg_face_rms":chosen["row"]["pre_jpeg_face_rms"],
                    "pre_jpeg_max_channel_delta":chosen["row"]["pre_jpeg_max_channel_delta"]}
                frozen.append({"identity":identity,"method":"H10","arm":arm,
                    "path":str(selected),"sha256":digest(selected),"source_view":"neutral_front"})
                print(identity,arm,"fixed-step frozen",chosen["row"]["native_all_conditions_valid"],flush=True)
            cases.append(case)
        if len(frozen)!=6:
            raise RuntimeError("Incomplete six-JPEG freeze")
        write_json(args.out/"frozen-inputs.json",{"schema_version":1,"items":frozen})
        write_json(args.out/"freeze-results.json",{"status":"six_outputs_frozen",
            "protocol_sha256":digest(args.out/"protocol.json"),
            "frozen_manifest_sha256":digest(args.out/"frozen-inputs.json"),
            "optimization_seconds_excluding_load":perf_counter()-start,"cases":cases})
        distortion_after_freeze(args.out,cases)
        # Only now may the three other same-person views be opened.
        score_native_after_freeze(args.out,cases,groups,expected,models,calibrations)
    except Exception as exc:
        write_json(args.out/"failure.json",{"type":type(exc).__name__,"message":str(exc),
                                              "frozen_count":len(frozen)})
        raise


def synthetic_di_chain_check()->dict:
    """Finite difference the Torch resize/pad and TF-to-Torch crop VJP."""
    import torch
    import tensorflow as tf
    rng=np.random.default_rng(29)
    entry={"identity":False,"side":105,"left":4,"top":3}
    base=torch.from_numpy(rng.normal(size=(3,112,112)).astype(np.float64))
    direction=torch.from_numpy(rng.normal(size=(3,112,112)).astype(np.float64))
    weight=torch.from_numpy(rng.normal(size=(3,112,112)).astype(np.float64))
    coefficient=torch.tensor(.17,dtype=torch.float64,requires_grad=True)
    transformed=transform_crop(base+coefficient*direction,entry)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tf_pixel=tf.constant(transformed.detach().numpy())
        tape.watch(tf_pixel)
        tf_value=tf.reduce_sum(tf_pixel*tf.constant(weight.numpy()))
    tf_gradient=tape.gradient(tf_value,tf_pixel)
    if tf_gradient is None or not np.isfinite(tf_gradient.numpy()).all():
        raise AssertionError("Synthetic TensorFlow crop VJP invalid")
    injected=torch.from_numpy(tf_gradient.numpy())
    # Mirror ghost_term: external value plus differentiable Torch VJP injection.
    value=torch.tensor(float(tf_value.numpy()))+((transformed-transformed.detach())*injected).sum()
    value.backward()
    analytic=float(coefficient.grad)
    epsilon=1e-5
    def scalar(v):
        return float((transform_crop(base+v*direction,entry)*weight).sum())
    numeric=(scalar(.17+epsilon)-scalar(.17-epsilon))/(2*epsilon)
    error=abs(analytic-numeric)
    if not np.isfinite(error) or error>1e-6:
        raise AssertionError(f"DI resize/pad chained VJP failed finite difference: {error}")
    unchanged=torch.equal(transform_crop(base,{"identity":True,"side":112,"left":0,"top":0}),base)
    if not unchanged:
        raise AssertionError("DI identity branch changed crop pixels")
    result={"synthetic_seed":29,"geometry":entry,"epsilon":epsilon,
            "analytic_directional_gradient":analytic,"numeric_directional_gradient":numeric,
            "absolute_error":error,"identity_exact":unchanged}
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest",action="store_true",help="No models or photos")
    for name in ("manifest","sface_calibration","ghostface_calibration","yunet",
                 "sface","ghostface","clone_diagnostic","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path)
    args=parser.parse_args()
    if args.selftest:
        print(json.dumps(synthetic_di_chain_check(),indent=2))
    else:
        required=("manifest","sface_calibration","ghostface_calibration","yunet",
                  "sface","ghostface","clone_diagnostic","out")
        missing=[name for name in required if getattr(args,name) is None]
        if missing:parser.error("Missing: "+", ".join(missing))
        run(args)
