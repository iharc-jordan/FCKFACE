"""H11: three-step dots versus their exact in-grid complement, development only.

The optimization surrogate is H10's fixed-condition SFace/GhostFaceNet loss.
Only the carrier support changes. All four forward-step-3 JPEGs freeze before
any additional same-person gallery view is opened. Native checkpoint scores
are diagnostics, never selectors. Official ArcFace scores are a separate pass
over the frozen manifest after root appearance screening.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from optimize_transfer_dots import surrogate_loss, summarize, write_json

IDS=("frll-001","frll-003")
ARMS=("dots","inverse_dots")
VIEWS=("neutral_front","smiling_front","neutral_left_3quarter","neutral_right_3quarter")
OBJECTIVE=("export","jpeg75_420","blur","crop90")
CONDITIONS=("export","jpeg85_420","jpeg75_420","resize960","half_restore","crop90","blur")
STEPS=3
SEED=0
LR=.15
H10_SOURCE_SHA256="a04a0d6b42fb8f6375cca20bcc871ff46cf659049c5ceb5977d1f345111eb273"


def support_masks(carrier):
    """Use H5's literal dot mask and its exact complement inside the 14² grid."""
    import torch
    x0,y0,x1,y1=carrier.box
    bw,bh=x1-x0,y1-y0
    height,width=carrier.roi.shape[:2]
    yy,xx=np.mgrid[:height,:width]
    u=(xx+carrier.left-x0)/bw*14
    v=(yy+carrier.top-y0)/bh*14
    cx,cy=np.floor(u).astype(int),np.floor(v).astype(int)
    inside=(cx>=0)&(cx<14)&(cy>=0)&(cy<14)
    original=carrier.dot_mask[0].numpy().astype(bool)
    if np.any(original&~inside):
        raise AssertionError("H5 dots extend outside their grid")
    complement=inside&~original
    if np.any(original&complement) or not np.array_equal(original|complement,inside):
        raise AssertionError("Dot/inverse supports are not exact in-grid complements")
    face=carrier.face.numpy()
    denominator=int(face.sum())
    grid_face=int((inside&face).sum())
    counts={"face_mask_pixels":denominator,"grid_and_face_pixels":grid_face,
            "dots_pixels":int((original&face).sum()),
            "inverse_dots_pixels":int((complement&face).sum())}
    if counts["dots_pixels"]+counts["inverse_dots_pixels"]!=grid_face:
        raise AssertionError("Face-masked support counts do not partition grid")
    counts["dots_fraction_of_face"]=counts["dots_pixels"]/denominator
    counts["inverse_dots_fraction_of_face"]=counts["inverse_dots_pixels"]/denominator
    counts["dots_fraction_of_grid_face"]=counts["dots_pixels"]/grid_face
    counts["inverse_dots_fraction_of_grid_face"]=counts["inverse_dots_pixels"]/grid_face
    return torch.from_numpy(complement.astype(np.float32)).unsqueeze(0),counts,inside


def carrier_for(source,box,matrix,arm):
    from optimize_gradient_art import Carrier
    carrier=Carrier(source,box,matrix,"dots_14x14")
    inverse,counts,_=support_masks(carrier)
    if arm=="inverse_dots":
        carrier.dot_mask=inverse
    elif arm!="dots":
        raise ValueError(f"Unknown H11 support: {arm}")
    return carrier,counts


def synthetic_carrier_check():
    """No model or real photo: exact H5 dots + inverse = in-grid face field."""
    import torch
    from optimize_gradient_art import Carrier
    source=np.full((256,256,3),127,np.uint8)
    box=(40.,40.,216.,216.)
    scale=112/176
    matrix=np.array([[scale,0,-40*scale],[0,scale,-40*scale]],np.float32)
    carrier=Carrier(source,box,matrix,"dots_14x14")
    inverse,counts,inside=support_masks(carrier)
    param=torch.ones((3,14,14),dtype=torch.float32)
    dots=carrier.field(param)
    carrier.dot_mask=inverse
    complement=carrier.field(param)
    expected=torch.from_numpy(inside.astype(np.float32)).unsqueeze(0)*carrier.mask
    maximum=float((dots+complement-expected).abs().max())
    outside_grid=int(torch.count_nonzero((dots+complement)[:,~torch.from_numpy(inside)]))
    if maximum>1e-7 or outside_grid:
        raise AssertionError("Dot and inverse rendered fields do not sum to the in-grid mask")
    return {"synthetic_face_support":counts,"parameter_count_per_arm":3*14*14,
            "theoretical_dot_cell_area_fraction":float(np.pi*.3**2),
            "maximum_field_partition_error":maximum,"outside_grid_nonzero":outside_grid}


def verify_inputs(args):
    from fckface_lab.calibration import Calibration
    from fckface_lab.datasets import digest,grouped_images
    repo=Path(__file__).resolve().parents[1]
    if args.out.exists() or args.out.resolve().is_relative_to(repo):
        raise ValueError("Output must be a new external directory")
    if digest(Path(__file__).parent/"optimize_transfer_dots.py")!=H10_SOURCE_SHA256:
        raise ValueError("Frozen H10 surrogate source changed")
    manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    groups=grouped_images(args.manifest,"development")
    if any(identity not in groups or not set(VIEWS).issubset(groups[identity]) for identity in IDS):
        raise ValueError("Frozen H11 development people need all four views")
    expected={(row["identity"],row["view"]):row["sha256"]
              for row in manifest["images"] if row["split"]=="development"}
    calibrations={"SFace":json.loads(args.sface_calibration.read_text(encoding="utf-8")),
                  "GhostFaceNet":json.loads(args.ghostface_calibration.read_text(encoding="utf-8"))}
    model_paths={"yunet":args.yunet,"sface":args.sface,"ghostface":args.ghostface}
    for name,cal in calibrations.items():
        for model_name in (("yunet","sface") if name=="SFace" else ("yunet","ghostface")):
            record=cal["pipeline"]["model_artifacts"][model_name]
            if digest(model_paths[model_name])!=record["sha256"] or model_paths[model_name].stat().st_size!=record["bytes"]:
                raise ValueError(f"Calibrated model differs: {name}/{model_name}")
        if digest(args.manifest)!=cal["dataset"]["manifest_sha256"]:
            raise ValueError(f"Manifest differs from {name} calibration")
        for source,sha in cal["pipeline"]["source_sha256"].items():
            if source.startswith("fckface_lab/") and digest(Path(__file__).parent/source)!=sha:
                raise ValueError(f"Calibrated preprocessing changed: {source}")
    diagnostic=json.loads(args.clone_diagnostic.read_text(encoding="utf-8"))
    if (not diagnostic["gradient_finite"] or
        any(row["unit_cosine"]<.999 or row["unit_max_abs_difference"]>.02
            for row in diagnostic["parity"].values()) or
        not any(row["relative_error"]<=.1 or row["absolute_error"]<=1e-6
                for row in diagnostic["unit_l2_finite_differences"])):
        raise ValueError("Ghost clone diagnostic failed")
    thresholds={name:Calibration(**cal["calibration"]).threshold for name,cal in calibrations.items()}
    return groups,expected,calibrations,model_paths,thresholds


def optimize(source,box,points,models,torch_sface,ghost_gradient,native_clean,
             thresholds,arm,directory:Path):
    import torch
    from optimize_alignment_dots import native_objective
    from optimize_gradient_art import affine
    carrier,counts=carrier_for(source,box,affine(points["export"]),arm)
    directory.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(SEED)
    param=torch.nn.Parameter(torch.from_numpy(rng.standard_normal((3,14,14)).astype(np.float32)*.1))
    adam=torch.optim.Adam([param],lr=LR)
    start=perf_counter()
    final=None
    for step in range(1,STEPS+1):
        adam.zero_grad(set_to_none=True)
        delta=carrier.project(param)
        loss,jpeg,surrogate=surrogate_loss(source,box,carrier,delta,points,
            torch_sface,ghost_gradient,native_clean,thresholds,{},"adam")
        loss.backward()
        grad=param.grad
        if grad is None or not bool(torch.isfinite(grad).all()) or float(grad.abs().mean())<=0:
            raise RuntimeError(f"Invalid coefficient gradient at step {step}")
        adam.step()
        if step==STEPS:
            native_score,native=native_objective(jpeg,box,models,native_clean,thresholds)
            selected=directory/"selected.jpg"
            selected.write_bytes(jpeg)
            final={"step":step,"forward_before_step3_update":True,
                   "jpeg_sha256":hashlib.sha256(jpeg).hexdigest(),
                   "pre_jpeg_face_rms":float(torch.sqrt(torch.mean(delta.detach()[:,carrier.face]**2))),
                   "pre_jpeg_max_channel_delta":float(delta.detach().abs().max()),
                   "surrogate_worst_margin":float(loss.detach()),
                   "surrogate_source_cosines":surrogate,
                   "native_worst_margin_or_failure_one":native_score,"native":native,
                   "native_all_four_conditions_valid":all(
                       native[name][condition]["status"]=="valid"
                       for name in ("SFace","GhostFaceNet") for condition in OBJECTIVE),
                   "support":counts,"parameter_count":param.numel(),
                   "search_seconds":perf_counter()-start,"selected_path":str(selected)}
            write_json(directory/"step03-diagnostic.json",final)
    return final


def freeze_distortion(output:Path,cases:list)->None:
    from fckface_lab.datasets import digest
    from fckface_lab.imaging import decode_image,export_jpeg,make_variants
    from fckface_lab.patterns import actual_distortion
    rows={}
    for case in cases:
        source=decode_image(Path(case["source_path"]))
        control=make_variants(export_jpeg(source),case["box"])
        rows[case["identity"]]={}
        for arm in ARMS:
            record=case["arms"][arm]
            candidate=Path(record["selected_path"])
            if digest(candidate)!=record["jpeg_sha256"]:
                raise ValueError("Selected JPEG changed before distortion measurement")
            variants=make_variants(candidate.read_bytes(),case["box"])
            metrics={}
            for condition in CONDITIONS:
                a,b=control[condition],variants[condition]
                if a.status!="valid" or b.status!="valid":
                    metrics[condition]={"status":"inconclusive","reason":a.reason or b.reason}
                else:
                    x0,y0,x1,y1=b.target_box
                    metrics[condition]={"status":"valid",**actual_distortion(
                        a.image,b.image,(x0,y0,x1-x0,y1-y0)),
                        "processed_jpeg_sha256":hashlib.sha256(b.jpeg).hexdigest()}
            rows[case["identity"]][arm]=metrics
    write_json(output/"distortion.json",{"basis":"exact clean/edited condition JPEG pixels",
                                          "identities":rows})


def contact_sheets(output:Path,cases:list)->None:
    from PIL import Image,ImageDraw,ImageOps
    from fckface_lab.imaging import decode_image
    columns=("clean",)+ARMS
    for mode in ("full","face"):
        cell_w,cell_h=(420,590) if mode=="full" else (420,460)
        sheet=Image.new("RGB",(cell_w*len(columns),cell_h*len(cases)),"#f6f7f9")
        draw=ImageDraw.Draw(sheet)
        for row,case in enumerate(cases):
            x0,y0,x1,y1=case["box"]
            for col,arm in enumerate(columns):
                path=(Path(case["source_path"]) if arm=="clean"
                      else Path(case["arms"][arm]["selected_path"]))
                image=Image.fromarray(decode_image(path))
                if mode=="face":
                    mx,my=.2*(x1-x0),.2*(y1-y0)
                    image=image.crop((max(0,int(x0-mx)),max(0,int(y0-my)),
                        min(image.width,int(x1+mx)),min(image.height,int(y1+my))))
                thumbnail=ImageOps.contain(image,(cell_w-20,cell_h-52),Image.Resampling.LANCZOS)
                sheet.paste(thumbnail,(col*cell_w+(cell_w-thumbnail.width)//2,
                    row*cell_h+38+(cell_h-52-thumbnail.height)//2))
                draw.text((col*cell_w+12,row*cell_h+12),f"{case['identity']} | {arm}",fill="#162033")
        sheet.save(output/f"{mode}-contact.jpg",quality=93,subsampling=0)


def score_native_after_freeze(output:Path,cases:list,groups:dict,expected:dict,
                              models:dict,calibrations:dict)->None:
    from fckface_lab.calibration import Calibration
    from fckface_lab.datasets import digest
    from fckface_lab.evaluation import Reference
    from fckface_lab.imaging import decode_image,export_jpeg
    from optimize_regions import evaluate_conditions
    frozen_path=output/"frozen-inputs.json"
    frozen=json.loads(frozen_path.read_text(encoding="utf-8"))["items"]
    if len(frozen)!=len(IDS)*len(ARMS) or {row["identity"] for row in frozen}!=set(IDS):
        raise ValueError("Incomplete frozen inputs; gallery access refused")
    for row in frozen:
        if digest(Path(row["path"]))!=row["sha256"]:
            raise ValueError("Frozen selected JPEG changed")
    thresholds={name:Calibration(**cal["calibration"]) for name,cal in calibrations.items()}
    report={"status":"in_progress","purpose":"H11 frozen development native scoring",
            "frozen_manifest_sha256":digest(frozen_path),"conditions":CONDITIONS,"identities":{}}
    for case in cases:
        identity=case["identity"]
        photos={};photo_hashes={}
        for view in VIEWS:
            path=groups[identity][view]
            sha=digest(path)
            if sha!=expected[(identity,view)]:
                raise ValueError(f"Gallery image changed: {identity}/{view}")
            photos[view]=decode_image(path);photo_hashes[view]=sha
        person={"gallery_sha256":photo_hashes,"models":{}}
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
                selected=next(row for row in frozen if row["identity"]==identity and row["arm"]==arm)
                candidate=evaluate_conditions(Path(selected["path"]).read_bytes(),case["box"],
                    model,refs,thresholds[name],identity,CONDITIONS,save_dir=root/arm)
                record["arms"][arm]={"candidate":candidate,"summary":summarize(candidate),
                    "clean_eligible":eligible,"selected_sha256":selected["sha256"]}
            person["models"][name]=record
        report["identities"][identity]=person
        write_json(output/"native-seven-condition-progress.json",report)
        print(identity,"native seven-condition scored",flush=True)
    report["status"]="complete"
    write_json(output/"native-seven-condition.json",report)


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

    groups,expected,calibrations,model_paths,thresholds=verify_inputs(args)
    cv2.setNumThreads(2);torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=False)
    snapshots=args.out/"source-snapshot";snapshots.mkdir()
    source_hashes={}
    for name in ("optimize_support_probe.py","optimize_transfer_dots.py",
                 "optimize_alignment_dots.py","optimize_gradient_art.py",
                 "optimize_ensemble_dots.py","optimize_regions.py"):
        source=Path(__file__).parent/name
        target=snapshots/name
        shutil.copyfile(source,target)
        source_hashes[name]=digest(source)
        if digest(target)!=source_hashes[name]:
            raise ValueError(f"Source changed during snapshot: {name}")
    pinned_dir=args.out/"model-inputs";pinned_dir.mkdir()
    pinned={}
    for name,path in model_paths.items():
        target=pinned_dir/path.name
        shutil.copyfile(path,target)
        reference="GhostFaceNet" if name=="ghostface" else "SFace"
        artifact=calibrations[reference]["pipeline"]["model_artifacts"][name]
        if digest(target)!=artifact["sha256"] or target.stat().st_size!=artifact["bytes"]:
            raise ValueError(f"Private model copy differs from calibration: {name}")
        pinned[name]=target
    protocol={"hypothesis":"H11 sparse dot support limits unseen-model transfer",
              "development_only":True,"identities":IDS,"arms":ARMS,"source_view":"neutral_front",
              "parameters_per_arm":588,"steps":STEPS,"seed":SEED,"adam_lr":LR,
              "support":"H5 14x14 bounding grid dot radius 0.3 versus exact in-grid complement, both times identical feathered face mask",
              "theoretical_dot_cell_area_fraction":float(np.pi*.3**2),
              "exact_support_fractions":"both identities' face-masked pixel counts frozen in support-before-optimization.json before any arm search; depend on clean detector box and face mask",
              "carrier":"H5 face RMS16, per-channel cap64, same cell parameter lookup",
              "objective_conditions":OBJECTIVE,"final_conditions":CONDITIONS,
              "gradient_model_condition_forwards_per_arm":STEPS*len(OBJECTIVE)*2,
              "native_checkpoint_model_condition_queries_per_arm":len(OBJECTIVE)*2,
              "selection":"unconditional forward step3 JPEG; one native diagnostic checkpoint only",
              "prediction":"Root appearance pass, valid all7, inverse ArcFace worst-gallery cosine at least .05 lower on EACH person than dots, and SFace/Ghost worst normalized margins worsen at most .02 each; otherwise no expansion",
              "distortion_balance":"report exact-export and every processed-condition face RMS, flag pair difference >.25; not attributed solely to support if unbalanced",
              "gallery_barrier":"four selected JPEGs and SHA-256 manifest frozen before other same-person views",
              "surrogate_limit":"H7 fixed-affine backward path; JPEG, blur, rounding, detection have no derivative",
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
    ghost_gradient,gradient_report=tf_value_gradient_functions(clone)
    write_json(args.out/"ghost-gradient-preflight.json",gradient_report)
    if not gradient_report["compiled_accepted"]:
        raise RuntimeError("H7 compiled Ghost gradient failed parity preflight")
    torch_sface=convert(onnx.load(str(pinned["sface"]))).eval()
    for parameter in torch_sface.parameters():parameter.requires_grad_(False)
    cases=[];frozen=[];start=perf_counter()
    try:
        prepared=[]
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
                raise ValueError("Native clean target boxes disagree")
            native_clean={name:item.feature for name,item in native.items()}
            clean_conditions=make_variants(export_jpeg(source),box)
            points={condition:detect_points(models["SFace"].detector,clean_conditions[condition])
                    for condition in OBJECTIVE}
            from optimize_gradient_art import affine
            _,support=carrier_for(source,box,affine(points["export"]),"dots")
            case={"identity":identity,"source_path":str(path),"source_sha256":digest(path),
                  "box":box,"support":support,"arms":{}}
            prepared.append((case,source,native_clean,points))
        write_json(args.out/"support-before-optimization.json",
            {"status":"frozen_before_first_arm","parameter_count_per_arm":588,
             "identities":{case["identity"]:case["support"] for case,_,_,_ in prepared}})
        for case,source,native_clean,points in prepared:
            identity=case["identity"]
            box=case["box"]
            for arm in ARMS:
                directory=args.out/identity/arm
                record=optimize(source,box,points,models,torch_sface,ghost_gradient,
                    native_clean,thresholds,arm,directory)
                if record["support"]!=case["support"]:
                    raise AssertionError("Support changed after preoptimization freeze")
                case["arms"][arm]=record
                frozen.append({"identity":identity,"method":"H11","arm":arm,
                    "path":record["selected_path"],"sha256":record["jpeg_sha256"],
                    "source_view":"neutral_front"})
                print(identity,arm,"fixed-step frozen",record["native_all_four_conditions_valid"],flush=True)
            cases.append(case)
        if len(frozen)!=4:
            raise RuntimeError("Incomplete four-JPEG freeze")
        write_json(args.out/"frozen-inputs.json",{"schema_version":1,"items":frozen})
        write_json(args.out/"freeze-results.json",{"status":"four_outputs_frozen",
            "protocol_sha256":digest(args.out/"protocol.json"),
            "frozen_manifest_sha256":digest(args.out/"frozen-inputs.json"),
            "optimization_seconds_excluding_load":perf_counter()-start,"cases":cases})
        freeze_distortion(args.out,cases)
        contact_sheets(args.out,cases)
        # The three other same-person clean views may be opened only now.
        score_native_after_freeze(args.out,cases,groups,expected,models,calibrations)
    except Exception as exc:
        write_json(args.out/"failure.json",{"type":type(exc).__name__,"message":str(exc),
                                              "frozen_count":len(frozen)})
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest",action="store_true",help="Only synthetic support checks; no models/photos")
    for name in ("manifest","sface_calibration","ghostface_calibration","yunet",
                 "sface","ghostface","clone_diagnostic","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path)
    args=parser.parse_args()
    if args.selftest:
        print(json.dumps(synthetic_carrier_check(),indent=2))
    else:
        required=("manifest","sface_calibration","ghostface_calibration","yunet",
                  "sface","ghostface","clone_diagnostic","out")
        missing=[name for name in required if getattr(args,name) is None]
        if missing:parser.error("Missing: "+", ".join(missing))
        run(args)
