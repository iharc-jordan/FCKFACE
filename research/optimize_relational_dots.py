"""H8B fixed-step, development-only identity-relation dot optimization.

The candidate geometry and BPDA follow H7's fixed_condition_landmarks arm.
The relation is a vector of centered similarities to four-view centroids of
OTHER development identities. This is an unvalidated transfer hypothesis.
No H8 same-person gallery is read until all six pilot JPEGs are frozen.

The forward paths are exact condition JPEGs. Their backward paths use H7's
fixed-affine identity BPDA, a converted ONNX SFace, and a float32 author-H5
Ghost clone. The Ghost clone is an approximation to author H5; its parity
diagnostic is required and both source and edited relations use clone space
during gradient steps. Every arm exports forward step 18 unconditionally.
Native checkpoint scores are diagnostic only; detector failures are inconclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

IDS = ("frll-021", "frll-026", "frll-029", "frll-030", "frll-032",
       "frll-037", "frll-041", "frll-042")
PILOT_IDS = IDS[:2]
VIEWS = ("neutral_front", "smiling_front", "neutral_left_3quarter",
         "neutral_right_3quarter")
OBJECTIVE = ("export", "jpeg75_420", "blur", "crop90")
CONDITIONS = ("export", "jpeg85_420", "jpeg75_420", "resize960",
              "half_restore", "crop90", "blur")
ARMS = ("baseline", "relational", "permuted_relational")
STEPS = 18
CHECKPOINT_EVERY = 3
WEIGHT = .25
PERMUTATION_SEED = 20260925


def unit_numpy(feature: np.ndarray) -> np.ndarray:
    value = np.asarray(feature, np.float64).reshape(-1)
    length = float(np.linalg.norm(value))
    if not np.all(np.isfinite(value)) or not np.isfinite(length) or length <= 0:
        raise ValueError("Nonfinite or zero feature")
    return value / length


def validate_anchors(anchors: np.ndarray, dimension: int | None = None) -> np.ndarray:
    value = np.asarray(anchors, np.float64)
    if value.ndim != 2 or value.shape[0] < 2 or (dimension is not None and value.shape[1] != dimension):
        raise ValueError("At least two anchors of the model feature dimension are required")
    if not np.all(np.isfinite(value)) or np.max(np.abs(np.linalg.norm(value, axis=1)-1)) > 1e-5:
        raise ValueError("Every anchor must be finite and unit length")
    return value


def relation_numpy(feature: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    unit = unit_numpy(feature)
    anchors = validate_anchors(anchors, unit.size)
    scores = anchors @ unit
    centered = scores-scores.mean()
    length = float(np.linalg.norm(centered))
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("Zero or nonfinite centered relation")
    return centered/length


def displacement_numpy(source_relation: np.ndarray, edited_relation: np.ndarray) -> np.ndarray:
    return edited_relation - np.dot(source_relation, edited_relation)*source_relation


def agreement_numpy(s_source: np.ndarray, g_source: np.ndarray,
                    s_edited: list[np.ndarray], g_edited: list[np.ndarray],
                    s_anchors: np.ndarray, g_anchors: np.ndarray) -> tuple[float, list[float]]:
    if len(s_edited) != len(g_edited) or not s_edited:
        raise ValueError("Paired nonempty conditions required")
    s0 = relation_numpy(s_source, s_anchors)
    g0 = relation_numpy(g_source, g_anchors)
    values = [float(np.dot(displacement_numpy(s0, relation_numpy(s, s_anchors)),
                           displacement_numpy(g0, relation_numpy(g, g_anchors))))
              for s, g in zip(s_edited, g_edited)]
    return min(values), values


def permutation(count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("At least two anchors required")
    return np.random.default_rng(PERMUTATION_SEED).permutation(count)


def unit_torch(feature):
    import torch
    value = feature.reshape(-1)
    length = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(length)) or float(length.detach()) <= 0:
        raise ValueError("Invalid Torch feature")
    return value/length


def relation_torch(feature, anchors):
    import torch
    scores = anchors @ unit_torch(feature)
    centered = scores-scores.mean()
    length = torch.linalg.vector_norm(centered)
    if not bool(torch.isfinite(length)) or float(length.detach()) <= 1e-8:
        raise ValueError("Zero Torch relation")
    return centered/length


def relation_tf(feature, anchors):
    import tensorflow as tf
    value = tf.reshape(feature, (-1,))
    unit = tf.math.l2_normalize(value, axis=0)
    scores = tf.linalg.matvec(anchors, unit)
    centered = scores-tf.reduce_mean(scores)
    # Check the raw norm before dividing; silently clipping this would make
    # a nonexistent identity relation look like a usable zero vector.
    length = tf.linalg.norm(centered)
    if not np.isfinite(float(length.numpy())) or float(length.numpy()) <= 1e-8:
        raise ValueError("Zero TensorFlow relation")
    return centered/length


def displacement_torch(source_relation, edited_relation):
    import torch
    return edited_relation-torch.dot(source_relation, edited_relation)*source_relation


def displacement_tf(source_relation, edited_relation):
    import tensorflow as tf
    return edited_relation-tf.tensordot(source_relation, edited_relation, axes=1)*source_relation


def normalized_margin(score, threshold: float):
    if not -1 < threshold < 1:
        raise ValueError("Expected cosine threshold strictly inside (-1,1)")
    return (score-threshold)/(1-threshold)


def native_objective(jpeg: bytes, box, models: dict, source_features: dict,
                     thresholds: dict, anchors: dict, arm: str):
    """Fresh native model features; invalid conditions cannot win selection."""
    from fckface_lab.imaging import make_variants
    from optimize_gradient_art import cosine

    variants = make_variants(jpeg, box)
    rows, edited, margins = {}, {"SFace": [], "GhostFaceNet": []}, []
    valid = True
    for condition in OBJECTIVE:
        rows[condition] = {}
        variant = variants[condition]
        for model_name in ("SFace", "GhostFaceNet"):
            if variant.status == "valid" and variant.image is not None:
                result = models[model_name].embed(variant.image, expected_box=variant.target_box)
                status, reason = result.status, result.reason
                feature = result.feature if status == "valid" else None
            else:
                status, reason, feature = "inconclusive", variant.reason, None
            if feature is None:
                valid = False
                rows[condition][model_name] = {"status": status, "reason": reason,
                                               "source_cosine": None, "normalized_margin": None}
            else:
                score = cosine(feature, source_features[model_name])
                margin = float(normalized_margin(score, thresholds[model_name]))
                edited[model_name].append(feature)
                margins.append(margin)
                rows[condition][model_name] = {"status": status, "reason": reason,
                                               "source_cosine": score, "normalized_margin": margin}
    if not valid:
        return None, {"conditions": rows, "status": "inconclusive", "reason": "invalid_model_condition"}
    if arm == "baseline":
        agreement, per_condition = 0.0, []
    else:
        try:
            agreement, per_condition = agreement_numpy(
                source_features["SFace"], source_features["GhostFaceNet"],
                edited["SFace"], edited["GhostFaceNet"],
                anchors["SFace"], anchors["GhostFaceNet"])
        except ValueError as exc:
            return None, {"conditions": rows, "status": "inconclusive",
                          "reason": f"undefined_relation: {exc}"}
    worst_margin = max(margins)
    score = worst_margin-(WEIGHT*agreement if arm != "baseline" else 0.0)
    return score, {"conditions": rows, "status": "valid", "worst_margin": worst_margin,
                   "minimum_agreement": agreement, "agreement_by_condition": dict(zip(OBJECTIVE, per_condition)),
                   "objective": score}


def build_anchors(groups: dict, expected: dict, models: dict, output: Path):
    """Use only four-view complete OTHER development identities, before attacks."""
    from fckface_lab.datasets import digest
    from fckface_lab.imaging import decode_image

    identities = sorted(set(groups)-set(IDS))
    accepted, coverage = [], {}
    vectors = {"SFace": [], "GhostFaceNet": []}
    for identity in identities:
        samples = {name: [] for name in vectors}
        row = {"views": {}, "eligible": False}
        for view in VIEWS:
            path = groups[identity].get(view)
            if path is None or (identity, view) not in expected:
                row["views"][view] = {"status": "missing"}
                continue
            observed = digest(path)
            if observed != expected[(identity, view)]:
                raise ValueError(f"Manifest image hash changed: {identity}/{view}")
            image = decode_image(path)
            row["views"][view] = {"path": str(path), "sha256": observed, "models": {}}
            for name, model in models.items():
                result = model.embed(image)
                row["views"][view]["models"][name] = {"status": result.status, "reason": result.reason}
                if result.status == "valid":
                    samples[name].append(unit_numpy(result.feature))
        if (len(row["views"]) == len(VIEWS)
                and all(len(samples[name]) == len(VIEWS) for name in samples)):
            centroids = {name: unit_numpy(np.mean(samples[name], axis=0)) for name in samples}
            for name in vectors:
                vectors[name].append(centroids[name])
            row["eligible"] = True
            accepted.append(identity)
        coverage[identity] = row
    if len(accepted) < 2:
        raise ValueError("Fewer than two both-model, four-view anchor identities")
    arrays = {name: validate_anchors(np.stack(rows)) for name, rows in vectors.items()}
    np.savez_compressed(output/"anchors.npz", identities=np.asarray(accepted),
                        SFace=arrays["SFace"].astype(np.float32),
                        GhostFaceNet=arrays["GhostFaceNet"].astype(np.float32))
    (output/"anchor-coverage.json").write_text(json.dumps({
        "excluded_evaluation_ids": IDS, "candidate_identity_count": len(identities),
        "accepted_identity_count": len(accepted), "accepted_ids": accepted,
        "requirement": "all four specified views valid on both native models",
        "coverage": coverage}, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return arrays, accepted


def surrogate_source_relations(source, points, torch_sface, clone, anchors):
    """Use each differentiable model's own clean source feature for r0."""
    import cv2
    import torch
    import tensorflow as tf
    from optimize_gradient_art import affine, aligned_rgb
    from fckface_lab.ghostface import author_align
    s_crop = aligned_rgb(source, affine(points))
    s_input = torch.from_numpy(np.ascontiguousarray(s_crop.transpose(2,0,1).astype(np.float32)))
    with torch.no_grad():
        s_feature = unit_torch(torch_sface(s_input.unsqueeze(0)))
    g_crop = author_align(source, points)
    g_input = tf.constant(g_crop[None].astype(np.float32))
    g_feature = tf.math.l2_normalize(clone((g_input-127.5)*.0078125, training=False),axis=1)[0]
    if (not np.all(np.isfinite(g_feature.numpy()))
            or float(tf.linalg.norm(g_feature).numpy()) <= 0):
        raise ValueError("Ghost clone source feature nonfinite")
    s_anchor = torch.from_numpy(anchors["SFace"].astype(np.float32))
    g_anchor = tf.constant(anchors["GhostFaceNet"].astype(np.float32))
    return {
        "SFace": relation_torch(s_feature,s_anchor).detach(),
        "GhostFaceNet": tf.stop_gradient(relation_tf(g_feature,g_anchor))}


def surrogate_loss(source, box, carrier, delta, clean_landmarks, models,
                   torch_sface, clone, native_source_features, source_relations,
                   anchors, thresholds, arm):
    """One forward per model/condition, two exact partial-gradient branches.

    The SFace Torch branch and Ghost TensorFlow branch each differentiate the
    *same* active maximum margin and minimum agreement. TensorFlow's pixel VJP
    is injected at the Ghost crop proxy, so Torch backprop adds both branches
    through their separate fixed-affine grids to the shared dot parameters.
    There is no zero-displacement normalization. JPEG/blur remain BPDA.
    """
    import torch
    import torch.nn.functional as F
    import tensorflow as tf
    from fckface_lab.ghostface import author_align
    from fckface_lab.imaging import export_jpeg, make_variants
    from optimize_alignment_dots import crop_offset, ghost_matrix, warp_grid
    from optimize_gradient_art import affine, aligned_rgb

    jpeg = export_jpeg(carrier.render(delta))
    variants = make_variants(jpeg, box)
    s_anchors = torch.from_numpy(anchors["SFace"].astype(np.float32))
    g_anchors = tf.constant(anchors["GhostFaceNet"].astype(np.float32))
    # H7's baseline margins compare the surrogate edited feature to the
    # calibrated native clean source, not a surrogate clean feature.
    source_features = {
        "SFace": torch.from_numpy(native_source_features["SFace"].astype(np.float32)),
        "GhostFaceNet": tf.constant(native_source_features["GhostFaceNet"].astype(np.float32))}
    s_margins, s_v, s_scores = [], [], []
    g_proxies, g_pixels = [], []
    for condition in OBJECTIVE:
        variant = variants[condition]
        if variant.status != "valid" or variant.image is None:
            raise RuntimeError(f"Invalid surrogate condition {condition}: {variant.reason}")
        points = clean_landmarks[condition]
        offset = crop_offset(condition, source.shape)
        sm = affine(points)
        sg = warp_grid(carrier, np.vstack((sm,[0,0,1])), offset)
        gg = warp_grid(carrier, ghost_matrix(points), offset)
        sd = F.grid_sample(delta.unsqueeze(0),sg,mode="bilinear",padding_mode="zeros",align_corners=True)[0]
        gd = F.grid_sample(delta.unsqueeze(0),gg,mode="bilinear",padding_mode="zeros",align_corners=True)[0]
        sc = aligned_rgb(variant.image,sm)
        se = torch.from_numpy(np.ascontiguousarray(sc.transpose(2,0,1).astype(np.float32)))
        sp = se+sd-sd.detach()
        sf = unit_torch(torch_sface(sp.unsqueeze(0)))
        score = torch.dot(sf, source_features["SFace"])
        s_scores.append(float(score.detach()))
        s_margins.append(normalized_margin(score,thresholds["SFace"]))
        if arm != "baseline":
            s_v.append(displacement_torch(source_relations["SFace"],relation_torch(sf,s_anchors)))
        gc = author_align(variant.image,points)
        ge = torch.from_numpy(np.ascontiguousarray(gc.transpose(2,0,1).astype(np.float32)))
        gp = ge+gd-gd.detach()
        g_proxies.append(gp)
        g_pixels.append(tf.constant(gc[None].astype(np.float32)))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        for pixel in g_pixels:
            tape.watch(pixel)
        g_margins, g_v, g_scores = [], [], []
        for pixel in g_pixels:
            raw = clone((pixel-127.5)*.0078125,training=False)
            gf = tf.math.l2_normalize(raw,axis=1)[0]
            if (not np.all(np.isfinite(gf.numpy()))
                    or float(tf.linalg.norm(raw).numpy()) <= 0):
                raise ValueError("Nonfinite Ghost clone feature")
            score = tf.tensordot(gf, source_features["GhostFaceNet"],axes=1)
            g_scores.append(float(score.numpy()))
            g_margins.append(normalized_margin(score,thresholds["GhostFaceNet"]))
            if arm != "baseline":
                g_v.append(displacement_tf(source_relations["GhostFaceNet"],relation_tf(gf,g_anchors)))
        margins = []
        for k in range(len(OBJECTIVE)):
            margins += [(float(s_margins[k].detach()),"SFace",k),
                        (float(g_margins[k].numpy()),"GhostFaceNet",k)]
        # Python's first-item max/min resolves exact ties deterministically.
        # H7's stacked Torch max can distribute a tied subgradient; ties are
        # valid subgradients here, but an exact tied trajectory need not match.
        worst, active_model, active_condition = max(margins,key=lambda row:row[0])
        agreements = ([float(torch.dot(s_v[k].detach(),torch.from_numpy(g_v[k].numpy())).item())
                       for k in range(len(OBJECTIVE))] if arm != "baseline" else [])
        agreement_index = int(np.argmin(agreements)) if agreements else None
        agreement = agreements[agreement_index] if agreements else 0.0
        # TF receives the Ghost partial of the active max and of -0.25*A.
        g_loss = (g_margins[active_condition] if active_model=="GhostFaceNet"
                  else tf.constant(0.,tf.float32))
        if arm != "baseline":
            s_constant = tf.constant(s_v[agreement_index].detach().numpy())
            g_loss = g_loss-WEIGHT*tf.tensordot(s_constant,g_v[agreement_index],axes=1)
    g_gradients = tape.gradient(g_loss,g_pixels)
    connected = ({active_condition} if active_model=="GhostFaceNet" else set())
    if agreement_index is not None:
        connected.add(agreement_index)
    # A missing gradient for an active Ghost contribution is an error. Inactive
    # conditions (including all Ghost pixels for a baseline SFace winner) have
    # exactly zero partial derivative and may legitimately return None.
    if any(g_gradients[k] is None for k in connected):
        raise RuntimeError(f"Missing active Ghost gradient for conditions {sorted(connected)}")
    g_gradients = [tf.zeros_like(pixel) if grad is None else grad
                   for pixel,grad in zip(g_pixels,g_gradients)]
    if any(not np.all(np.isfinite(grad.numpy())) for grad in g_gradients):
        raise RuntimeError("Missing or nonfinite Ghost partial gradient")
    # Torch receives the SFace partial of that same active max and -0.25*A.
    s_loss = (s_margins[active_condition] if active_model=="SFace"
              else torch.tensor(0.,dtype=torch.float32))
    if arm != "baseline":
        g_constant = torch.from_numpy(g_v[agreement_index].numpy())
        s_loss = s_loss-WEIGHT*torch.dot(s_v[agreement_index],g_constant)
    actual = worst-(WEIGHT*agreement if arm != "baseline" else 0.0)
    # The scalar's value is the true objective; the two framework VJPs supply
    # its full partial derivative without double-counting either branch.
    loss = torch.tensor(actual,dtype=torch.float32)+(s_loss-s_loss.detach())
    for proxy,grad in zip(g_proxies,g_gradients):
        gradient = torch.from_numpy(np.ascontiguousarray(grad.numpy()[0].transpose(2,0,1)))
        loss = loss+((proxy-proxy.detach())*gradient).sum()
    if not np.isfinite(actual):
        raise ValueError("Nonfinite surrogate objective")
    return loss, jpeg, {"objective":actual,"worst_margin":worst,
        "active_margin_model":active_model,"active_margin_condition":OBJECTIVE[active_condition],
        "minimum_agreement":agreement,"active_agreement_condition":
            OBJECTIVE[agreement_index] if agreement_index is not None else None,
        "agreement_by_condition":dict(zip(OBJECTIVE,agreements)),
        "source_cosine":{"SFace":dict(zip(OBJECTIVE,s_scores)),
                         "GhostFaceNet":dict(zip(OBJECTIVE,g_scores))}}


def optimize(source,box,clean_landmarks,models,torch_sface,clone,
             thresholds,native_sources,surrogate_relations,
             anchors,arm,checkpoint_dir:Path):
    import torch
    from optimize_gradient_art import Carrier, affine
    from fckface_lab.imaging import export_jpeg
    carrier=Carrier(source,box,affine(clean_landmarks["export"]),"dots_14x14")
    checkpoint_dir.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(0)
    param=torch.nn.Parameter(torch.from_numpy(rng.standard_normal((3,14,14)).astype(np.float32)*.1))
    optimizer=torch.optim.Adam([param],lr=.15)
    log=[];selected=None;t0=perf_counter()
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        delta=carrier.project(param)
        loss,jpeg,surrogate=surrogate_loss(source,box,carrier,delta,clean_landmarks,
            models,torch_sface,clone,native_sources,surrogate_relations,anchors,thresholds,arm)
        loss.backward()
        optimizer.step()
        if step%CHECKPOINT_EVERY==0:
            score,native=native_objective(jpeg,box,models,native_sources,thresholds,anchors,arm)
            row={"step":step,"arm":arm,"native_objective":score,"native":native,
                 "surrogate":surrogate,"pre_jpeg_face_rms":float(torch.sqrt(torch.mean(delta.detach()[:,carrier.face]**2))),
                 "pre_jpeg_max_channel_delta":float(delta.detach().abs().max()),
                 "jpeg_sha256":hashlib.sha256(jpeg).hexdigest()}
            log.append(row)
            (checkpoint_dir/f"step{step:02d}.jpg").write_bytes(jpeg)
            (checkpoint_dir/f"step{step:02d}.json").write_text(
                json.dumps(row,indent=2,allow_nan=False)+"\n",encoding="utf-8")
            if step==STEPS:
                selected={"step":step,"score":score,"jpeg":jpeg,"row":row}
    if selected is None:
        raise RuntimeError("Fixed final step was not exported")
    selected["seconds"]=perf_counter()-t0
    return selected,log


def preflight(args):
    from fckface_lab.datasets import digest, grouped_images
    from fckface_lab.calibration import Calibration
    manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    groups=grouped_images(args.manifest,"development")
    if not set(IDS).issubset(groups):
        raise ValueError("Frozen H8 evaluation IDs not all development identities")
    expected={(row["identity"],row["view"]):row["sha256"] for row in manifest["images"]
              if row["split"]=="development"}
    paths={"yunet":args.yunet,"sface":args.sface,"ghostface":args.ghostface}
    cal_paths={"SFace":args.sface_calibration,"GhostFaceNet":args.ghostface_calibration}
    calibrations={name:json.loads(path.read_text(encoding="utf-8")) for name,path in cal_paths.items()}
    for name,cal in calibrations.items():
        for key in (("yunet","sface") if name=="SFace" else ("yunet","ghostface")):
            if digest(paths[key])!=cal["pipeline"]["model_artifacts"][key]["sha256"]:
                raise ValueError(f"{name} calibration model mismatch: {key}")
        for source,sha in cal["pipeline"]["source_sha256"].items():
            if source.startswith("fckface_lab/") and digest(Path(__file__).parent/source)!=sha:
                raise ValueError(f"{name} calibration pipeline mismatch: {source}")
    diagnostic=json.loads(args.clone_diagnostic.read_text(encoding="utf-8"))
    if (not diagnostic["gradient_finite"]
            or any(row["unit_cosine"]<.999 or row["unit_max_abs_difference"]>.02
                   for row in diagnostic["parity"].values())
            or not any(row["relative_error"]<=.1 or row["absolute_error"]<=1e-6
                       for row in diagnostic["unit_l2_finite_differences"])):
        raise ValueError("H6 float32 Ghost clone diagnostic failed")
    thresholds={name:Calibration(**cal["calibration"]).threshold for name,cal in calibrations.items()}
    return groups,expected,thresholds,cal_paths,paths


def condition_summary(rows: dict) -> dict:
    valid = len(rows)==len(CONDITIONS) and all(row["status"]=="valid" for row in rows.values())
    return {"valid_all":valid,
            "nonmatch_all":valid and all(row.get("own_identity_matched") is False for row in rows.values()),
            "valid_conditions":sum(row["status"]=="valid" for row in rows.values()),
            "inconclusive_conditions":[condition for condition,row in rows.items() if row["status"]!="valid"],
            "worst_gallery_cosine":max((row["maximum_cosine"] for row in rows.values()
                                       if row.get("maximum_cosine") is not None),default=None)}


def score_native_after_freeze(output:Path,records:list,groups:dict,expected:dict,
                              models:dict,cal_paths:dict,arms:tuple[str,...]) -> None:
    """Seven-condition development SFace/Ghost scoring after all JPEG hashes freeze."""
    from fckface_lab.calibration import Calibration
    from fckface_lab.datasets import digest
    from fckface_lab.evaluation import Reference
    from fckface_lab.imaging import decode_image,export_jpeg
    from optimize_regions import evaluate_conditions

    frozen=json.loads((output/"frozen-inputs.json").read_text(encoding="utf-8"))["items"]
    if len(frozen)!=len(records)*len(arms):
        raise ValueError("Cannot read gallery before complete candidate freeze")
    for item in frozen:
        if digest(Path(item["path"]))!=item["sha256"]:
            raise ValueError("Frozen candidate changed before gallery scoring")
    calibrations={name:Calibration(**json.loads(path.read_text(encoding="utf-8"))["calibration"])
                  for name,path in cal_paths.items()}
    results={"status":"in_progress","purpose":"H8B frozen development SFace/Ghost scoring",
             "frozen_manifest_sha256":digest(output/"frozen-inputs.json"),
             "conditions":CONDITIONS,"gallery_views":VIEWS,
             "calibration_sha256":{name:digest(path) for name,path in cal_paths.items()},"identities":{}}
    root=output/"native-seven-condition"
    for case in records:
        identity=case["identity"]
        source=decode_image(groups[identity]["neutral_front"])
        box=case["clean_box"]
        result={"gallery":{},"models":{}}
        gallery_images={}
        for view in VIEWS:
            path=groups[identity][view]
            observed=digest(path)
            if observed!=expected[(identity,view)]:
                raise ValueError(f"Gallery image hash changed: {identity}/{view}")
            result["gallery"][view]={"path":str(path),"sha256":observed}
            gallery_images[view]=decode_image(path)
        for name,model in models.items():
            refs=[];reference_status={}
            for view,image in gallery_images.items():
                embedded=model.embed(image)
                reference_status[view]={"status":embedded.status,"reason":embedded.reason}
                if embedded.status=="valid":
                    refs.append(Reference(f"{identity}:{view}",identity,embedded.feature))
            base=root/identity/name
            control=evaluate_conditions(export_jpeg(source),box,model,refs,
                calibrations[name],identity,CONDITIONS,save_dir=base/"control")
            eligible=(len(refs)==len(VIEWS) and all(row["status"]=="valid" and
                row.get("own_identity_matched") is True for row in control.values()))
            model_result={"reference_status":reference_status,"clean_eligible":eligible,
                          "control":control,"control_summary":condition_summary(control),"arms":{}}
            for arm in arms:
                item=next(row for row in frozen if row["identity"]==identity and row["arm"]==arm)
                candidate=evaluate_conditions(Path(item["path"]).read_bytes(),box,model,refs,
                    calibrations[name],identity,CONDITIONS,save_dir=base/arm)
                model_result["arms"][arm]={"candidate":candidate,
                    "summary":condition_summary(candidate),"eligible":eligible,
                    "selected_sha256":item["sha256"]}
            result["models"][name]=model_result
        results["identities"][identity]=result
        (output/"native-seven-condition-progress.json").write_text(
            json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(identity,"native seven-condition scored",flush=True)
    results["status"]="complete"
    (output/"native-seven-condition.json").write_text(
        json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main(args):
    import cv2
    import onnx
    import torch
    import tensorflow as tf
    from onnx2torch import convert
    from fckface_lab.datasets import digest
    from fckface_lab.imaging import decode_image, export_jpeg, make_variants
    from fckface_lab.recognition import SFaceModel
    from fckface_lab.ghostface import GhostFaceModel
    from optimize_alignment_dots import detect_points
    from optimize_ensemble_dots import load_float32_clone

    torch.set_num_threads(2);cv2.setNumThreads(2)
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.diagnostic_replay:
        if args.anchors_from is None or args.ids != IDS[:1]:
            raise ValueError("Diagnostic replay requires --ids frll-021 and --anchors-from original run")
        arms=("baseline",)
    else:
        if args.anchors_from is not None or args.ids not in (PILOT_IDS,IDS):
            raise ValueError("Only the frozen two-ID pilot or full eight-ID set is allowed")
        arms=ARMS
    groups,expected,thresholds,cal_paths,model_paths=preflight(args)
    args.out.mkdir(parents=True)
    snapshot=args.out/"source-snapshot";snapshot.mkdir()
    sources=("optimize_relational_dots.py","optimize_alignment_dots.py",
             "optimize_gradient_art.py","optimize_ensemble_dots.py",
             "optimize_regions.py")
    source_hashes={}
    for name in sources:
        path=Path(__file__).parent/name
        shutil.copyfile(path,snapshot/name)
        source_hashes[name]=digest(path)
    provenance={"experiment":"H8B fixed-step identity relation displacement",
        "development_only":True,"gallery_read_only_after_complete_freeze":True,
        "no_arcface_reads_in_optimizer":True,
        "evaluation_ids":args.ids,"excluded_anchor_ids":IDS,"views":VIEWS,
        "arms":arms,"steps":STEPS,"checkpoint_every":CHECKPOINT_EVERY,
        "gradient_model_condition_evaluations_per_arm":STEPS*len(OBJECTIVE)*2,
        "objective_conditions":OBJECTIVE,"final_conditions":CONDITIONS,
        "dots":"H5 dots_14x14; RMS16; channel cap64",
        "seed":0,"weight":WEIGHT,"permutation_seed":PERMUTATION_SEED,
        "selection_rule":"Export forward step 18 unconditionally in every arm. Native checks at 3,6,9,12,15,18 are diagnostic only. An invalid final detector result remains inconclusive.",
        "manifest_sha256":digest(args.manifest),
        "calibration_sha256":{name:digest(path) for name,path in cal_paths.items()},
        "model_sha256":{name:digest(path) for name,path in model_paths.items()},
        "clone_diagnostic_sha256":digest(args.clone_diagnostic),
        "source_sha256":source_hashes,"thresholds":thresholds,
        "gradient_note":"Each model has its own source relation in its differentiable feature space. H7 BPDA uses exact JPEG forward pixels; Ghost float32 clone remains approximate to native author H5. Both SFace and Ghost partial gradients of the same active max/min are chained to shared dot parameters. First max/min tie is a valid deterministic subgradient, unlike H7's possible distributed tied Torch max. TF remains eager; timings and 144 forward model-condition evaluations are reported separately.",
        "gate":"Pilot 021,026 only. Extend remaining six only if relational beats each control by >=.02 median ArcFace worst-condition gallery cosine and both identities improve with valid conditions. ArcFace only after all pilot JPEGs frozen; missing prediction retires. No repeated seed."}
    if args.diagnostic_replay:
        original=args.anchors_from
        old=json.loads((original/"protocol.json").read_text(encoding="utf-8"))
        for key in ("manifest_sha256","calibration_sha256","model_sha256","clone_diagnostic_sha256"):
            if old[key]!=provenance[key]:
                raise ValueError(f"Diagnostic replay input differs from original: {key}")
        original_source=original/"source-snapshot"/"optimize_relational_dots.py"
        if digest(original_source)!=old["source_sha256"]["optimize_relational_dots.py"]:
            raise ValueError("Original executed source snapshot changed")
        if old["source_sha256"]["optimize_alignment_dots.py"]!=source_hashes["optimize_alignment_dots.py"] or old["source_sha256"]["optimize_gradient_art.py"]!=source_hashes["optimize_gradient_art.py"] or old["source_sha256"]["optimize_ensemble_dots.py"]!=source_hashes["optimize_ensemble_dots.py"]:
            raise ValueError("Replay changed an imported optimizer source")
        provenance["diagnostic_replay"]={"original":str(original),
            "original_failure":"No valid native checkpoint; no checkpoint rows persisted; frozen_count=0",
            "original_executed_source_sha256":digest(original_source),
            "replay_source_sha256":source_hashes["optimize_relational_dots.py"],
            "original_anchors_sha256":digest(original/"anchors.npz"),
            "anchors_unused_by_baseline_loss":True,
            "change":"Persist each checkpoint JPEG and native status immediately; pin calibrated model copies; reuse original anchors as unused metadata; baseline 021 only; no optimization math changes"}
    # onnx2tf can modify its input ONNX in place. Pin private exact-byte
    # copies before loading either native recognizer or converted SFace, so a
    # concurrent conversion cannot silently mix model versions in one run.
    private_inputs=args.out/"model-inputs";private_inputs.mkdir()
    pinned={}
    for name,path in model_paths.items():
        target=private_inputs/path.name
        shutil.copyfile(path,target)
        if digest(target)!=provenance["model_sha256"][name]:
            raise ValueError(f"Model changed while snapshotting: {name}")
        pinned[name]=target
    provenance["private_model_inputs"]={name:{"path":str(path),"sha256":digest(path)}
                                        for name,path in pinned.items()}
    (args.out/"protocol.json").write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    models={"SFace":SFaceModel(pinned["yunet"],pinned["sface"]),
            "GhostFaceNet":GhostFaceModel(pinned["yunet"],pinned["ghostface"])}
    if args.diagnostic_replay:
        with np.load(args.anchors_from/"anchors.npz",allow_pickle=False) as saved:
            anchor_ids=list(saved["identities"].astype(str))
            anchors={name:validate_anchors(saved[name]) for name in ("SFace","GhostFaceNet")}
        old_coverage=json.loads((args.anchors_from/"anchor-coverage.json").read_text(encoding="utf-8"))
        if anchor_ids!=old_coverage["accepted_ids"] or set(anchor_ids)&set(IDS):
            raise ValueError("Original anchor pool changed or includes an H8 identity")
        shutil.copyfile(args.anchors_from/"anchors.npz",args.out/"anchors.npz")
        shutil.copyfile(args.anchors_from/"anchor-coverage.json",args.out/"anchor-coverage.json")
    else:
        anchors,anchor_ids=build_anchors(groups,expected,models,args.out)
    clone=load_float32_clone(models["GhostFaceNet"].model)
    torch_sface=convert(onnx.load(str(pinned["sface"]))).eval()
    for parameter in torch_sface.parameters():parameter.requires_grad_(False)
    permutation_index=permutation(len(anchor_ids))
    (args.out/"permutation.json").write_text(json.dumps({"seed":PERMUTATION_SEED,
        "anchor_ids":anchor_ids,"ghost_reordered_ids":[anchor_ids[i] for i in permutation_index],
        "indices":permutation_index.tolist()},indent=2)+"\n",encoding="utf-8")
    records=[];frozen=[];start=perf_counter()
    try:
        for identity in args.ids:
            path=groups[identity]["neutral_front"]
            if digest(path)!=expected[(identity,"neutral_front")]:
                raise ValueError(f"Changed source image {identity}")
            image=decode_image(path)
            clean={name:model.embed(image) for name,model in models.items()}
            if any(item.status!="valid" for item in clean.values()):
                raise RuntimeError(f"Invalid clean source {identity}")
            box=clean["SFace"].selected_box
            if box!=clean["GhostFaceNet"].selected_box:
                raise ValueError("Native source boxes disagree")
            native_sources={name:item.feature for name,item in clean.items()}
            clean_variants=make_variants(export_jpeg(image),box)
            points={condition:detect_points(models["SFace"].detector,clean_variants[condition])
                    for condition in OBJECTIVE}
            case={"identity":identity,"source_path":str(path),"source_sha256":digest(path),
                  "clean_box":box,"arms":{}}
            for arm in arms:
                local_anchors={"SFace":anchors["SFace"],
                    "GhostFaceNet":anchors["GhostFaceNet"][permutation_index]
                    if arm=="permuted_relational" else anchors["GhostFaceNet"]}
                surrogate_relations=(surrogate_source_relations(
                    image,points["export"],torch_sface,clone,local_anchors)
                    if arm!="baseline" else {})
                directory=args.out/identity/arm;directory.mkdir(parents=True)
                best,log=optimize(image,box,points,models,torch_sface,clone,thresholds,
                    native_sources,surrogate_relations,local_anchors,arm,directory/"checkpoints")
                selected=directory/"selected.jpg";selected.write_bytes(best["jpeg"])
                (directory/"checkpoints.json").write_text(json.dumps(log,indent=2,allow_nan=False)+"\n",encoding="utf-8")
                case["arms"][arm]={"selected_step":best["step"],
                    "native_diagnostic_objective":best["score"],
                    "native_final_status":best["row"]["native"]["status"],
                    "selected_sha256":digest(selected),"path":str(selected),"seconds":best["seconds"],
                    "pre_jpeg_face_rms":best["row"]["pre_jpeg_face_rms"],
                    "pre_jpeg_max_channel_delta":best["row"]["pre_jpeg_max_channel_delta"]}
                frozen.append({"identity":identity,"method":"H8B","arm":arm,
                               "path":str(selected),"sha256":digest(selected),
                               "source_view":"neutral_front"})
                print(identity,arm,"fixed-step frozen",best["step"],
                      best["row"]["native"]["status"],flush=True)
            records.append(case)
        if len(frozen)!=len(args.ids)*len(arms):
            raise RuntimeError("Incomplete freeze")
        result={"status":"frozen_source_only","protocol":provenance,"anchor_ids":anchor_ids,
                "wall_seconds":perf_counter()-start,"cases":records,
                "arcface_scoring_allowed_only_after_this_complete_manifest":True}
        (args.out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (args.out/"frozen-inputs.json").write_text(json.dumps({"schema_version":1,"items":frozen},
            indent=2,allow_nan=False)+"\n",encoding="utf-8")
        # The complete six-JPEG manifest and hashes are fixed before any
        # same-person gallery image beyond each neutral_front source is read.
        if not args.diagnostic_replay:
            score_native_after_freeze(args.out,records,groups,expected,models,cal_paths,arms)
    except Exception as exc:
        (args.out/"failure.json").write_text(json.dumps({"type":type(exc).__name__,
            "message":str(exc),"frozen_count":len(frozen)},indent=2)+"\n",encoding="utf-8")
        raise


def selftest():
    """Pure synthetic formula, permutation, and both framework partials."""
    import torch
    import tensorflow as tf
    rng=np.random.default_rng(71)
    n=8
    sa=np.stack([unit_numpy(x) for x in rng.normal(size=(n,5))])
    ga=np.stack([unit_numpy(x) for x in rng.normal(size=(n,7))])
    s0=unit_numpy(rng.normal(size=5));g0=unit_numpy(rng.normal(size=7))
    se=s0+.09*rng.normal(size=5);ge=g0+.11*rng.normal(size=7)
    threshold_s=.45;threshold_g=.39
    def scalar(s,g,ganchors):
        a,_=agreement_numpy(s0,g0,[s],[g],sa,ganchors)
        m=max(normalized_margin(float(np.dot(unit_numpy(s),s0)),threshold_s),
              normalized_margin(float(np.dot(unit_numpy(g),g0)),threshold_g))
        return m-WEIGHT*a
    def partials(s,g,ganchors):
        st=torch.tensor(s,dtype=torch.float64,requires_grad=True)
        gt=tf.Variable(g,dtype=tf.float64)
        ss=torch.tensor(s0,dtype=torch.float64);gs=tf.constant(g0,dtype=tf.float64)
        sr=relation_torch(st,torch.tensor(sa,dtype=torch.float64))
        sv=displacement_torch(relation_torch(ss,torch.tensor(sa,dtype=torch.float64)),sr)
        with tf.GradientTape() as tape:
            gr=relation_tf(gt,tf.constant(ganchors,dtype=tf.float64))
            gv=displacement_tf(relation_tf(gs,tf.constant(ganchors,dtype=tf.float64)),gr)
            gm=normalized_margin(tf.tensordot(gt/tf.linalg.norm(gt),gs,axes=1),threshold_g)
            sm=normalized_margin(torch.dot(st/torch.linalg.vector_norm(st),ss),threshold_s)
            active="SFace" if float(sm.detach())>=float(gm.numpy()) else "GhostFaceNet"
            gbranch=(gm if active=="GhostFaceNet" else tf.constant(0.,tf.float64))
            gbranch-=WEIGHT*tf.tensordot(tf.constant(sv.detach().numpy()),gv,axes=1)
        ggrad=tape.gradient(gbranch,gt)
        sbranch=(sm if active=="SFace" else torch.tensor(0.,dtype=torch.float64))
        sbranch-=WEIGHT*torch.dot(sv,torch.tensor(gv.numpy()))
        sbranch.backward()
        return st.grad.numpy(),ggrad.numpy()
    outcomes={}
    for name,ganchors in (("aligned",ga),("permuted",ga[permutation(n)])):
        sg,gg=partials(se,ge,ganchors)
        eps=1e-6
        finite_s=np.array([(scalar(se+eps*np.eye(5)[i],ge,ganchors)-
                            scalar(se-eps*np.eye(5)[i],ge,ganchors))/(2*eps) for i in range(5)])
        finite_g=np.array([(scalar(se,ge+eps*np.eye(7)[i],ganchors)-
                            scalar(se,ge-eps*np.eye(7)[i],ganchors))/(2*eps) for i in range(7)])
        err_s=float(np.max(np.abs(sg-finite_s)));err_g=float(np.max(np.abs(gg-finite_g)))
        if err_s>1e-5 or err_g>1e-5:
            raise AssertionError(f"{name} framework partial differs from finite difference: {err_s}, {err_g}")
        zero,values=agreement_numpy(s0,g0,[s0],[g0],sa,ganchors)
        if abs(zero)>1e-12 or any(abs(x)>1e-12 for x in values):
            raise AssertionError("Zero edit yielded nonzero relational displacement")
        outcomes[name]={"torch_sface_max_abs_error":err_s,"tensorflow_ghost_max_abs_error":err_g,
                        "objective":scalar(se,ge,ganchors),"zero_displacement_agreement":zero}
    if abs(outcomes["aligned"]["objective"]-outcomes["permuted"]["objective"])<1e-5:
        raise AssertionError("Fixed permutation failed to alter synthetic correspondence")
    print(json.dumps({"synthetic_seed":71,"permutation_seed":PERMUTATION_SEED,
                      "finite_difference_step":1e-6,"results":outcomes},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest",action="store_true",help="Run synthetic math and gradient checks; no model/photo inference")
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--sface-calibration",type=Path)
    parser.add_argument("--ghostface-calibration",type=Path)
    parser.add_argument("--yunet",type=Path)
    parser.add_argument("--sface",type=Path)
    parser.add_argument("--ghostface",type=Path)
    parser.add_argument("--clone-diagnostic",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--ids",nargs="+",default=list(PILOT_IDS))
    parser.add_argument("--diagnostic-replay",action="store_true",
                        help="Replay only frll-021 baseline with identical settings and persisted checkpoints")
    parser.add_argument("--anchors-from",type=Path,
                        help="Original run directory supplying the frozen anchor pool for diagnostic replay")
    args=parser.parse_args()
    if args.selftest:
        selftest()
    else:
        required=("manifest","sface_calibration","ghostface_calibration","yunet",
                  "sface","ghostface","clone_diagnostic","out")
        missing=[name for name in required if getattr(args,name) is None]
        if missing:
            parser.error("Required for inference: "+", ".join(missing))
        args.ids=tuple(args.ids)
        main(args)
