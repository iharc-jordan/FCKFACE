"""Offline SFace identity-subspace diagnostic on disjoint development identities.

No photo optimization, image export, or recognition threshold is performed.
Training images, embeddings, and matrices stay in the explicit external output.
The calibration artifact is used to lock model and preprocessing provenance.
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

from fckface_lab.imaging import decode_image
from fckface_lab.recognition import SFaceModel


EXCLUDED_SCREEN_IDS = ("frll-001", "frll-003", "frll-004", "frll-007",
                       "frll-011", "frll-013", "frll-018", "frll-019")
VIEWS = ("neutral_front", "smiling_front", "neutral_left_3quarter",
         "neutral_right_3quarter")
RANKS = (8, 16)
WITHIN_SHRINKAGE = .10
RANDOM_SEED = 0
EMBEDDING_DIM = 128


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(features: np.ndarray, model: dict | str | Path, rank: int = 8,
            control: str = "learned") -> np.ndarray:
    """Project L2-normalized SFace features into a saved rank-8/16 subspace.

    ``model`` is the external ``subspace.npz`` path or a mapping of its arrays.
    Returned rows are L2 normalized. Projection scores are diagnostics only;
    no threshold or match decision is implied.
    """
    if rank not in RANKS or control not in ("learned", "random"):
        raise ValueError("Use rank 8/16 and learned/random control")
    if isinstance(model, (str, Path)):
        with np.load(model, allow_pickle=False) as data:
            mean = data["mean"].copy()
            directions = data[f"{control}_{rank}"].copy()
    else:
        mean = np.asarray(model["mean"])
        directions = np.asarray(model[f"{control}_{rank}"])
    x = np.asarray(features, dtype=np.float64)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    if (x.shape[1] != EMBEDDING_DIM or mean.shape != (EMBEDDING_DIM,)
            or directions.shape != (EMBEDDING_DIM, rank)
            or not np.isfinite(x).all() or not np.isfinite(directions).all()):
        raise ValueError("Invalid SFace features or saved projection")
    projected = (x - mean) @ directions
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.maximum(norms, 1e-12)
    return projected[0] if single else projected


def pair_diagnostic(projected: np.ndarray, identities: list[str]) -> dict:
    similarities = projected @ projected.T
    upper = np.triu_indices(len(identities), 1)
    same = np.fromiter((identities[i] == identities[j] for i, j in zip(*upper)),
                       dtype=bool, count=len(upper[0]))
    values = similarities[upper]
    def summary(items):
        if not len(items):
            return {"count": 0}
        return {"count": int(len(items)), "mean": float(np.mean(items)),
                "std": float(np.std(items)), "p05": float(np.quantile(items, .05)),
                "p50": float(np.quantile(items, .5)), "p95": float(np.quantile(items, .95))}
    return {"same_identity": summary(values[same]),
            "different_identity": summary(values[~same])}


def freeze_selection(manifest_path: Path, out: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["identity_splits"]
    development = sorted(identity for identity, split in splits.items()
                         if split == "development" and identity not in EXCLUDED_SCREEN_IDS)
    if len(development) != 52 or any(splits[identity] != "development"
                                     for identity in development):
        raise ValueError("Expected 52 nonscreen development identities")
    rows = sorted((row for row in manifest["images"]
                   if row["identity"] in development and row["view"] in VIEWS),
                  key=lambda row: (row["identity"], VIEWS.index(row["view"])))
    if len(rows) != 52 * 4 or len({(row["identity"], row["view"]) for row in rows}) != len(rows):
        raise ValueError("Expected exactly four prespecified views per training identity")
    images = []
    for row in rows:
        path = (manifest_path.parent / row["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or digest(path) != row["sha256"]:
            raise ValueError(f"Manifest image escaped or changed: {row['path']}")
        images.append({"identity": row["identity"], "view": row["view"],
                       "path": str(path), "sha256": row["sha256"]})
    selection = {
        "scope": "identity-subspace-v1 fixed development data; no score-based choice",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "all lexically sorted development IDs except first-eight screen IDs; four fixed views",
        "training_ids": development, "excluded_screen_ids": EXCLUDED_SCREEN_IDS,
        "excluded_calibration_ids": sorted(k for k, v in splits.items() if v == "calibration"),
        "excluded_held_out_ids": sorted(k for k, v in splits.items() if v == "held_out"),
        "views": VIEWS, "images": images,
        "manifest_sha256": digest(manifest_path),
        "minimum_valid_views_per_identity": 2,
        "within_isotropic_shrinkage": WITHIN_SHRINKAGE,
        "ranks": RANKS, "random_control_seed": RANDOM_SEED,
    }
    destination = out / "selection.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return images, selection


def validate_calibration(calibration_path: Path, manifest: Path,
                         yunet: Path, sface: Path):
    artifact = json.loads(calibration_path.read_text(encoding="utf-8"))
    if artifact.get("status") != "complete" or artifact.get("purpose") != "development_native_sface_calibration_only":
        raise ValueError("Need completed native SFace calibration")
    pipeline = artifact["pipeline"]
    for name, path in (("yunet", yunet), ("sface", sface)):
        if digest(path) != pipeline["model_artifacts"][name]["sha256"]:
            raise ValueError(f"Calibration {name} model hash differs")
    root = Path(__file__).parent / "fckface_lab"
    for name in ("recognition.py", "imaging.py", "evaluation.py"):
        if digest(root / name) != pipeline["source_sha256"][f"fckface_lab/{name}"]:
            raise ValueError(f"Calibration preprocessing code differs: {name}")
    if digest(manifest) != artifact["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from calibration")
    detector = pipeline["detector_input"]
    if detector["maximum_side_px"] != 640 or detector["score_threshold"] != .9:
        raise ValueError("Detector setup differs from calibrated pipeline")
    return artifact


def fit(features: np.ndarray, identities: list[str]):
    if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM:
        raise ValueError("Expected normalized 128D SFace embeddings")
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4):
        raise ValueError("Invalid or unnormalized embeddings")
    unique = sorted(set(identities))
    groups = [features[np.array(identities) == identity] for identity in unique]
    if any(len(group) < 2 for group in groups):
        raise ValueError("Fitting needs at least two valid views per included identity")
    means = np.stack([group.mean(axis=0) for group in groups])
    global_mean = means.mean(axis=0)
    within = sum((group - group.mean(axis=0)).T @ (group - group.mean(axis=0))
                 for group in groups) / sum(len(group) - 1 for group in groups)
    centered_means = means - global_mean
    between = centered_means.T @ centered_means / (len(groups) - 1)
    isotropic = float(np.trace(within) / EMBEDDING_DIM)
    if not np.isfinite(isotropic) or isotropic <= 0:
        raise ValueError("Within-identity covariance has no positive scale")
    shrunk = (1 - WITHIN_SHRINKAGE) * within + WITHIN_SHRINKAGE * isotropic * np.eye(EMBEDDING_DIM)
    within_values, within_vectors = np.linalg.eigh(shrunk)
    if np.any(within_values <= 0):
        raise ValueError("Shrunk within covariance is not positive definite")
    whitening = (within_vectors / np.sqrt(within_values)) @ within_vectors.T
    symmetric = whitening @ between @ whitening
    symmetric = (symmetric + symmetric.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    directions = whitening @ eigenvectors[:, order[:max(RANKS)]]
    if not np.isfinite(directions).all() or np.any(eigenvalues[:max(RANKS)] <= 0):
        raise ValueError("Generalized eigenvectors are invalid")
    rng = np.random.default_rng(RANDOM_SEED)
    random_basis, _ = np.linalg.qr(rng.standard_normal((EMBEDDING_DIM, max(RANKS))))
    saved = {"mean": global_mean.astype(np.float64)}
    diagnostics = {"training_identity_count": len(unique),
                   "valid_embedding_count": len(features),
                   "within_isotropic_scale": isotropic,
                   "shrunk_within_condition_number": float(within_values[-1] / within_values[0]),
                   "retained_generalized_eigenvalues": eigenvalues[:max(RANKS)].tolist(),
                   "generalized_eigenvalue_8": float(eigenvalues[7]),
                   "generalized_eigenvalue_16": float(eigenvalues[15]),
                   "whitening_condition_number": float(np.sqrt(within_values[-1] / within_values[0])),
                   "all_finite": True}
    for rank in RANKS:
        learned = directions[:, :rank]
        _, spectrum, right = np.linalg.svd(learned, full_matrices=False)
        random = random_basis[:, :rank] @ np.diag(spectrum) @ right
        if not np.allclose(np.linalg.svd(random, compute_uv=False), spectrum, rtol=1e-10, atol=1e-10):
            raise ValueError("Random control did not match singular spectrum")
        saved[f"learned_{rank}"] = learned
        saved[f"random_{rank}"] = random
        diagnostics[f"rank_{rank}"] = {
            "learned_singular_values": spectrum.tolist(),
            "random_singular_values": np.linalg.svd(random, compute_uv=False).tolist(),
            "learned_frobenius_norm": float(np.linalg.norm(learned)),
            "random_frobenius_norm": float(np.linalg.norm(random)),
            "learned_max_abs_coefficient": float(np.max(np.abs(learned))),
            "random_max_abs_coefficient": float(np.max(np.abs(random))),
            "within_metric_orthogonality_error": float(
                np.linalg.norm(learned.T @ shrunk @ learned - np.eye(rank))),
        }
    return saved, diagnostics


def run(args):
    out = args.out.resolve()
    source_root = Path(__file__).resolve().parents[1]
    if out == source_root or out.is_relative_to(source_root):
        raise ValueError("Private subspace outputs must be outside the source repository")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Subspace output already exists: {out}")
    artifact = validate_calibration(args.calibration, args.manifest, args.yunet, args.sface)
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    images, selection = freeze_selection(args.manifest, out)
    selection_seconds = time.perf_counter() - began
    # Selection and all image hashes are fixed on disk before any embedding.
    cv2.setNumThreads(2)
    model = SFaceModel(args.yunet, args.sface)
    features = []
    valid_identities = []
    valid_views = []
    detections = []
    began_acquisition = time.perf_counter()
    for index, row in enumerate(images, start=1):
        result = model.embed(decode_image(Path(row["path"])))
        detections.append({"identity": row["identity"], "view": row["view"],
                           "sha256": row["sha256"], "status": result.status,
                           "reason": result.reason, "detection_count": result.detection_count,
                           "selected_box_xyxy": result.selected_box})
        if result.status == "valid":
            features.append(result.feature.astype(np.float64))
            valid_identities.append(row["identity"])
            valid_views.append(row["view"])
        if index % 40 == 0:
            print(f"Embedded {index}/{len(images)} fixed development views", flush=True)
    acquisition_seconds = time.perf_counter() - began_acquisition
    counts = {identity: valid_identities.count(identity)
              for identity in selection["training_ids"]}
    included = sorted(identity for identity, count in counts.items() if count >= 2)
    excluded_insufficient = {identity: count for identity, count in counts.items() if count < 2}
    if len(included) < 16:
        raise ValueError("Too few identities with two valid views for rank-16 subspace")
    keep = np.array([identity in included for identity in valid_identities])
    x = np.asarray(features, dtype=np.float64)[keep]
    ids = [identity for identity, accepted in zip(valid_identities, keep) if accepted]
    views = [view for view, accepted in zip(valid_views, keep) if accepted]
    began_fit = time.perf_counter()
    matrices, diagnostics = fit(x, ids)
    fit_seconds = time.perf_counter() - began_fit
    began_diagnostic = time.perf_counter()
    pair_stats = {}
    for rank in RANKS:
        for control in ("learned", "random"):
            projected = project(x, matrices, rank, control)
            pair_stats[f"{control}_{rank}"] = pair_diagnostic(projected, ids)
    diagnostic_seconds = time.perf_counter() - began_diagnostic
    model_path = out / "subspace.npz"
    np.savez_compressed(model_path, **matrices)
    embedding_path = out / "training-embeddings.npz"
    np.savez_compressed(embedding_path, features=x.astype(np.float32),
                        identities=np.array(ids), views=np.array(views))
    report = {
        "scope": "Offline development-only SFace identity subspace; no recognition decision or photo optimization",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": digest(out / "selection.json"),
        "subspace_sha256": digest(model_path),
        "private_training_embeddings_sha256": digest(embedding_path),
        "source_sha256": {"identity_subspace.py": digest(Path(__file__)),
                          "recognition.py": digest(Path(__file__).parent / "fckface_lab" / "recognition.py"),
                          "imaging.py": digest(Path(__file__).parent / "fckface_lab" / "imaging.py"),
                          "manifest": digest(args.manifest), "calibration": digest(args.calibration),
                          "yunet": digest(args.yunet), "sface": digest(args.sface)},
        "versions": {"python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__},
        "calibration_threshold_recorded_not_applied": artifact["calibration"]["threshold"],
        "opencv_threads": cv2.getNumThreads(),
        "selection_identity_count": len(selection["training_ids"]),
        "selection_image_count": len(images),
        "included_identity_count": len(included), "included_identity_ids": included,
        "insufficient_valid_views": excluded_insufficient,
        "valid_embedding_count_used": len(x),
        "excluded_screen_ids": EXCLUDED_SCREEN_IDS,
        "excluded_calibration_identity_count": len(selection["excluded_calibration_ids"]),
        "excluded_held_out_identity_count": len(selection["excluded_held_out_ids"]),
        "detection_results": detections,
        "numerical_diagnostics": diagnostics,
        "projected_pair_diagnostics": pair_stats,
        "timings_seconds": {"selection_and_hash_freeze": selection_seconds,
                            "image_decode_detection_embedding": acquisition_seconds,
                            "covariance_eigenproblem": fit_seconds,
                            "projected_pair_diagnostics": diagnostic_seconds,
                            "total": time.perf_counter() - began},
        "limitations": [
            "Only development identities informed this subspace; same/different pair summaries are in-sample diagnostics.",
            "The calibrated SFace threshold is provenance only and was not applied to projected cosines.",
            "Rank and shrinkage were frozen before fitting; no model-based photo edits were attempted.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection_images": len(images), "included_identities": len(included),
                      "used_embeddings": len(x), "total_seconds": report["timings_seconds"]["total"],
                      "out": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "calibration", "yunet", "sface", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
