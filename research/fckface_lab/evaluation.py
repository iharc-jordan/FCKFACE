"""Per-reference recognition results and a conservative held-out release gate."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from .calibration import Calibration
from .recognition import EmbeddingResult

HELDOUT_MODELS = ("AdaFace", "MagFace", "EdgeFace", "TransFace")


@dataclass(frozen=True)
class Reference:
    name: str
    identity: str
    feature: np.ndarray


@dataclass(frozen=True)
class GalleryEvaluation:
    status: str  # valid, invalid, or inconclusive
    per_reference: Mapping[str, float] = field(default_factory=dict)
    matching_references: tuple[str, ...] = ()
    own_source_score: float | None = None
    own_identity_matched: bool | None = None
    reason: str | None = None

    @property
    def nonmatch(self) -> bool:
        return self.status == "valid" and not self.matching_references


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if not a.size or a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Invalid embedding pair")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Zero or invalid embedding norm")
    return float(np.dot(a, b) / norm)


def evaluate_gallery(
    probe: EmbeddingResult,
    references: Sequence[Reference],
    calibration: Calibration,
    *,
    own_identity: str,
    own_source_name: str,
    expected_reference_names: Sequence[str] | None = None,
) -> GalleryEvaluation:
    """Compare every native reference, including the person's own source.

    Any missing or invalid reference makes the whole gallery inconclusive;
    partial gallery results cannot be counted as a nonmatch. This function
    accepts one model's calibrated scores only. Never average across models.
    """
    if probe.status != "valid" or probe.feature is None:
        return GalleryEvaluation("inconclusive" if probe.status == "inconclusive" else "invalid",
                                 reason=f"probe_{probe.reason or probe.status}")
    names = [reference.name for reference in references]
    if not names or len(names) != len(set(names)):
        return GalleryEvaluation("invalid", reason="empty_or_duplicate_reference_names")
    expected = set(expected_reference_names or names)
    if set(names) != expected or own_source_name not in expected:
        return GalleryEvaluation("inconclusive", reason="missing_or_unexpected_references")
    source = next(ref for ref in references if ref.name == own_source_name)
    if source.identity != own_identity:
        return GalleryEvaluation("invalid", reason="own_source_identity_mismatch")
    scores: dict[str, float] = {}
    matches: list[str] = []
    own_matched = False
    try:
        for reference in references:
            score = cosine(probe.feature, reference.feature)
            scores[reference.name] = score
            if calibration.matches(score):
                matches.append(reference.name)
                own_matched |= reference.identity == own_identity
    except ValueError:
        return GalleryEvaluation("inconclusive", scores, tuple(matches),
                                 reason="invalid_reference_embedding")
    return GalleryEvaluation("valid", scores, tuple(matches), scores[own_source_name],
                             own_matched)


@dataclass(frozen=True)
class ModelPhotoResults:
    # Keys are (processing_condition, gallery_name). Controls are the clean
    # photo, measured before an altered candidate is considered.
    controls: Mapping[tuple[str, str], GalleryEvaluation]
    altered: Mapping[tuple[str, str], GalleryEvaluation]


@dataclass(frozen=True)
class PhotoResult:
    photo_id: str
    models: Mapping[str, ModelPhotoResults]


@dataclass(frozen=True)
class GateResult:
    passed: bool
    eligible_count: int
    successful_count: int
    tested_count: int
    success_rate: float | None
    eligible_photo_ids: tuple[str, ...]
    successful_photo_ids: tuple[str, ...]
    ineligible_photo_ids: tuple[str, ...]
    failed_photo_ids: tuple[str, ...]
    reasons: Mapping[str, str]


@dataclass(frozen=True)
class DevelopmentReport:
    """Descriptive development-model counts; never a release decision."""

    model_name: str
    total_comparisons: int
    valid_comparisons: int
    invalid_comparisons: int
    inconclusive_comparisons: int
    matching_comparisons: int
    nonmatching_comparisons: int
    per_reference_scores: Mapping[str, Mapping[tuple[str, str], Mapping[str, float]]]


def development_report(
    photos: Sequence[PhotoResult],
    *,
    model_name: str = "SFace",
    conditions: Sequence[str],
    galleries: Sequence[str],
) -> DevelopmentReport:
    """Summarize altered-image evidence without pooling raw model scores."""
    if not conditions or not galleries:
        raise ValueError("Conditions and galleries are required")
    valid = invalid = inconclusive = matching = nonmatching = 0
    scores: dict[str, dict[tuple[str, str], Mapping[str, float]]] = {}
    for photo in photos:
        scores[photo.photo_id] = {}
        for condition in conditions:
            for gallery in galleries:
                key = (condition, gallery)
                result = (photo.models.get(model_name).altered.get(key)
                          if model_name in photo.models else None)
                if result is None or result.status == "inconclusive":
                    inconclusive += 1
                elif result.status == "invalid":
                    invalid += 1
                elif result.status == "valid":
                    valid += 1
                    matching += bool(result.matching_references)
                    nonmatching += not result.matching_references
                    scores[photo.photo_id][key] = dict(result.per_reference)
                else:
                    invalid += 1
    return DevelopmentReport(model_name, len(photos) * len(conditions) * len(galleries),
                             valid, invalid, inconclusive, matching, nonmatching, scores)


def release_gate(
    photos: Sequence[PhotoResult],
    *,
    conditions: Sequence[str],
    galleries: Sequence[str],
    minimum_success_rate: float = .95,
) -> GateResult:
    """Judge the four frozen, independent recognizers on eligible new photos.

    Eligibility requires a *valid matching clean control* for every named
    model, condition, and gallery. For an eligible photo, a model succeeds only
    if every altered comparison is valid and has zero gallery matches. At least
    three of four models must succeed. Missing altered data is failure. SFace
    is deliberately excluded from this held-out gate.
    """
    if (not conditions or not galleries or len(set(conditions)) != len(conditions)
            or len(set(galleries)) != len(galleries)
            or not 0 < minimum_success_rate <= 1):
        raise ValueError("Nonempty unique conditions/galleries and a valid rate required")
    keys = {(condition, gallery) for condition in conditions for gallery in galleries}
    eligible: list[str] = []
    successful: list[str] = []
    ineligible: list[str] = []
    failed: list[str] = []
    reasons: dict[str, str] = {}
    seen: set[str] = set()
    for photo in photos:
        if photo.photo_id in seen:
            raise ValueError("Duplicate photo ID")
        seen.add(photo.photo_id)
        control_ok = all(
            model in photo.models
            and all((key in photo.models[model].controls
                     and photo.models[model].controls[key].status == "valid"
                     and photo.models[model].controls[key].own_identity_matched is True)
                    for key in keys)
            for model in HELDOUT_MODELS
        )
        if not control_ok:
            ineligible.append(photo.photo_id)
            reasons[photo.photo_id] = "clean_control_missing_invalid_or_unmatched"
            continue
        eligible.append(photo.photo_id)
        passing_models = sum(
            all(key in photo.models[model].altered
                and photo.models[model].altered[key].nonmatch for key in keys)
            for model in HELDOUT_MODELS
        )
        if passing_models >= 3:
            successful.append(photo.photo_id)
        else:
            failed.append(photo.photo_id)
            reasons[photo.photo_id] = f"only_{passing_models}_of_4_models_nonmatching_all_conditions"
    rate = len(successful) / len(eligible) if eligible else None
    return GateResult(
        bool(rate is not None and rate >= minimum_success_rate),
        len(eligible), len(successful), len(photos), rate,
        tuple(eligible), tuple(successful), tuple(ineligible), tuple(failed), reasons,
    )
