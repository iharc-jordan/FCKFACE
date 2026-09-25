"""Empirical native-score threshold calibration from labelled pairs."""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
import math
from typing import Sequence


@dataclass(frozen=True)
class CalibrationPair:
    same_identity: bool
    score: float | None  # None means detection, selection, or feature failure


@dataclass(frozen=True)
class Calibration:
    threshold: float
    target_fmr: float
    empirical_fmr: float
    empirical_fnmr: float
    valid_coverage: float
    valid_own: int
    valid_impostor: int
    invalid_pairs: int

    def matches(self, score: float) -> bool:
        if not math.isfinite(score):
            raise ValueError("Non-finite similarity score")
        return score >= self.threshold

    def as_dict(self) -> dict[str, float | int]:
        """Serializable frozen threshold and calibration evidence."""
        return vars(self).copy()


def calibrate(pairs: Sequence[CalibrationPair], target_fmr: float = .001) -> Calibration:
    """Choose the most permissive threshold with empirical FMR <= target.

    A match is ``score >= threshold``. The threshold is the next float above
    the boundary impostor score, so ties at that boundary cannot leak into a
    nominally compliant FMR. The finite sample estimate is reported; it is not
    a population guarantee. Use disjoint calibration and evaluation identities.
    """
    if not 0 <= target_fmr < 1:
        raise ValueError("target_fmr must be in [0, 1)")
    if not pairs:
        raise ValueError("No labelled calibration pairs")
    own: list[float] = []
    impostors: list[float] = []
    invalid = 0
    for pair in pairs:
        if pair.score is None or not math.isfinite(pair.score):
            invalid += 1
            continue
        (own if pair.same_identity else impostors).append(float(pair.score))
    if not own or not impostors:
        raise ValueError("Calibration needs valid own-identity and impostor pairs")
    # At most k false matches are allowed. Move one representable float above
    # the (k+1)th largest impostor, excluding *all* ties at that boundary.
    sorted_impostors = sorted(impostors)
    k = math.floor(target_fmr * len(impostors))
    threshold = math.nextafter(sorted_impostors[-(k + 1)], math.inf)
    fmr = (len(impostors) - bisect_left(sorted_impostors, threshold)) / len(impostors)
    fnmr = sum(score < threshold for score in own) / len(own)
    return Calibration(threshold, target_fmr, fmr, fnmr,
                       (len(own) + len(impostors)) / len(pairs),
                       len(own), len(impostors), invalid)
