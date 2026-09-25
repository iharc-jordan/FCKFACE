"""H12 appearance-only contrast-chimera renderer (development data only).

This script uses YuNet solely to locate the clean face and eyes. It never loads
recognition weights, embeds a face, or reads another view of either person.
All outputs, including contact sheets, must stay outside the public repository.
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
from PIL import Image, ImageDraw

from fckface_lab.imaging import decode_image, export_jpeg
from fckface_lab.patterns import actual_distortion, face_mask
from fckface_lab.recognition import (
    DETECTOR_SCORE_THRESHOLD, detector_frame, restore_face_coordinates,
    select_face,
)


IDS = ("frll-032", "frll-037")
SOURCE_VIEW = "neutral_front"
ARMS = ("eye_positive_chimera", "monotone_control", "full_negative_diagnostic")
PALETTE_DARK = np.array((31, 45, 111), dtype=np.float32)
PALETTE_LIGHT = np.array((139, 213, 231), dtype=np.float32)
EYE_BAND = {"u_abs_max": .95, "v_min": -.35, "v_max": .30}
FACE_MASK_THRESHOLD = .5
RMS_TOLERANCE = .25
CHANGED_FRACTION_TOLERANCE = .02
BRACKET_STEPS = 64
BISECTION_STEPS = 18


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preflight(manifest_path: Path, yunet_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "frll":
        raise ValueError("Expected FRLL manifest")
    if any(manifest["identity_splits"].get(identity) != "development" for identity in IDS):
        raise ValueError("Both fixed H12 identities must be in development")
    cases = []
    for identity in IDS:
        matches = [row for row in manifest["images"]
                   if row["identity"] == identity and row["view"] == SOURCE_VIEW
                   and row["split"] == "development"]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one development source for {identity}")
        row = matches[0]
        path = (manifest_path.parent / row["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Source path escaped dataset root")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {identity}")
        cases.append({"identity": identity, "source": path, "sha256": row["sha256"]})
    if not yunet_path.is_file():
        raise FileNotFoundError(yunet_path)
    return cases


def detect_clean(detector: cv2.FaceDetectorYN, source: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    frame = detector_frame(bgr)
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    faces = restore_face_coordinates(
        faces, (source.shape[1], source.shape[0]), (frame.shape[1], frame.shape[0]))
    face, reason = select_face(faces)
    if face is None:
        raise ValueError(f"Clean YuNet face selection inconclusive: {reason}")
    return face


def masks(source: np.ndarray, face: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    box = tuple(map(float, face[:4]))  # YuNet XYWH
    eyes = np.asarray(face[4:8], dtype=np.float64).reshape(2, 2)
    eyes = eyes[np.argsort(eyes[:, 0])]
    middle = eyes.mean(axis=0)
    between = eyes[1] - eyes[0]
    distance = float(np.linalg.norm(between))
    if distance < 8 or not np.isfinite(distance):
        raise ValueError("Unusable clean eye landmarks")
    axis = between / distance
    down = np.array((-axis[1], axis[0]))
    yy, xx = np.mgrid[:source.shape[0], :source.shape[1]].astype(np.float32)
    u = ((xx - middle[0]) * axis[0] + (yy - middle[1]) * axis[1]) / distance
    v = ((xx - middle[0]) * down[0] + (yy - middle[1]) * down[1]) / distance
    eye_band = (np.abs(u) <= EYE_BAND["u_abs_max"])
    eye_band &= (v >= EYE_BAND["v_min"]) & (v <= EYE_BAND["v_max"])
    face_support = face_mask(source.shape, box) >= FACE_MASK_THRESHOLD
    return face_support, face_support & ~eye_band, box


def render(source: np.ndarray, support: np.ndarray, mode: str, b: float = 0.) -> tuple[np.ndarray, float]:
    """A hard mask preserves exact source anatomy outside the graphic area.

    Palette luminance is strictly increasing in t. Thus t=1-y has negative
    luminance slope in every treated pixel; t=clip(y+b) is nondecreasing.
    """
    rgb = source[support].astype(np.float32)
    y = (rgb @ np.array((.2126, .7152, .0722), dtype=np.float32)) / 255.
    if mode == "negative":
        t = 1. - y
        clipping_fraction = 0.
    elif mode == "monotone":
        raw = y + np.float32(b)
        clipping_fraction = float(np.mean((raw < 0) | (raw > 1)))
        t = np.clip(raw, 0., 1.)
    else:
        raise ValueError(mode)
    output = source.copy()
    output[support] = np.rint(PALETTE_DARK + t[:, None] *
                               (PALETTE_LIGHT - PALETTE_DARK)).astype(np.uint8)
    return output, clipping_fraction


def exported_metrics(clean_export: np.ndarray, image: np.ndarray, box: tuple) -> tuple[bytes, dict]:
    jpeg = export_jpeg(image)
    decoded = decode_image(jpeg)
    return jpeg, actual_distortion(clean_export, decoded, box)


def match_monotone(source: np.ndarray, support: np.ndarray, box: tuple,
                   clean_export: np.ndarray, target_rms: float) -> tuple[float, str, dict, bytes, float, dict]:
    """Find nearest-zero b by exact-JPEG RMS alone; never use a recognizer."""
    tried: dict[float, tuple[bytes, dict, float]] = {}

    def evaluate(b: float) -> float:
        if b not in tried:
            image, clipping = render(source, support, "monotone", b)
            jpeg, metrics = exported_metrics(clean_export, image, box)
            tried[b] = (jpeg, metrics, clipping)
        return tried[b][1]["face_rms"] - target_rms

    zero = evaluate(0.)
    bracket = None
    path = "zero"
    if abs(zero) > RMS_TOLERANCE:
        for sign, label in ((-1, "negative_b_first"), (1, "positive_b_fallback")):
            previous = 0.
            previous_value = zero
            for step in range(1, BRACKET_STEPS + 1):
                current = sign * step / BRACKET_STEPS
                value = evaluate(current)
                if previous_value * value <= 0:
                    bracket = (previous, current)
                    path = label
                    break
                previous, previous_value = current, value
            if bracket is not None:
                break
        if bracket is not None:
            left, right = bracket
            for _ in range(BISECTION_STEPS):
                middle = (left + right) / 2
                value = evaluate(middle)
                if evaluate(left) * value <= 0:
                    right = middle
                else:
                    left = middle
        else:
            path = "no_bracket"
    best = min(tried, key=lambda b: (abs(tried[b][1]["face_rms"] - target_rms), abs(b), b))
    jpeg, metrics, clipping = tried[best]
    search = {"path": path, "bracket": list(bracket) if bracket else None,
              "samples": len(tried), "b": best, "rms_difference": metrics["face_rms"] - target_rms,
              "rms_tolerance": RMS_TOLERANCE,
              "root_or_zero_found": path != "no_bracket"}
    return best, path, metrics, jpeg, clipping, search


def contact_sheet(source: np.ndarray, outputs: dict[str, bytes], box: tuple,
                  destination: Path, crop: bool) -> None:
    labels = ("Clean source", "Eye-positive chimera", "Monotone control", "Full negative")
    images = [Image.fromarray(source)] + [Image.open(Path(outputs[arm])) for arm in ARMS]
    if crop:
        x, y, w, h = box
        pad = .16
        rect = (max(0, int(x - pad * w)), max(0, int(y - pad * h)),
                min(source.shape[1], int(x + (1 + pad) * w)),
                min(source.shape[0], int(y + (1 + pad) * h)))
        images = [image.crop(rect) for image in images]
    tile_size = (440, 580) if not crop else (440, 500)
    sheet = Image.new("RGB", (tile_size[0] * 4, tile_size[1] + 30), "#f5f5f5")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(zip(labels, images, strict=True)):
        image.thumbnail((tile_size[0] - 12, tile_size[1] - 12), Image.Resampling.LANCZOS)
        x = index * tile_size[0] + (tile_size[0] - image.width) // 2
        y = 30 + (tile_size[1] - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((index * tile_size[0] + 8, 8), label, fill="#111111")
    sheet.save(destination, format="PNG")


def run(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(repo):
        raise ValueError("All rendered data must stay outside Git")
    cases = preflight(args.manifest, args.yunet)
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / "executed-source.py")
    source_hash = sha256(Path(__file__))
    protocol = {
        "hypothesis": "H12 appearance-first eye-positive contrast chimera, no recognition score",
        "status": "frozen_before_render", "identities": list(IDS), "view": SOURCE_VIEW,
        "arms": list(ARMS), "source_sha256": {r["identity"]: r["sha256"] for r in cases},
        "manifest_sha256": sha256(args.manifest), "yunet_sha256": sha256(args.yunet),
        "executed_source_sha256": source_hash,
        "detector": "YuNet on clean source only; 640px longest-side frame; threshold 0.9; unique face",
        "face_support": "existing face_mask >= 0.5; hard edge, no blend",
        "eye_band": {"coordinates": "u,v in interocular-distance units, image-left to right eye axis",
                     **EYE_BAND},
        "palette_rgb": {"dark": PALETTE_DARK.astype(int).tolist(),
                        "light": PALETTE_LIGHT.astype(int).tolist()},
        "luma": "BT.709 coefficients on decoded sRGB RGB/255",
        "chimera": "P(1-y) outside original-positive eye band; strictly negative core slope",
        "control": "P(clamp(y+b,0,1)) outside same eye band; nondecreasing slope",
        "full_negative": "P(1-y) over entire oval face; diagnostic, not RMS-matched",
        "control_match": {"metric": "face RMS of decoded Q95 4:4:4 JPEG against decoded clean Q95 4:4:4 JPEG",
                          "search": "b=0, then nearest-zero sign bracket in [-1,0] at 1/64 intervals; if none, [0,1]; 18 bisections",
                          "rms_tolerance": RMS_TOLERANCE,
                          "changed_face_fraction_tolerance": CHANGED_FRACTION_TOLERANCE,
                          "no_bracket_or_unbalanced": "keep six fixed outputs; no isolated polarity inference"},
        "visual_gate": "root reviews full photos and face crops before any recognition scoring",
        "recognition": "none in this renderer; reserved final models remain untouched",
    }
    write_json(args.out / "protocol.json", protocol)
    started = perf_counter()
    results = {"status": "in_progress", "protocol_sha256": sha256(args.out / "protocol.json"),
               "source_sha256": source_hash, "cases": []}
    detector = cv2.FaceDetectorYN.create(
        str(args.yunet), "", (320, 320), score_threshold=DETECTOR_SCORE_THRESHOLD)
    try:
        for case in cases:
            if sha256(case["source"]) != case["sha256"]:
                raise ValueError(f"Source changed before rendering: {case['identity']}")
            source = decode_image(case["source"])
            face = detect_clean(detector, source)
            face_support, treatment_support, box = masks(source, face)
            clean_export = decode_image(export_jpeg(source))
            negative, _ = render(source, treatment_support, "negative")
            negative_jpeg, negative_metrics = exported_metrics(clean_export, negative, box)
            full_negative, _ = render(source, face_support, "negative")
            full_jpeg, full_metrics = exported_metrics(clean_export, full_negative, box)
            b, path, control_metrics, control_jpeg, clipping, search = match_monotone(
                source, treatment_support, box, clean_export, negative_metrics["face_rms"])
            fraction_gap = control_metrics["changed_fraction_face"] - negative_metrics["changed_fraction_face"]
            balanced = (path != "no_bracket" and abs(search["rms_difference"]) <= RMS_TOLERANCE
                        and abs(fraction_gap) <= CHANGED_FRACTION_TOLERANCE)
            directory = args.out / case["identity"]
            directory.mkdir()
            blobs = {"eye_positive_chimera": negative_jpeg,
                     "monotone_control": control_jpeg,
                     "full_negative_diagnostic": full_jpeg}
            files = {}
            for arm, blob in blobs.items():
                path_out = directory / f"{arm}.jpg"
                path_out.write_bytes(blob)
                files[arm] = {"path": str(path_out), "sha256": sha256(path_out), "bytes": len(blob)}
            contact_sheet(source, {arm: row["path"] for arm, row in files.items()},
                          box, directory / "full-contact.png", crop=False)
            contact_sheet(source, {arm: row["path"] for arm, row in files.items()},
                          box, directory / "face-contact.png", crop=True)
            results["cases"].append({
                "identity": case["identity"], "source_sha256": case["sha256"],
                "clean_yunet_box_xywh": list(box), "clean_yunet_landmarks": face[4:14].tolist(),
                "image_dimensions": [source.shape[1], source.shape[0]],
                "geometric_face_pixels": int(face_support.sum()),
                "geometric_treatment_pixels": int(treatment_support.sum()),
                "support_identical_chimera_control": True,
                "arms": files,
                "chimera_metrics": negative_metrics, "control_metrics": control_metrics,
                "full_negative_metrics": full_metrics, "control_match": search,
                "changed_fraction_difference": fraction_gap,
                "control_clipped_t_fraction": clipping,
                "balanced_polarity_comparison": balanced,
                "full_contact_sha256": sha256(directory / "full-contact.png"),
                "face_contact_sha256": sha256(directory / "face-contact.png"),
            })
            write_json(args.out / "progress.json", results)
            print(case["identity"], "rendered", "balanced" if balanced else "unbalanced", flush=True)
        if len(results["cases"]) != 2 or sum(len(row["arms"]) for row in results["cases"]) != 6:
            raise RuntimeError("Incomplete six-JPEG render")
        results["seconds_after_preflight"] = perf_counter() - started
        results["status"] = "appearance_review_pending"
        write_json(args.out / "frozen-appearance.json", results)
    except Exception as exc:
        write_json(args.out / "failure.json", {"error": type(exc).__name__, "detail": str(exc),
                                               "completed_cases": len(results["cases"])})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--yunet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
