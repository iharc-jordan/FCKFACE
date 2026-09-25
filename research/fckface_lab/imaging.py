"""Decode and export the exact pixels submitted to recognition.

All coordinates are ``(left, top, right, bottom)`` in decoded RGB pixels.
Generated JPEGs are decoded again before they reach a model. Private EXIF and
ICC data are intentionally not copied into generated files.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageOps

Box: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageVariant:
    name: str
    status: str  # valid or inconclusive
    image: np.ndarray | None
    jpeg: bytes | None
    target_box: Box | None
    reason: str | None = None


def decode_image(source: bytes | str | Path) -> np.ndarray:
    """Apply orientation and embedded ICC profile, then return uint8 sRGB RGB."""
    blob = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    with Image.open(BytesIO(blob)) as opened:
        image = ImageOps.exif_transpose(opened)
        icc = image.info.get("icc_profile")
        if icc:
            try:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(icc))
                target_profile = ImageCms.createProfile("sRGB")
                image = ImageCms.profileToProfile(
                    image.convert("RGB"), source_profile, target_profile,
                    outputMode="RGB",
                )
            except (OSError, ValueError) as exc:
                raise ValueError("Invalid embedded ICC profile") from exc
        else:
            image = image.convert("RGB")
        image.load()
        return np.asarray(image, dtype=np.uint8).copy()


def export_jpeg(image: np.ndarray, *, quality: int = 95, subsampling: int = 0) -> bytes:
    """Re-encode RGB pixels without metadata (default JPEG Q95, 4:4:4)."""
    _check_rgb(image)
    out = BytesIO()
    Image.fromarray(image, mode="RGB").save(
        out, format="JPEG", quality=quality, subsampling=subsampling,
    )
    return out.getvalue()


def _check_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB pixels")
    if min(image.shape[:2]) < 2:
        raise ValueError("Image is too small")


def _box_in_image(box: Box, width: int, height: int, *, min_side: float = 16) -> bool:
    left, top, right, bottom = box
    return (0 <= left < right <= width and 0 <= top < bottom <= height
            and right - left >= min_side and bottom - top >= min_side)


def _encode_variant(name: str, image: Image.Image, box: Box | None,
                    *, quality: int = 95, subsampling: int = 0) -> ImageVariant:
    if box is not None and not _box_in_image(box, *image.size):
        return ImageVariant(name, "inconclusive", None, None, box, "target_outside_or_too_small")
    payload = export_jpeg(np.asarray(image.convert("RGB"), dtype=np.uint8),
                          quality=quality, subsampling=subsampling)
    return ImageVariant(name, "valid", decode_image(payload), payload, box)


def make_variants(exported_jpeg: bytes, target_box: Box | None = None) -> dict[str, ImageVariant]:
    """Build fixed post-export conditions from the *decoded* base JPEG.

    ``target_box`` is in the base JPEG's decoded coordinates. A variant whose
    transformed target falls outside the frame or below 16 pixels is marked
    inconclusive. That condition can never count as a nonmatch.
    """
    base = decode_image(exported_jpeg)
    _check_rgb(base)
    h, w = base.shape[:2]
    if target_box is not None and not _box_in_image(target_box, w, h):
        raise ValueError("Invalid target box for exported image")
    original = Image.fromarray(base, mode="RGB")
    results: dict[str, ImageVariant] = {
        "export": ImageVariant("export", "valid", base, exported_jpeg, target_box),
        "jpeg85_420": _encode_variant("jpeg85_420", original, target_box,
                                      quality=85, subsampling=2),
        "jpeg75_420": _encode_variant("jpeg75_420", original, target_box,
                                      quality=75, subsampling=2),
    }
    scale = min(1.0, 960 / max(w, h))
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    scaled_box = (tuple((target_box[0] * new_w / w, target_box[1] * new_h / h,
                         target_box[2] * new_w / w, target_box[3] * new_h / h))
                  if target_box is not None else None)
    results["resize960"] = _encode_variant(
        "resize960", original.resize((new_w, new_h), Image.Resampling.LANCZOS),
        scaled_box,
    )
    half = original.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)
    results["half_restore"] = _encode_variant(
        "half_restore", half.resize((w, h), Image.Resampling.LANCZOS), target_box,
    )
    left, top = round(w * .05), round(h * .05)
    right, bottom = w - left, h - top
    cropped_box = (tuple((target_box[0] - left, target_box[1] - top,
                          target_box[2] - left, target_box[3] - top))
                   if target_box is not None else None)
    results["crop90"] = _encode_variant(
        "crop90", original.crop((left, top, right, bottom)), cropped_box,
    )
    results["blur"] = _encode_variant(
        "blur", original.filter(ImageFilter.GaussianBlur(radius=1)), target_box,
    )
    return results
