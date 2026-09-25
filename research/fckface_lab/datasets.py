"""Acquire the permitted FRLL research data and freeze identity-disjoint splits.

Images and manifests belong in an explicit directory outside the source checkout.
No photos or embeddings are uploaded by this module.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request
import zipfile

ARTICLE = "https://api.figshare.com/v2/articles/5047666/versions/5"
SOURCE = "https://doi.org/10.6084/m9.figshare.5047666.v5"
SPLIT_VERSION = "frll-identities-v1"
VIEWS = (
    "neutral_front", "smiling_front", "neutral_left_3quarter",
    "neutral_right_3quarter", "smiling_left_3quarter", "smiling_right_3quarter",
    "neutral_left_profile", "neutral_right_profile", "smiling_left_profile",
    "smiling_right_profile",
)


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def identity_split(identities: list[str]) -> dict[str, str]:
    """Freeze by identity, independent of recognition scores or appearance."""
    ids = sorted(set(identities), key=lambda value: hashlib.sha256(
        f"{SPLIT_VERSION}:{value}".encode()).hexdigest())
    if len(ids) != 102:
        raise ValueError(f"Expected 102 FRLL identities, found {len(ids)}")
    return {value: ("development" if index < 60 else
                    "calibration" if index < 80 else "held_out")
            for index, value in enumerate(ids)}


def fetch_frll(root: Path, views: tuple[str, ...] = VIEWS) -> Path:
    root = root.expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if root == source_root or root.is_relative_to(source_root):
        raise ValueError("Use a data directory outside the source repository")
    if not views or any(view not in VIEWS for view in views):
        raise ValueError("Unknown or empty FRLL views")
    dataset_root = root / "frll"
    archive_root = dataset_root / "archives"
    image_root = dataset_root / "images"
    archive_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE, timeout=60) as response:
        metadata = json.load(response)
    if metadata.get("version") != 5 or metadata.get("license", {}).get("name") != "CC BY 4.0":
        raise ValueError("Unexpected upstream dataset version or license; review before use")
    (dataset_root / "source-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    files = {item["name"]: item for item in metadata["files"]}
    archives = []
    for view in views:
        entry = files[f"{view}.zip"]
        archive = archive_root / entry["name"]
        if not archive.exists() or digest(archive, "md5") != entry["computed_md5"]:
            partial = archive.with_suffix(".zip.partial")
            request = urllib.request.Request(entry["download_url"], headers={"User-Agent": "FCKFACE-research/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            if digest(partial, "md5") != entry["computed_md5"]:
                raise ValueError(f"Upstream checksum mismatch for {entry['name']}")
            partial.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            for entry_info in bundle.infolist():
                name = PurePosixPath(entry_info.filename)
                if entry_info.is_dir() or "__MACOSX" in name.parts or name.name.startswith("."):
                    continue
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("Unsafe dataset archive path")
                if name.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tem"):
                    continue
                target = (image_root / Path(*name.parts)).resolve()
                if not target.is_relative_to(image_root.resolve()):
                    raise ValueError("Archive extraction escaped data directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = bundle.read(entry_info)
                if not target.exists() or target.read_bytes() != raw:
                    target.write_bytes(raw)
        archives.append({"name": archive.name, "source": entry["download_url"],
                         "sha256": digest(archive), "upstream_md5": entry["computed_md5"]})
        print(f"Verified {view}", flush=True)

    records = []
    for image_path in sorted(image_root.rglob("*")):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        match = re.fullmatch(r"(\d{3})_\d+", image_path.stem)
        if not match:
            raise ValueError(f"Unrecognized FRLL identity filename: {image_path.name}")
        records.append({"identity": f"frll-{match[1]}", "view": image_path.parent.name,
                        "path": image_path.relative_to(dataset_root).as_posix(),
                        "sha256": digest(image_path)})
    splits = identity_split([row["identity"] for row in records])
    for row in records:
        row["split"] = splits[row["identity"]]
    manifest = {"schema_version": 1, "dataset": "frll", "source": SOURCE,
                "license": "CC-BY-4.0", "attribution": "Lisa DeBruine and Benedict Jones, Face Research Lab London Set, version 5",
                "split_version": SPLIT_VERSION, "identity_splits": splits,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "archives": archives, "images": records}
    destination = dataset_root / "manifest.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("identity_splits") != splits:
            raise ValueError("Existing frozen identity split differs; refusing to overwrite")
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {destination} ({len(records)} images, {len(splits)} identities)", flush=True)
    return destination


def grouped_images(manifest_path: Path, split: str) -> dict[str, dict[str, Path]]:
    """Read paths from one named split; caller must explicitly request holdouts."""
    if split not in {"development", "calibration", "held_out"}:
        raise ValueError("Unknown split")
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for row in manifest["images"]:
        if row["split"] != split:
            continue
        candidate = (manifest_path.parent / row["path"]).resolve()
        if not candidate.is_relative_to(manifest_path.parent):
            raise ValueError("Dataset path escaped manifest directory")
        groups[row["identity"]][row["view"]] = candidate
    return dict(sorted(groups.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    options = parser.parse_args()
    fetch_frll(options.root, tuple(options.views))
