"""Utility script to download and extract Tiny-ImageNet-200."""
from __future__ import annotations

import argparse
import tarfile
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

try:  # Optional dependency used for a lightweight sanity check.
    from torchvision.datasets import ImageFolder
except ImportError:  # pragma: no cover - torchvision might be unavailable.
    ImageFolder = None  # type: ignore


URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Archive already exists at {out_path}, skipping download.")
        return out_path
    print(f"Downloading Tiny-ImageNet from {url}...")
    urlretrieve(url, out_path)
    print("Download complete.")
    return out_path


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} to {out_dir}...")
    if archive.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r") as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive.suffix}")
    print("Extraction complete.")


def reorganize_validation_split(dataset_root: Path) -> None:
    """Move validation images into class subdirectories and report counts."""

    val_root = dataset_root / "val"
    annotations_path = val_root / "val_annotations.txt"
    images_dir = val_root / "images"

    if not annotations_path.exists():
        print("Validation annotations file not found; skipping re-organization.")
        return

    if images_dir.exists():
        print("Reorganizing validation images according to annotations...")
        with annotations_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, class_id = parts[0], parts[1]
                class_dir = val_root / class_id
                class_dir.mkdir(exist_ok=True)

                src = images_dir / filename
                dst = class_dir / filename
                if dst.exists():
                    continue
                if src.exists():
                    src.replace(dst)

        # Remove the now-empty images directory if possible to avoid future confusion.
        try:
            images_dir.rmdir()
        except OSError:
            pass
    else:
        print("Validation images directory already organized; skipping moves.")

    counts = Counter()
    for class_dir in sorted(val_root.iterdir()):
        if class_dir.is_dir() and class_dir.name != "images":
            counts[class_dir.name] = len(list(class_dir.glob("*.JPEG")))

    if counts:
        print("Validation samples per class:")
        for class_id in sorted(counts):
            print(f"  {class_id}: {counts[class_id]}")

    if ImageFolder is not None:
        dataset = ImageFolder(str(val_root))
        assert len(dataset.classes) == 200, f"Expected 200 classes, got {len(dataset.classes)}"
        print("ImageFolder sanity check passed (200 classes detected).")
    else:
        print("torchvision not available; skipping ImageFolder sanity check.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Tiny-ImageNet-200 dataset")
    parser.add_argument("--out", type=Path, default=Path("./data"), help="Output directory")
    parser.add_argument(
        "--url",
        type=str,
        default=URL,
        help="Override dataset URL if hosting a local mirror.",
    )
    args = parser.parse_args()

    archive_dir = args.out
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / Path(args.url).name
    download(args.url, archive_path)
    extract(archive_path, archive_dir)

    dataset_root = archive_dir / "tiny-imagenet-200"
    reorganize_validation_split(dataset_root)

    print("Tiny-ImageNet ready at:", dataset_root)


if __name__ == "__main__":
    main()
