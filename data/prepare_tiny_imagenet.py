"""Utility script to download and extract Tiny-ImageNet-200."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


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
    print("Tiny-ImageNet ready at:", archive_dir / "tiny-imagenet-200")


if __name__ == "__main__":
    main()
