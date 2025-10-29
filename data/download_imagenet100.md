# ImageNet-100 Preparation Guide

ImageNet-100 is a community subset of ImageNet-1k containing 100 classes. Two popular approaches for obtaining the dataset are described below. Ensure that you respect the ImageNet license when downloading the original data.

## Option 1: Pre-built Community Subset

* A widely used list of ImageNet-100 class IDs is available [here](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt).
* If you already have access to the ImageNet-1k tar archives, extract only the classes listed in the file above into a new directory (e.g., `/data/imagenet100`).
* Maintain the standard ImageNet folder structure:
  * `train/<class_name>/*.JPEG`
  * `val/<class_name>/*.JPEG`

## Option 2: Programmatic Subsampling

1. Download the official ImageNet-1k dataset (requires credentials).
2. Use the provided class ID list to select a subset of 100 classes.
3. Run a script similar to the following to create the subset (replace paths accordingly):

```bash
python - <<'PY'
import shutil
from pathlib import Path

root = Path("/path/to/imagenet-1k")
out = Path("/data/imagenet100")
with open("imagenet100.txt") as f:
    classes = [line.strip() for line in f if line.strip()]
for split in ["train", "val"]:
    for cls in classes:
        src = root / split / cls
        dst = out / split / cls
        dst.mkdir(parents=True, exist_ok=True)
        for img in src.glob("*.JPEG"):
            shutil.copy(img, dst / img.name)
PY
```

4. Verify that there are 1300 training images and 50 validation images per class (matching ImageNet-1k statistics).

## Dataset Configuration

Update `configs/imagenet100_vitb.yaml` with the path to your subset. The datamodule assumes the standard ImageNet transforms (224×224 crops, normalization) and supports both self-supervised pretraining and downstream evaluation splits.
