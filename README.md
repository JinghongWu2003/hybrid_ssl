# Hybrid MAE + SimCLR Self-Supervised Learning

```
            +----------------+
            |   Encoder      |
            +--------+-------+
                     |
        +------------+------------+
        |                         |
  +-----v-----+             +-----v-----+
  | SimCLR    |             |   MAE      |
  | Projector |             |  Decoder   |
  +-----+-----+             +-----+-----+
        |                         |
   InfoNCE Loss             Reconstruction Loss
        \____________________  __________________/
                             \/
                       Dynamic Alpha
                             |
                      Downstream Tasks
                       (Linear Probe / FT)
```

Hybrid SSL is a production-ready PyTorch framework that jointly trains a masked autoencoder (MAE) decoder and a SimCLR-style contrastive projector on top of a shared encoder. The objective combines reconstruction and contrastive losses with a dynamic alpha schedule, enabling robust pretraining on mid-scale datasets like ImageNet-100 and smaller benchmarks such as Tiny-ImageNet or STL-10. Downstream evaluation utilities include linear probing and fine-tuning for CIFAR-10/100, Flowers102, Caltech-101, and Galaxy10-DECALS.

## Key Features

* Shared transformer or convolutional encoders with dual MAE and SimCLR heads.
* Dynamic alpha weighting that smoothly transitions from reconstruction-focused warmup to balanced contrastive learning.
* Comprehensive dataset datamodules with strong augmentations for self-supervision and evaluation.
* AMP-ready, single-GPU friendly training loops with optional distributed support and mixed precision.
* Modular utilities for seeding, logging (TensorBoard & WandB), checkpointing, and gradient balancing.
* Ready-to-run configs, scripts, tests, and a Colab notebook for reproducible experiments.

## Quickstart

1. Clone the repository and install dependencies:

```bash
pip install -e .
```

2. (Optional) Prepare Tiny-ImageNet locally:

```bash
python data/prepare_tiny_imagenet.py --out ./data/tiny-imagenet-200
```

3. Run joint pretraining on a single GPU (Tiny-ImageNet + ViT-Small):

```bash
python train.py --config configs/tiny_imagenet_vit_small.yaml
```

4. Launch the Google Colab workflow:

Open `scripts/run_colab.ipynb` and follow the guided cells. See the dedicated section below for a quick overview of the exact steps.

5. Evaluate with a linear probe (e.g., CIFAR-10):

```bash
python eval_linear.py --config configs/linear_probe_cifar10.yaml
```

6. Fine-tune on Flowers102:

```bash
python finetune.py --config configs/finetune_flowers102.yaml
```

## Repository Structure

```
hybrid_ssl/
  configs/                 # YAML configs for pretraining & downstream tasks
  data/                    # Dataset helpers and preparation scripts
  losses/                  # Reconstruction, contrastive, and scheduler utilities
  models/                  # Encoders and heads composing the hybrid model
  utils/                   # Logging, seeding, masking, metrics, checkpoint helpers
  scripts/                 # Command-line scripts and Colab notebook
  tests/                   # PyTest coverage for critical components
  train.py                 # Joint MAE + SimCLR pretraining entry point
  eval_linear.py           # Linear probing utility for frozen encoders
  finetune.py              # Fine-tuning utility for downstream tasks
  visualize.py             # Reconstructions, embedding projections, attention maps
```

## Dataset Notes

* **ImageNet-100:** Follow `data/download_imagenet100.md` to subsample from ImageNet-1k or use a pre-built community subset. Update `configs/imagenet100_vitb.yaml` with the correct root path.
* **Tiny-ImageNet-200:** Use `data/prepare_tiny_imagenet.py` to download and prepare the dataset automatically. The training script expects the ImageNet-style folder layout.
* **STL-10:** The datamodule leverages the unlabeled split for self-supervised pretraining and provides labeled splits for evaluation.
* **Flowers102 & Caltech-101:** Downloaded automatically via torchvision datasets.
* **Galaxy10-DECALS:** Loaded through the HuggingFace `datasets` library when available. The datamodule falls back to a custom HTTP download if `datasets` is not installed.

## Dynamic Alpha Scheduling & Ablations

The total loss is computed as

\[ L(t) = \alpha(t) \cdot L_{\text{rec}} + (1 - \alpha(t)) \cdot L_{\text{contrast}} \]

where `alpha(t)` follows a warmup-to-cosine schedule controlled by `alpha_warmup_epochs`, `alpha_final`, and `alpha_schedule`. Modify these parameters in your YAML config to perform ablations.

## Expected Resource Usage

* **Tiny-ImageNet + ViT-Small (batch=128):** ~13GB GPU memory, ~45 minutes per epoch on an RTX 5080 16GB.
* **ImageNet-100 + ViT-Base (batch=128):** ~15GB GPU memory, ~1.5 hours per epoch on the same GPU.
* Mixed precision (`amp: true`) is enabled by default in the provided configs to keep memory usage manageable.

## Reproducibility Tips

* Set deterministic behavior via `--deterministic` in `train.py` (enables deterministic cuDNN kernels at a small performance cost).
* Ensure all random seeds are set using the `utils/seed.py` helper.
* Save alpha schedule plots and TensorBoard logs using the built-in logging utilities.

## Switching to ImageNet-100

Update the dataset section of your config to point at the ImageNet-100 root directory and adjust batch size or learning rate as necessary:

```yaml
dataset:
  name: "imagenet100"
  root: "/data/imagenet100"
  img_size: 224
```

Then launch pretraining with:

```bash
python train.py --config configs/imagenet100_vitb.yaml
```

## Ablations & Extensions

* Modify `mask_ratio` in the config to control the percentage of masked patches.
* Change `alpha_final` and `alpha_schedule` to explore different loss trade-offs.
* Swap encoders between ViT and ResNet variants by updating `model.encoder`.
* Enable optional gradient balancing with `--grad_balance` for multi-task stability.

## Logging & Visualization

The training loop logs losses, learning rate, alpha(t), and reconstruction samples to TensorBoard. Optional Weights & Biases logging can be enabled via `--wandb` or config flags. Visualization utilities save reconstructions, t-SNE/UMAP projections, and attention maps to `runs/<experiment>/figs/`.

## Running on Google Colab

1. **Create a new notebook** and switch the runtime to **GPU** (`Runtime -> Change runtime type -> GPU`).
2. **Clone the repository** (or pull from your own fork) and install dependencies:

   ```python
   !git clone https://github.com/<your-username>/hybrid_ssl.git
   %cd hybrid_ssl
   !pip install -U pip
   !pip install -e .
   ```

   If you prefer mounting Google Drive, insert the standard Drive mounting cell (`from google.colab import drive; drive.mount('/content/drive')`) before cloning so checkpoints can be saved persistently.

3. **Download Tiny-ImageNet** (default Colab example) using the provided helper:

   ```python
   !python data/prepare_tiny_imagenet.py --out ./data/tiny-imagenet-200
   ```

4. **Run joint pretraining** with the Tiny-ImageNet configuration:

   ```python
   !python train.py --config configs/tiny_imagenet_vit_small.yaml --logging.out_dir runs/colab_tiny
   ```

   The config uses mixed precision (`amp: true`) which fits within the default Colab GPU memory budget. Reduce `train.batch_size` in the YAML if you encounter OOM errors.

5. **Launch the linear probe** once pretraining finishes:

   ```python
   !python eval_linear.py --config configs/linear_probe_cifar10.yaml --checkpoint.encoder_ckpt checkpoints/tiny_vits_last.pt
   ```

   Adjust the checkpoint path to the artifact saved during your pretraining run (the notebook highlights the exact filename).

6. **Use the visualization utility** to export reconstructions and embedding plots:

   ```python
   !python visualize.py --config configs/tiny_imagenet_vit_small.yaml --checkpoint checkpoints/tiny_vits_last.pt --output runs/colab_tiny/figs
   ```

7. For a fully scripted experience, simply run the cells in `scripts/run_colab.ipynb`; they mirror the commands above and add optional extras such as TensorBoard-in-Colab integration.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
