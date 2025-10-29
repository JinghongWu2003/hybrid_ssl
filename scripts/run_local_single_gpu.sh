#!/usr/bin/env bash
set -euo pipefail

CONFIG_PRETRAIN=${1:-configs/tiny_imagenet_vit_small.yaml}
CONFIG_LP=${2:-configs/linear_probe_cifar10.yaml}

python train.py --config "$CONFIG_PRETRAIN"
python eval_linear.py --config "$CONFIG_LP"
