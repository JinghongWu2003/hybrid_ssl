#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/finetune_flowers102.yaml}
python finetune.py --config "$CONFIG"
