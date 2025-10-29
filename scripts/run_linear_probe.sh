#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/linear_probe_cifar10.yaml}
python eval_linear.py --config "$CONFIG"
