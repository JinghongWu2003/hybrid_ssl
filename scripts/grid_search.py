"""Utility for running grid searches over Hybrid SSL hyperparameters.

This script automates launching multiple `train.py` runs with different
hyper-parameter combinations.  It is designed to cover the most common knobs
we tweak when experimenting with the Hybrid MAE + SimCLR objective:

* ``model.mask_ratio`` — how much of the image we mask for the MAE branch.
* ``loss.alpha_final`` — target weight for the reconstruction loss.
* ``model.temp`` — the InfoNCE temperature for the contrastive head.
* ``optim.lr`` — base learning rate for AdamW.

The defaults focus on the STL-10 + ResNet-18 configuration and keep the search
manageable on a single GPU.  You can extend/override the search space from the
command line.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml


DEFAULT_GRID: Mapping[str, Sequence[object]] = {
    "model.mask_ratio": [0.5, 0.7],
    "loss.alpha_final": [0.4, 0.55, 0.7],
    "model.temp": [0.1, 0.2],
    "optim.lr": [5e-4, 1e-3],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple train.py runs for a Cartesian grid of hyper-parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Base YAML configuration to copy before applying overrides.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grid_runs"),
        help="Directory where per-run configs, logs, and results.csv will be stored.",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=None,
        metavar="KEY=v1,v2,...",
        help=(
            "Override or extend the default grid. Use dotted keys (e.g. 'model.mask_ratio') "
            "and comma-separated values. You can repeat this flag multiple times."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optionally override the number of epochs for every run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N combinations (useful for smoke tests).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Forward the --wandb flag to train.py for every run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands without executing them.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use when invoking train.py.",
    )
    return parser.parse_args(argv)


def parse_scalar(raw: str) -> object:
    raw = raw.strip()
    if not raw:
        raise ValueError("Grid values cannot be empty strings.")
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, NameError):
        # Treat bare words as strings, e.g. cosine schedule names.
        return raw


def parse_grid(overrides: Iterable[str] | None) -> Dict[str, List[object]]:
    if not overrides:
        return {k: list(v) for k, v in DEFAULT_GRID.items()}
    grid: Dict[str, List[object]] = {k: list(v) for k, v in DEFAULT_GRID.items()}
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Invalid grid specification '{spec}'. Expected KEY=v1,v2,... format.")
        key, values = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid grid specification '{spec}'. Key cannot be empty.")
        value_list = [parse_scalar(v) for v in values.split(",")]
        if not value_list:
            raise ValueError(f"Invalid grid specification '{spec}'. Must provide at least one value.")
        grid[key] = value_list
    return grid


def set_by_path(data: MutableMapping[str, object], dotted_key: str, value: object) -> None:
    keys = dotted_key.split(".")
    cursor: MutableMapping[str, object] = data
    for key in keys[:-1]:
        node = cursor.get(key)
        if not isinstance(node, MutableMapping):
            node = {}
            cursor[key] = node
        cursor = node  # type: ignore[assignment]
    cursor[keys[-1]] = value


def iter_grid(grid: Mapping[str, Sequence[object]]) -> Iterable[Dict[str, object]]:
    if not grid:
        return []
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def ensure_subdirs(base_dir: Path) -> Dict[str, Path]:
    configs_dir = base_dir / "configs"
    logs_dir = base_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {"configs": configs_dir, "logs": logs_dir}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    grid = parse_grid(args.grid)
    combinations = list(iter_grid(grid))
    if args.limit is not None:
        combinations = combinations[: args.limit]
    if not combinations:
        print("No combinations generated. Nothing to do.")
        return 0

    base_config = yaml.safe_load(args.base_config.read_text())
    base_name = args.base_config.stem

    output_dir = args.output_dir.resolve()
    dirs = ensure_subdirs(output_dir)
    results_path = output_dir / "results.csv"
    new_results_file = not results_path.exists()

    with results_path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["run_name", "status", "return_code", "duration_sec"] + list(grid.keys())
        if new_results_file:
            writer.writerow(header)

        for index, combo in enumerate(combinations, start=1):
            run_name = f"{base_name}_grid_{index:03d}"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            config_copy = copy.deepcopy(base_config)
            config_copy["experiment"] = run_name
            logging_cfg = config_copy.setdefault("logging", {})
            logging_cfg["out_dir"] = str(run_dir / "logs")
            if args.epochs is not None:
                train_cfg = config_copy.setdefault("train", {})
                train_cfg["epochs"] = args.epochs

            for key, value in combo.items():
                set_by_path(config_copy, key, value)

            config_path = dirs["configs"] / f"{run_name}.yaml"
            config_path.write_text(yaml.safe_dump(config_copy, sort_keys=False))

            cmd = [args.python, "train.py", "--config", str(config_path)]
            if args.wandb:
                cmd.append("--wandb")

            print(f"\n[{index}/{len(combinations)}] Launching: {' '.join(cmd)}")
            for key, value in combo.items():
                print(f"  - {key} = {value}")

            if args.dry_run:
                status = "dry-run"
                return_code = 0
                duration = 0.0
            else:
                log_path = dirs["logs"] / f"{run_name}.log"
                start_time = time.time()
                with log_path.open("w") as log_file:
                    result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                duration = time.time() - start_time
                return_code = result.returncode
                status = "ok" if return_code == 0 else "failed"
                print(f"    -> Finished in {duration / 60:.1f} min with status {status} (code {return_code}).")

            writer.writerow(
                [run_name, status, return_code, f"{duration:.2f}"]
                + [combo[key] for key in grid.keys()]
            )
            csv_file.flush()

    print(f"\nGrid search finished. Results saved to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
