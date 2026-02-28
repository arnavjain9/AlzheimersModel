"""
experiments/run_experiment.py — Main experiment runner script.

Loads a YAML config from experiments/configs/, instantiates a DiseaseState,
runs the damage context across each epoch, evaluates the model, and writes
longitudinal results to disk.

Usage::

    python experiments/run_experiment.py --config configs/braak_iii_iv.yaml

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Alzheimer's simulation experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Directory to write evaluation results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Experiment runner not yet implemented. "
        f"Config: {args.config}, Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
