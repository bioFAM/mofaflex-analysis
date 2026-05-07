from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate MOFA-FLEX on synthetic data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_baseline.toml"),
        help="Path to the TOML config file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    metrics = run_pipeline(config)
    print(f"Finished synthetic MOFA-FLEX run in {config.experiment.output_dir}")
    print(f"reconstruction_rmse={metrics['reconstruction_rmse']:.6f}")
    print(f"mean_matched_abs_weight_correlation={metrics['mean_matched_abs_weight_correlation']:.6f}")


if __name__ == "__main__":
    main()
