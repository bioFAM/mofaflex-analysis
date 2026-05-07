from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import shutil

import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.sln208_prior_noise_refinement import (
        DEFAULT_DATA_PATH,
        run,
    )
    from mofaflex_sensitivity.sln208_prior_noise_refinement_plotting import (
        aggregate_with_errorbars,
        save_metric_grid_plot,
    )
else:
    from .sln208_prior_noise_refinement import (
        DEFAULT_DATA_PATH,
        run,
    )
    from .sln208_prior_noise_refinement_plotting import (
        aggregate_with_errorbars,
        save_metric_grid_plot,
    )


DEFAULT_OUT_DIR = Path("artifacts/sln208_prior_noise_refinement_sweep")
DEFAULT_NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
SUMMARY_PANELS = [
    ("mean_average_precision_true", "Mean AP vs true prior", None),
    ("mean_average_precision_noisy", "Mean AP vs noisy prior", None),
    ("fraction_pathways_ap_true_gt_noisy", "Fraction pathways improved", None),
    ("mean_top_true_size_jaccard_true", "Mean top-k Jaccard vs true", None),
]
SOURCE_SUMMARY_COLUMNS = [
    "average_precision_true",
    "average_precision_noisy",
    "average_precision_gain_true_minus_noisy",
    "best_f1_true",
    "best_f1_noisy",
    "best_f1_gain_true_minus_noisy",
    "top_true_size_jaccard_true",
    "top_true_size_jaccard_noisy",
]
FOCUS_SOURCE_PANELS = [
    ("average_precision_true", "mean AP vs true prior", None),
    ("average_precision_noisy", "mean AP vs noisy prior", None),
    ("average_precision_gain_true_minus_noisy", "mean AP gain", None),
    ("top_true_size_jaccard_true", "mean top-k Jaccard vs true", None),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep paired false-positive/false-negative prior noise levels on SLN208, "
            "train MOFA-FLEX, and summarize refinement robustness."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of training seeds.")
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Optional shared noise-mask seed. If omitted, each run uses its training seed.",
    )
    parser.add_argument("--n-uninformed-factors", type=int, default=3)
    parser.add_argument("--annotation-confidence", type=float, default=0.99)
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--early-stopper-patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--n-particles", type=int, default=1)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["rna", "prot"],
        default=["rna", "prot"],
        help="Modalities to include when fitting MOFA-FLEX.",
    )
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--focus-source",
        type=str,
        choices=["hallmark", "mca"],
        default=None,
        help="Optionally write a source-specific summary table and plot.",
    )
    return parser


def _slugify_noise_level(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def save_summary_plot(summary_df: pd.DataFrame, *, path: Path) -> None:
    save_metric_grid_plot(
        summary_df,
        x_col="noise_level",
        panels=SUMMARY_PANELS,
        path=path,
        title="SLN208 prior-noise refinement robustness",
    )


def save_summary_errorbar_plot(summary_per_run_df: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    value_cols = [column for column, _, _ in SUMMARY_PANELS]
    aggregated = aggregate_with_errorbars(summary_per_run_df, group_col="noise_level", value_cols=value_cols)
    save_metric_grid_plot(
        aggregated,
        x_col="noise_level",
        panels=SUMMARY_PANELS,
        path=path,
        title="SLN208 prior-noise refinement robustness across seeds (mean ± SD)",
        errorbars=True,
    )
    return aggregated


def _collect_refinement_paths(out_dir: Path, *, noise_level: float | None = None) -> list[Path]:
    if noise_level is None:
        return sorted(out_dir.glob("noise_*/seed_*/pathway_refinement.csv"))
    noise_dir = out_dir / f"noise_{_slugify_noise_level(float(noise_level))}"
    refinement_paths = sorted(noise_dir.glob("seed_*/pathway_refinement.csv"))
    if not refinement_paths and (noise_dir / "pathway_refinement.csv").exists():
        refinement_paths = [noise_dir / "pathway_refinement.csv"]
    return refinement_paths


def _load_existing_run_summary(run_dir: Path) -> dict[str, object] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _source_summary_from_refinement_paths(refinement_paths: list[Path], *, include_seed: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for refinement_path in refinement_paths:
        refinement_df = pd.read_csv(refinement_path)
        source_summary = refinement_df.groupby("source")[SOURCE_SUMMARY_COLUMNS].mean().reset_index()
        if include_seed:
            noise_label = refinement_path.parent.parent.name.replace("noise_", "").replace("p", ".")
            seed_label = refinement_path.parent.name.replace("seed_", "")
            source_summary.insert(0, "seed", int(seed_label))
            source_summary.insert(0, "noise_level", float(noise_label))
        rows.append(source_summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_by_source(out_dir: Path, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for noise_level in summary_df["noise_level"].tolist():
        source_summary = _source_summary_from_refinement_paths(_collect_refinement_paths(out_dir, noise_level=float(noise_level)))
        if source_summary.empty:
            continue
        source_summary.insert(0, "noise_level", float(noise_level))
        rows.append(source_summary)

    source_df = pd.concat(rows, ignore_index=True)
    overall_df = summary_df[
        [
            "noise_level",
            "mean_average_precision_true",
            "mean_average_precision_noisy",
            "mean_ap_gain_true_minus_noisy",
            "mean_best_f1_gain_true_minus_noisy",
            "mean_top_true_size_jaccard_true",
            "mean_top_true_size_jaccard_noisy",
        ]
    ].rename(
        columns={
            "mean_average_precision_true": "average_precision_true",
            "mean_average_precision_noisy": "average_precision_noisy",
            "mean_ap_gain_true_minus_noisy": "average_precision_gain_true_minus_noisy",
            "mean_best_f1_gain_true_minus_noisy": "best_f1_gain_true_minus_noisy",
            "mean_top_true_size_jaccard_true": "top_true_size_jaccard_true",
            "mean_top_true_size_jaccard_noisy": "top_true_size_jaccard_noisy",
        }
    )
    overall_df["best_f1_true"] = pd.NA
    overall_df["best_f1_noisy"] = pd.NA
    overall_df.insert(1, "source", "overall")
    source_df = pd.concat([overall_df, source_df], ignore_index=True)
    source_df = source_df.sort_values(["source", "noise_level"]).reset_index(drop=True)
    source_df.to_csv(out_dir / "summary_overall_and_by_source.csv", index=False)
    return source_df


def summarize_by_source_per_run(out_dir: Path) -> pd.DataFrame:
    source_per_run_df = _source_summary_from_refinement_paths(_collect_refinement_paths(out_dir), include_seed=True)
    if source_per_run_df.empty:
        return pd.DataFrame()
    source_per_run_df = source_per_run_df.sort_values(["source", "noise_level", "seed"]).reset_index(drop=True)
    source_per_run_df.to_csv(out_dir / "summary_overall_and_by_source_per_run.csv", index=False)
    return source_per_run_df


def save_focus_source_outputs(source_df: pd.DataFrame, *, focus_source: str, out_dir: Path, plots_dir: Path) -> pd.DataFrame:
    focus_df = source_df[source_df["source"] == focus_source].copy().sort_values("noise_level").reset_index(drop=True)
    focus_df.to_csv(out_dir / f"summary_{focus_source}_only.csv", index=False)
    if focus_df.empty:
        return focus_df
    save_metric_grid_plot(
        focus_df,
        x_col="noise_level",
        panels=[(column, f"{focus_source.upper()} {label}", color) for column, label, color in FOCUS_SOURCE_PANELS],
        path=plots_dir / f"summary_{focus_source}_only.png",
        title=f"SLN208 prior-noise refinement robustness ({focus_source.upper()} only)",
    )
    return focus_df


def save_focus_source_errorbar_outputs(
    source_per_run_df: pd.DataFrame,
    *,
    focus_source: str,
    out_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    focus_per_run_df = (
        source_per_run_df[source_per_run_df["source"] == focus_source]
        .copy()
        .sort_values(["noise_level", "seed"])
        .reset_index(drop=True)
    )
    focus_per_run_df.to_csv(out_dir / f"summary_{focus_source}_only_per_run.csv", index=False)
    if focus_per_run_df.empty:
        return focus_per_run_df

    value_cols = [column for column, _, _ in FOCUS_SOURCE_PANELS]
    aggregated = aggregate_with_errorbars(focus_per_run_df, group_col="noise_level", value_cols=value_cols)
    aggregated.to_csv(out_dir / f"summary_{focus_source}_only_with_errorbars.csv", index=False)
    save_metric_grid_plot(
        aggregated,
        x_col="noise_level",
        panels=[(column, f"{focus_source.upper()} {label}", color) for column, label, color in FOCUS_SOURCE_PANELS],
        path=plots_dir / f"summary_{focus_source}_only_errorbars.png",
        title=f"SLN208 prior-noise refinement robustness across seeds ({focus_source.upper()} only, mean ± SD)",
        errorbars=True,
    )
    return aggregated


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    flat_models_dir = out_dir / "models"
    flat_models_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(args.seed)] if args.seeds is None else [int(seed) for seed in args.seeds]

    summaries: list[dict[str, object]] = []
    for noise_level in args.noise_levels:
        noise_dir = out_dir / f"noise_{_slugify_noise_level(noise_level)}"
        for seed in seeds:
            run_dir = noise_dir / f"seed_{seed:03d}"
            existing = _load_existing_run_summary(run_dir)
            if existing is not None:
                existing["noise_level"] = float(noise_level)
                existing["seed"] = int(seed)
                summaries.append(existing)
                print(f"Skipping existing run: noise={float(noise_level):.3f}, seed={seed}")
                continue
            run_args = argparse.Namespace(
                data_path=args.data_path,
                out_dir=run_dir,
                fpr=float(noise_level),
                fnr=float(noise_level),
                seed=seed,
                noise_seed=seed if args.noise_seed is None else args.noise_seed,
                n_uninformed_factors=args.n_uninformed_factors,
                annotation_confidence=args.annotation_confidence,
                max_epochs=args.max_epochs,
                early_stopper_patience=args.early_stopper_patience,
                lr=args.lr,
                n_particles=args.n_particles,
                modalities=args.modalities,
                save_model=args.save_model,
            )
            result = run(run_args)
            result["noise_level"] = float(noise_level)
            result["seed"] = int(seed)
            summaries.append(result)

            if args.save_model:
                nested_models_dir = run_args.out_dir / "models"
                for model_path in nested_models_dir.glob("*.h5"):
                    flat_path = flat_models_dir / model_path.name
                    if flat_path.exists() or flat_path.is_symlink():
                        flat_path.unlink()
                    shutil.copy2(model_path, flat_path)

    summary_per_run_df = pd.DataFrame.from_records(summaries).sort_values(["noise_level", "seed"]).reset_index(drop=True)
    summary_per_run_df.to_csv(out_dir / "summary_per_run.csv", index=False)
    summary_df = (
        summary_per_run_df.groupby("noise_level", as_index=False)
        .mean(numeric_only=True)
        .sort_values("noise_level")
        .reset_index(drop=True)
    )
    summary_df.to_csv(out_dir / "summary_across_noise_levels.csv", index=False)
    save_summary_plot(summary_df, path=plots_dir / "summary_across_noise_levels.png")
    source_df = summarize_by_source(out_dir, summary_df)
    if args.focus_source is not None:
        save_focus_source_outputs(source_df, focus_source=args.focus_source, out_dir=out_dir, plots_dir=plots_dir)
    if len(seeds) > 1:
        summary_with_errorbars_df = save_summary_errorbar_plot(
            summary_per_run_df,
            path=plots_dir / "summary_across_noise_levels_errorbars.png",
        )
        summary_with_errorbars_df.to_csv(out_dir / "summary_across_noise_levels_with_errorbars.csv", index=False)
        source_per_run_df = summarize_by_source_per_run(out_dir)
        if args.focus_source is not None:
            save_focus_source_errorbar_outputs(
                source_per_run_df,
                focus_source=args.focus_source,
                out_dir=out_dir,
                plots_dir=plots_dir,
            )

    best_noise_level = summary_df.loc[summary_df["mean_average_precision_true"].idxmax(), "noise_level"]
    payload = {
        "data_path": str(args.data_path),
        "out_dir": str(out_dir),
        "noise_levels": [float(x) for x in args.noise_levels],
        "seed": int(args.seed),
        "seeds": seeds,
        "noise_seed": (None if args.noise_seed is None else int(args.noise_seed)),
        "n_uninformed_factors": int(args.n_uninformed_factors),
        "annotation_confidence": float(args.annotation_confidence),
        "max_epochs": int(args.max_epochs),
        "early_stopper_patience": int(args.early_stopper_patience),
        "lr": float(args.lr),
        "n_particles": int(args.n_particles),
        "modalities": [str(modality) for modality in args.modalities],
        "focus_source": args.focus_source,
        "n_runs": int(len(summary_per_run_df)),
        "best_noise_level_by_mean_ap_true": float(best_noise_level),
    }
    (out_dir / "resolved_run.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return summary_df


def main() -> None:
    args = build_parser().parse_args()
    summary_df = run_sweep(args)
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
