from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.uninformed_sensitivity import (
        compute_processed_subset_r2,
        factor_subset_names,
        factor_total_r2_table,
        fit_mofaflex_model,
        match_factor_weights,
    )
    from mofaflex_sensitivity.plot_style import (
        CAT_PALETTE,
        DIVERGING_CMAP,
        clean_ax,
        place_legend,
        savefig,
        set_house_style,
    )
else:
    from .plot_style import CAT_PALETTE, DIVERGING_CMAP, clean_ax, place_legend, savefig, set_house_style
    from .uninformed_sensitivity import (
        compute_processed_subset_r2,
        factor_subset_names,
        factor_total_r2_table,
        fit_mofaflex_model,
        match_factor_weights,
    )


DEFAULT_OUT_DIR = Path("artifacts/synthetic_uninformed_dense_benchmark")


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    n_samples: int = 10_000
    n_features: int = 5_000
    n_true_sparse_factors: int = 50
    n_informed_true_priors: int = 40
    n_random_priors: int = 5
    true_dense_grid: tuple[int, ...] = (1, 2, 3, 5, 10)
    fitted_uninformed_grid: tuple[int, ...] = (1, 2, 3, 5, 10)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    prior_noise_fraction: float = 0.5
    likelihood: str = "Normal"
    nmf: bool = True
    factor_size_dist: str = "Uniform"
    annotation_confidence: float = 0.97
    weight_prior: str = "Horseshoe"
    factor_prior: str = "Normal"
    nonnegative_weights: bool = True
    nonnegative_factors: bool = True
    init_factors: float = 0.0
    init_scale: float = 0.1
    lr: float = 0.003
    batch_size: int = 0
    max_epochs: int = 10_000
    early_stopper_patience: int = 100
    n_particles: int = 1
    scale_per_group: bool = True
    disable_normal_scale_data: bool = False
    save_model: bool = True
    save_data: bool = True
    skip_existing: bool = True
    heatmap_seed: int = 0


@dataclass
class SyntheticTruth:
    mdata: Any
    true_weights: pd.DataFrame
    true_mask: pd.DataFrame
    prior_mask: pd.DataFrame
    prior_stats: pd.DataFrame
    true_factors: list[str]
    true_informed_sparse_factors: list[str]
    true_hidden_sparse_factors: list[str]
    true_dense_factors: list[str]
    random_prior_names: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic MOFA-FLEX benchmark for fitted uninformed-factor sensitivity with noisy true priors "
            "and random fake priors."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--n-features", type=int, default=5_000)
    parser.add_argument("--n-true-sparse-factors", type=int, default=50)
    parser.add_argument("--n-informed-true-priors", type=int, default=40)
    parser.add_argument("--n-random-priors", type=int, default=5)
    parser.add_argument("--true-dense-grid", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--fitted-uninformed-grid", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--prior-noise-fraction", type=float, default=0.5)
    parser.add_argument("--annotation-confidence", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=10_000)
    parser.add_argument("--early-stopper-patience", type=int, default=100)
    parser.add_argument("--n-particles", type=int, default=1)
    parser.add_argument("--scale-per-group", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-normal-scale-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heatmap-seed", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> SyntheticBenchmarkConfig:
    return SyntheticBenchmarkConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_true_sparse_factors=args.n_true_sparse_factors,
        n_informed_true_priors=args.n_informed_true_priors,
        n_random_priors=args.n_random_priors,
        true_dense_grid=tuple(args.true_dense_grid),
        fitted_uninformed_grid=tuple(args.fitted_uninformed_grid),
        seeds=tuple(args.seeds),
        prior_noise_fraction=args.prior_noise_fraction,
        annotation_confidence=args.annotation_confidence,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopper_patience=args.early_stopper_patience,
        n_particles=args.n_particles,
        scale_per_group=args.scale_per_group,
        disable_normal_scale_data=args.disable_normal_scale_data,
        save_model=args.save_model,
        save_data=args.save_data,
        skip_existing=args.skip_existing,
        heatmap_seed=args.heatmap_seed,
    )


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def dense_factor_indices(config: SyntheticBenchmarkConfig, true_dense: int) -> np.ndarray:
    start = config.n_true_sparse_factors
    return np.arange(start, start + true_dense, dtype=int)


def true_factor_names(config: SyntheticBenchmarkConfig, true_dense: int) -> list[str]:
    sparse_names = [f"true_sparse_{idx:02d}" for idx in range(config.n_true_sparse_factors)]
    dense_names = [f"true_dense_{idx:02d}" for idx in range(true_dense)]
    return [*sparse_names, *dense_names]


def generate_synthetic_truth(
    *,
    config: SyntheticBenchmarkConfig,
    true_dense: int,
    seed: int,
) -> SyntheticTruth:
    import mofaflex as mfl

    rng = np.random.default_rng(seed)
    total_true_factors = config.n_true_sparse_factors + true_dense
    dg = mfl.tl.DataGenerator(
        n_features=[config.n_features],
        n_samples=config.n_samples,
        likelihoods=[config.likelihood],
        n_fully_shared_factors=total_true_factors,
        n_partially_shared_factors=0,
        n_private_factors=0,
        factor_size_dist=config.factor_size_dist,
        n_active_factors=1.0,
        nmf=[config.nmf],
    )
    dg.generate(rng=rng, all_combs=False)
    _densify_true_factors(dg, true_dense=true_dense, config=config, rng=rng)

    mdata = dg.to_mudata(noisy=False)
    view_name = next(iter(mdata.mod.keys()))
    feature_names = pd.Index(mdata[view_name].var_names)
    factor_names = true_factor_names(config, true_dense)

    true_weights = pd.DataFrame(dg._ws[0], index=factor_names, columns=feature_names)
    true_mask = pd.DataFrame(dg._w_masks[0], index=factor_names, columns=feature_names)
    prior_mask, prior_stats, random_prior_names = build_prior_mask(
        true_mask=true_mask,
        config=config,
        rng=rng,
    )
    mdata[view_name].varm["w_mask"] = prior_mask.T.copy()
    mdata[view_name].varm["true_w"] = true_weights.T.copy()
    mdata[view_name].varm["true_w_mask"] = true_mask.T.copy()
    mdata.uns["synthetic_uninformed_dense_benchmark"] = {
        "true_dense_factors": int(true_dense),
        "seed": int(seed),
        "n_true_sparse_factors": int(config.n_true_sparse_factors),
        "n_informed_true_priors": int(config.n_informed_true_priors),
        "n_random_priors": int(config.n_random_priors),
        "prior_noise_fraction": float(config.prior_noise_fraction),
    }

    return SyntheticTruth(
        mdata=mdata,
        true_weights=true_weights,
        true_mask=true_mask,
        prior_mask=prior_mask,
        prior_stats=prior_stats,
        true_factors=factor_names,
        true_informed_sparse_factors=factor_names[: config.n_informed_true_priors],
        true_hidden_sparse_factors=factor_names[config.n_informed_true_priors : config.n_true_sparse_factors],
        true_dense_factors=factor_names[config.n_true_sparse_factors :],
        random_prior_names=random_prior_names,
    )


def _densify_true_factors(dg, *, true_dense: int, config: SyntheticBenchmarkConfig, rng: np.random.Generator) -> None:
    if true_dense <= 0:
        return

    dense_idx = dense_factor_indices(config, true_dense)
    w = np.array(dg._ws[0], copy=True)
    w_mask = np.array(dg._w_masks[0], copy=True)
    w[dense_idx, :] = rng.standard_normal((true_dense, config.n_features))
    if config.nmf:
        w[dense_idx, :] = np.abs(w[dense_idx, :])
    w_mask[dense_idx, :] = True

    y_loc = dg._z @ w
    sigma = np.asarray(dg._sigmas[0]).reshape(1, -1)
    y = rng.normal(loc=y_loc, scale=sigma)
    if config.nmf:
        y = np.abs(y)

    dg._ws[0] = w.astype(np.float32, copy=False)
    dg._w_masks[0] = w_mask
    dg._ys[0] = y.astype(np.float32, copy=False)


def build_prior_mask(
    *,
    true_mask: pd.DataFrame,
    config: SyntheticBenchmarkConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    true_prior_names = list(true_mask.index[: config.n_informed_true_priors])
    noisy_true_priors = []
    stats_rows: list[dict[str, Any]] = []

    for factor_name in true_prior_names:
        original = true_mask.loc[factor_name].to_numpy(dtype=bool)
        noisy = randomly_replace_mask_members(original, fraction=config.prior_noise_fraction, rng=rng)
        noisy_true_priors.append(pd.Series(noisy, index=true_mask.columns, name=factor_name))
        stats_rows.append(
            {
                "prior_name": factor_name,
                "source": "noisy_true",
                "true_factor": factor_name,
                "n_true_members": int(original.sum()),
                "n_prior_members": int(noisy.sum()),
                "jaccard_with_true": jaccard_bool(original, noisy),
            }
        )

    true_sizes = np.asarray([true_mask.loc[name].sum() for name in true_prior_names], dtype=int)
    random_prior_rows = []
    random_prior_names = []
    for idx in range(config.n_random_priors):
        size = int(rng.choice(true_sizes))
        size = min(max(size, 1), true_mask.shape[1])
        prior = np.zeros(true_mask.shape[1], dtype=bool)
        prior[rng.choice(true_mask.shape[1], size=size, replace=False)] = True
        prior_name = f"random_prior_{idx:02d}"
        random_prior_names.append(prior_name)
        random_prior_rows.append(pd.Series(prior, index=true_mask.columns, name=prior_name))
        stats_rows.append(
            {
                "prior_name": prior_name,
                "source": "random",
                "true_factor": None,
                "n_true_members": 0,
                "n_prior_members": int(prior.sum()),
                "jaccard_with_true": np.nan,
            }
        )

    prior_mask = pd.DataFrame([*noisy_true_priors, *random_prior_rows])
    prior_stats = pd.DataFrame.from_records(stats_rows)
    return prior_mask.astype(bool), prior_stats, random_prior_names


def randomly_replace_mask_members(mask: np.ndarray, *, fraction: float, rng: np.random.Generator) -> np.ndarray:
    noisy = np.array(mask, copy=True, dtype=bool)
    active = np.flatnonzero(noisy)
    inactive = np.flatnonzero(~noisy)
    n_swap = min(int(fraction * len(active)), len(active), len(inactive))
    if n_swap == 0:
        return noisy
    drop = rng.choice(active, size=n_swap, replace=False)
    add = rng.choice(inactive, size=n_swap, replace=False)
    noisy[drop] = False
    noisy[add] = True
    return noisy


def jaccard_bool(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return np.nan
    return float(np.logical_and(left, right).sum() / union)


def fit_benchmark_model(
    *,
    truth: SyntheticTruth,
    config: SyntheticBenchmarkConfig,
    fitted_uninformed: int,
    seed: int,
    model_path: Path | bool,
):
    import mofaflex as mfl
    import mofaflex._core.likelihoods.normal as normal_likelihood_module

    view_name = next(iter(truth.mdata.mod.keys()))
    data_opts = mfl.DataOptions(
        group_by=None,
        scale_per_group=config.scale_per_group and not config.disable_normal_scale_data,
        annotations_varm_key={view_name: "w_mask"},
        plot_data_overview=False,
    )
    model_opts = mfl.ModelOptions(
        n_factors=fitted_uninformed,
        weight_prior=config.weight_prior,
        factor_prior=config.factor_prior,
        likelihoods={view_name: config.likelihood},
        nonnegative_weights=config.nonnegative_weights,
        nonnegative_factors=config.nonnegative_factors,
        annotation_confidence=config.annotation_confidence,
        init_factors=config.init_factors,
        init_scale=config.init_scale,
    )
    training_opts = mfl.TrainingOptions(
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        early_stopper_patience=config.early_stopper_patience,
        n_particles=config.n_particles,
        lr=config.lr,
        seed=seed,
        save_path=model_path,
    )

    original_scale_data = normal_likelihood_module.Normal.scale_data
    if config.disable_normal_scale_data:
        normal_likelihood_module.Normal.scale_data = False
    try:
        return fit_mofaflex_model(truth.mdata, data_opts, model_opts, training_opts)
    finally:
        normal_likelihood_module.Normal.scale_data = original_scale_data


def evaluate_model(
    *,
    model,
    truth: SyntheticTruth,
    config: SyntheticBenchmarkConfig,
    true_dense: int,
    fitted_uninformed: int,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    view_name = next(iter(truth.mdata.mod.keys()))
    all_factors = list(model.factor_names)
    uninformed_names = factor_subset_names(model, "uninformed")
    informed_names = factor_subset_names(model, "informed")

    all_r2 = compute_processed_subset_r2(model, truth.mdata, factor_names=all_factors)
    informed_r2 = compute_processed_subset_r2(model, truth.mdata, factor_names=informed_names)
    uninformed_r2 = compute_processed_subset_r2(model, truth.mdata, factor_names=uninformed_names)
    per_factor_r2 = factor_total_r2_table(model)
    per_factor_r2["factor_type"] = per_factor_r2["factor"].map(
        lambda factor: "uninformed"
        if factor in uninformed_names
        else ("random_prior" if factor in truth.random_prior_names else "noisy_true_prior")
    )
    per_factor_r2.to_csv(run_dir / "per_factor_variance_explained.csv", index=False)

    weights = model.get_weights(return_type="pandas", ordered=False)[view_name].astype(float)
    factors = model.get_factors(return_type="pandas", ordered=False)[model.group_names[0]].astype(float)

    uninformed_corr, mean_abs_corr, max_abs_corr = uninformed_factor_correlation(factors, uninformed_names)
    uninformed_corr.to_csv(run_dir / "uninformed_factor_correlation.csv")

    true_informed_recovery = match_factor_weights(
        truth.true_weights.loc[truth.true_informed_sparse_factors],
        weights.reindex(truth.true_informed_sparse_factors).dropna(axis=0, how="all"),
    )
    hidden_sparse_recovery = match_factor_weights(
        truth.true_weights.loc[truth.true_hidden_sparse_factors],
        weights.loc[uninformed_names],
    )
    dense_recovery = match_factor_weights(
        truth.true_weights.loc[truth.true_dense_factors],
        weights.loc[uninformed_names],
    )
    dense_split = dense_split_metrics(dense_recovery["correlation_matrix"])

    true_informed_recovery["correlation_matrix"].to_csv(run_dir / "recovery_true_informed_sparse_correlation.csv")
    hidden_sparse_recovery["correlation_matrix"].to_csv(run_dir / "recovery_hidden_sparse_vs_uninformed_correlation.csv")
    dense_recovery["correlation_matrix"].to_csv(run_dir / "recovery_true_dense_vs_uninformed_correlation.csv")
    true_informed_recovery["matched_factor_correlations"].to_csv(run_dir / "recovery_true_informed_sparse_matches.csv", index=False)
    hidden_sparse_recovery["matched_factor_correlations"].to_csv(run_dir / "recovery_hidden_sparse_matches.csv", index=False)
    dense_recovery["matched_factor_correlations"].to_csv(run_dir / "recovery_true_dense_matches.csv", index=False)

    prior_usage = summarize_prior_usage(per_factor_r2)
    row = {
        "true_n_uninformed_factors": int(true_dense),
        "fitted_n_uninformed_factors": int(fitted_uninformed),
        "seed": int(seed),
        "n_informed_priors": int(len(informed_names)),
        "n_random_priors": int(len(truth.random_prior_names)),
        "total_variance_explained": float(all_r2["r2_total"]),
        "informed_variance_explained": float(informed_r2["r2_total"]),
        "uninformed_variance_explained": float(uninformed_r2["r2_total"]),
        "mean_abs_correlation_among_uninformed_factors": float(mean_abs_corr),
        "max_abs_correlation_among_uninformed_factors": float(max_abs_corr),
        "recovery_score_true_informed_factors": float(true_informed_recovery["mean_abs_corr"]),
        "recovery_score_hidden_sparse_factors": float(hidden_sparse_recovery["mean_abs_corr"]),
        "recovery_score_true_uninformed_factors": float(dense_recovery["mean_abs_corr"]),
        "dense_factor_mean_second_best_abs_corr": float(dense_split["mean_second_best_abs_corr"]),
        "dense_factor_mean_n_uninformed_above_0p5": float(dense_split["mean_n_uninformed_above_0p5"]),
        "random_prior_pruning_score": float(prior_usage["random_prior_pruning_score"]),
        "random_prior_usage_r2_mean": float(prior_usage["random_prior_usage_r2_mean"]),
        "random_prior_usage_r2_max": float(prior_usage["random_prior_usage_r2_max"]),
        "random_prior_usage_r2_sum": float(prior_usage["random_prior_usage_r2_sum"]),
        "true_prior_usage_r2_mean": float(prior_usage["true_prior_usage_r2_mean"]),
        "true_prior_usage_r2_sum": float(prior_usage["true_prior_usage_r2_sum"]),
        "random_to_true_prior_usage_ratio": float(prior_usage["random_to_true_prior_usage_ratio"]),
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    return row


def uninformed_factor_correlation(
    factors: pd.DataFrame,
    uninformed_names: list[str],
) -> tuple[pd.DataFrame, float, float]:
    if len(uninformed_names) == 0:
        return pd.DataFrame(), np.nan, np.nan
    corr = factors.loc[:, uninformed_names].corr(method="pearson").fillna(0.0)
    if len(uninformed_names) <= 1:
        return corr, 0.0, 0.0
    off_diag = corr.abs().where(~np.eye(len(corr), dtype=bool)).stack()
    return corr, float(off_diag.mean()), float(off_diag.max())


def dense_split_metrics(corr: pd.DataFrame, *, threshold: float = 0.5) -> dict[str, float]:
    if corr.empty:
        return {
            "mean_second_best_abs_corr": np.nan,
            "mean_n_uninformed_above_0p5": np.nan,
        }
    sorted_corr = np.sort(corr.to_numpy(dtype=float), axis=1)[:, ::-1]
    second_best = sorted_corr[:, 1] if sorted_corr.shape[1] > 1 else np.zeros(sorted_corr.shape[0], dtype=float)
    return {
        "mean_second_best_abs_corr": float(np.mean(second_best)),
        "mean_n_uninformed_above_0p5": float((corr.to_numpy(dtype=float) >= threshold).sum(axis=1).mean()),
    }


def summarize_prior_usage(per_factor_r2: pd.DataFrame) -> dict[str, float]:
    random_r2 = per_factor_r2.loc[per_factor_r2["factor_type"].eq("random_prior"), "matched_variance_explained"]
    true_r2 = per_factor_r2.loc[per_factor_r2["factor_type"].eq("noisy_true_prior"), "matched_variance_explained"]
    random_mean = float(random_r2.mean()) if not random_r2.empty else np.nan
    true_mean = float(true_r2.mean()) if not true_r2.empty else np.nan
    ratio = float(random_mean / true_mean) if true_mean and np.isfinite(true_mean) else np.nan
    return {
        "random_prior_usage_r2_mean": random_mean,
        "random_prior_usage_r2_max": float(random_r2.max()) if not random_r2.empty else np.nan,
        "random_prior_usage_r2_sum": float(random_r2.sum()) if not random_r2.empty else np.nan,
        "true_prior_usage_r2_mean": true_mean,
        "true_prior_usage_r2_sum": float(true_r2.sum()) if not true_r2.empty else np.nan,
        "random_to_true_prior_usage_ratio": ratio,
        "random_prior_pruning_score": float(1.0 - ratio) if np.isfinite(ratio) else np.nan,
    }


def save_truth_tables(*, truth: SyntheticTruth, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    truth.true_weights.to_csv(data_dir / "true_weights.csv")
    truth.true_mask.to_csv(data_dir / "true_mask.csv")
    truth.prior_mask.to_csv(data_dir / "model_prior_mask.csv")
    truth.prior_stats.to_csv(data_dir / "model_prior_stats.csv", index=False)
    if data_dir.joinpath("synthetic_data.h5mu").exists():
        return
    truth.mdata.write_h5mu(data_dir / "synthetic_data.h5mu")


def run_single(
    *,
    config: SyntheticBenchmarkConfig,
    true_dense: int,
    fitted_uninformed: int,
    seed: int,
    out_dir: Path,
    truth: SyntheticTruth | None = None,
) -> dict[str, Any]:
    data_dir = out_dir / "data" / f"true_dense_{true_dense:02d}" / f"seed_{seed:03d}"
    run_dir = out_dir / "runs" / f"true_dense_{true_dense:02d}" / f"fit_uninformed_{fitted_uninformed:02d}" / f"seed_{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if config.skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text())

    if truth is None:
        truth = generate_synthetic_truth(config=config, true_dense=true_dense, seed=seed)
    if config.save_data:
        save_truth_tables(truth=truth, data_dir=data_dir)

    model_path: Path | bool = run_dir / "model.h5" if config.save_model else False
    model = fit_benchmark_model(
        truth=truth,
        config=config,
        fitted_uninformed=fitted_uninformed,
        seed=seed,
        model_path=model_path,
    )
    return evaluate_model(
        model=model,
        truth=truth,
        config=config,
        true_dense=true_dense,
        fitted_uninformed=fitted_uninformed,
        seed=seed,
        run_dir=run_dir,
    )


def run_benchmark(config: SyntheticBenchmarkConfig, *, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))

    rows: list[dict[str, Any]] = []
    total_runs = len(config.true_dense_grid) * len(config.fitted_uninformed_grid) * len(config.seeds)
    run_idx = 0
    for true_dense in config.true_dense_grid:
        for seed in config.seeds:
            truth: SyntheticTruth | None = None
            data_dir = out_dir / "data" / f"true_dense_{true_dense:02d}" / f"seed_{seed:03d}"
            missing_runs = [
                fitted_uninformed
                for fitted_uninformed in config.fitted_uninformed_grid
                if not (
                    config.skip_existing
                    and (
                        out_dir
                        / "runs"
                        / f"true_dense_{true_dense:02d}"
                        / f"fit_uninformed_{fitted_uninformed:02d}"
                        / f"seed_{seed:03d}"
                        / "metrics.json"
                    ).exists()
                )
            ]
            if missing_runs:
                truth = generate_synthetic_truth(config=config, true_dense=true_dense, seed=seed)
                if config.save_data:
                    save_truth_tables(truth=truth, data_dir=data_dir)

            for fitted_uninformed in config.fitted_uninformed_grid:
                run_idx += 1
                print(
                    f"[{run_idx:03d}/{total_runs:03d}] true_dense={true_dense}, "
                    f"fit_uninformed={fitted_uninformed}, seed={seed}",
                    flush=True,
                )
                rows.append(
                    run_single(
                        config=config,
                        true_dense=true_dense,
                        fitted_uninformed=fitted_uninformed,
                        seed=seed,
                        out_dir=out_dir,
                        truth=truth,
                    )
                )

    summary = pd.DataFrame.from_records(rows).sort_values(
        ["true_n_uninformed_factors", "fitted_n_uninformed_factors", "seed"]
    )
    summary.to_csv(out_dir / "summary_per_run.csv", index=False)
    aggregate = aggregate_summary(summary)
    aggregate.to_csv(out_dir / "summary_by_regime.csv", index=False)
    plot_summary(summary=summary, aggregate=aggregate, out_dir=out_dir, heatmap_seed=config.heatmap_seed)
    return summary


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "total_variance_explained",
        "informed_variance_explained",
        "uninformed_variance_explained",
        "mean_abs_correlation_among_uninformed_factors",
        "max_abs_correlation_among_uninformed_factors",
        "recovery_score_true_informed_factors",
        "recovery_score_hidden_sparse_factors",
        "recovery_score_true_uninformed_factors",
        "dense_factor_mean_second_best_abs_corr",
        "dense_factor_mean_n_uninformed_above_0p5",
        "random_prior_pruning_score",
        "random_prior_usage_r2_mean",
        "random_prior_usage_r2_max",
        "true_prior_usage_r2_mean",
        "random_to_true_prior_usage_ratio",
    ]
    aggregate = summary.groupby(["true_n_uninformed_factors", "fitted_n_uninformed_factors"])[metric_cols].agg(
        ["mean", "std", "count"]
    )
    aggregate = aggregate.reset_index()
    aggregate.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in aggregate.columns.to_flat_index()
    ]
    return aggregate.sort_values(["true_n_uninformed_factors", "fitted_n_uninformed_factors"]).reset_index(drop=True)


def plot_summary(
    *,
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    out_dir: Path,
    heatmap_seed: int,
    plots_dir: Path | None = None,
) -> None:
    set_house_style()
    plots_dir = plots_dir or out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_variance_explained(summary, plots_dir / "variance_explained_vs_fitted_uninformed.png")
    plot_informed_uninformed_variance(summary, plots_dir / "informed_vs_uninformed_variance_explained.png")
    plot_uninformed_redundancy(summary, plots_dir / "uninformed_factor_redundancy_vs_fitted_uninformed.png")
    plot_recovery(summary, plots_dir / "true_structure_recovery_vs_fitted_uninformed.png")
    plot_prior_usage(summary, plots_dir / "true_vs_random_prior_usage.png")
    save_selected_uninformed_correlation_heatmaps(
        summary,
        out_dir=out_dir,
        heatmap_seed=heatmap_seed,
        plots_dir=plots_dir,
    )


def style_relplot_grid(g: sns.FacetGrid, *, legend_mode: str = "outside") -> None:
    for ax in g.axes.flat:
        clean_ax(ax)
    if g.legend is not None:
        g.legend.set_frame_on(False)
    elif g.axes.size == 1:
        place_legend(g.axes.flat[0], mode=legend_mode)


def lineplot_by_true_dense(
    df: pd.DataFrame,
    *,
    y: str,
    ylabel: str,
    path: Path,
    title: str,
) -> None:
    g = sns.relplot(
        data=df,
        x="fitted_n_uninformed_factors",
        y=y,
        hue="true_n_uninformed_factors",
        kind="line",
        marker="o",
        errorbar="sd",
        height=3.5,
        aspect=1.35,
        palette=CAT_PALETTE,
    )
    g.set_axis_labels("Fitted uninformed factors", ylabel)
    style_relplot_grid(g)
    g.fig.suptitle(title, y=1.04)
    savefig(g.fig, path)
    plt.close(g.fig)


def plot_variance_explained(summary: pd.DataFrame, path: Path) -> None:
    lineplot_by_true_dense(
        summary,
        y="total_variance_explained",
        ylabel="Total variance explained",
        path=path,
        title="Total variance explained vs fitted uninformed factors",
    )


def plot_informed_uninformed_variance(summary: pd.DataFrame, path: Path) -> None:
    long = summary.melt(
        id_vars=["true_n_uninformed_factors", "fitted_n_uninformed_factors", "seed"],
        value_vars=["informed_variance_explained", "uninformed_variance_explained"],
        var_name="factor_subset",
        value_name="variance_explained",
    )
    long["factor_subset"] = long["factor_subset"].map(
        {
            "informed_variance_explained": "Informed",
            "uninformed_variance_explained": "Uninformed",
        }
    )
    g = sns.relplot(
        data=long,
        x="fitted_n_uninformed_factors",
        y="variance_explained",
        hue="factor_subset",
        col="true_n_uninformed_factors",
        col_wrap=3,
        kind="line",
        marker="o",
        errorbar="sd",
        height=3.4,
        aspect=1.15,
        palette=CAT_PALETTE,
    )
    g.set_axis_labels("Fitted uninformed factors", "Variance explained")
    g.set_titles("True dense = {col_name}")
    style_relplot_grid(g)
    g.fig.suptitle("Informed vs uninformed variance explained", y=1.03)
    savefig(g.fig, path)
    plt.close(g.fig)


def plot_uninformed_redundancy(summary: pd.DataFrame, path: Path) -> None:
    long = summary.melt(
        id_vars=["true_n_uninformed_factors", "fitted_n_uninformed_factors", "seed"],
        value_vars=[
            "mean_abs_correlation_among_uninformed_factors",
            "max_abs_correlation_among_uninformed_factors",
        ],
        var_name="metric",
        value_name="absolute_correlation",
    )
    long["metric"] = long["metric"].map(
        {
            "mean_abs_correlation_among_uninformed_factors": "Mean |corr|",
            "max_abs_correlation_among_uninformed_factors": "Max |corr|",
        }
    )
    g = sns.relplot(
        data=long,
        x="fitted_n_uninformed_factors",
        y="absolute_correlation",
        hue="metric",
        col="true_n_uninformed_factors",
        col_wrap=3,
        kind="line",
        marker="o",
        errorbar="sd",
        height=3.4,
        aspect=1.15,
        palette=CAT_PALETTE,
    )
    g.set_axis_labels("Fitted uninformed factors", "Uninformed-factor |correlation|")
    g.set_titles("True dense = {col_name}")
    style_relplot_grid(g)
    g.fig.suptitle("Redundancy among inferred uninformed factors", y=1.03)
    savefig(g.fig, path)
    plt.close(g.fig)


def plot_recovery(summary: pd.DataFrame, path: Path) -> None:
    long = summary.melt(
        id_vars=["true_n_uninformed_factors", "fitted_n_uninformed_factors", "seed"],
        value_vars=[
            "recovery_score_true_informed_factors",
            "recovery_score_hidden_sparse_factors",
            "recovery_score_true_uninformed_factors",
        ],
        var_name="target",
        value_name="mean_abs_pearson",
    )
    long["target"] = long["target"].map(
        {
            "recovery_score_true_informed_factors": "40 informed sparse",
            "recovery_score_hidden_sparse_factors": "10 hidden sparse",
            "recovery_score_true_uninformed_factors": "true dense",
        }
    )
    g = sns.relplot(
        data=long,
        x="fitted_n_uninformed_factors",
        y="mean_abs_pearson",
        hue="target",
        col="true_n_uninformed_factors",
        col_wrap=3,
        kind="line",
        marker="o",
        errorbar="sd",
        height=3.4,
        aspect=1.15,
        palette=CAT_PALETTE,
    )
    g.set_axis_labels("Fitted uninformed factors", "Mean matched |Pearson|")
    g.set_titles("True dense = {col_name}")
    style_relplot_grid(g)
    g.fig.suptitle("Recovery of true synthetic structure", y=1.03)
    savefig(g.fig, path)
    plt.close(g.fig)


def plot_prior_usage(summary: pd.DataFrame, path: Path) -> None:
    long = summary.melt(
        id_vars=["true_n_uninformed_factors", "fitted_n_uninformed_factors", "seed"],
        value_vars=["true_prior_usage_r2_mean", "random_prior_usage_r2_mean"],
        var_name="prior_type",
        value_name="mean_per_factor_variance_explained",
    )
    long["prior_type"] = long["prior_type"].map(
        {
            "true_prior_usage_r2_mean": "40 noisy true priors",
            "random_prior_usage_r2_mean": "5 random priors",
        }
    )
    g = sns.relplot(
        data=long,
        x="fitted_n_uninformed_factors",
        y="mean_per_factor_variance_explained",
        hue="prior_type",
        col="true_n_uninformed_factors",
        col_wrap=3,
        kind="line",
        marker="o",
        errorbar="sd",
        height=3.4,
        aspect=1.15,
        palette=CAT_PALETTE,
    )
    g.set_axis_labels("Fitted uninformed factors", "Mean per-prior variance explained")
    g.set_titles("True dense = {col_name}")
    style_relplot_grid(g)
    g.fig.suptitle("Usage of noisy true priors vs random fake priors", y=1.03)
    savefig(g.fig, path)
    plt.close(g.fig)


def save_selected_uninformed_correlation_heatmaps(
    summary: pd.DataFrame,
    *,
    out_dir: Path,
    heatmap_seed: int,
    plots_dir: Path | None = None,
) -> None:
    heatmap_dir = (plots_dir or out_dir / "plots") / f"uninformed_factor_correlation_seed_{heatmap_seed:03d}"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    selected = summary.loc[summary["seed"].eq(heatmap_seed), :].copy()
    for _, row in selected.iterrows():
        run_dir = Path(row["run_dir"])
        matrix_path = run_dir / "uninformed_factor_correlation.csv"
        if not matrix_path.exists():
            continue
        corr = pd.read_csv(matrix_path, index_col=0)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        sns.heatmap(corr, ax=ax, vmin=-1.0, vmax=1.0, cmap=DIVERGING_CMAP, center=0.0, annot=True, fmt=".2f")
        ax.set_title(
            f"True dense {int(row['true_n_uninformed_factors'])}, fitted {int(row['fitted_n_uninformed_factors'])}"
        )
        clean_ax(ax)
        fig.tight_layout()
        savefig(
            fig,
            heatmap_dir
            / f"true_dense_{int(row['true_n_uninformed_factors']):02d}_fit_uninformed_{int(row['fitted_n_uninformed_factors']):02d}.png",
        )
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    run_benchmark(config, out_dir=args.out_dir.resolve())


if __name__ == "__main__":
    main()
