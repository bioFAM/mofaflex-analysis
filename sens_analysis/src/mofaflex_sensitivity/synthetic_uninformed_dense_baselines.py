from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.synthetic_uninformed_dense_benchmark import (
        SyntheticBenchmarkConfig,
        SyntheticTruth,
        aggregate_summary,
        dense_split_metrics,
        generate_synthetic_truth,
        plot_summary,
        save_truth_tables,
        summarize_prior_usage,
        uninformed_factor_correlation,
    )
    from mofaflex_sensitivity.uninformed_sensitivity import match_factor_weights
else:
    from .synthetic_uninformed_dense_benchmark import (
        SyntheticBenchmarkConfig,
        SyntheticTruth,
        aggregate_summary,
        dense_split_metrics,
        generate_synthetic_truth,
        plot_summary,
        save_truth_tables,
        summarize_prior_usage,
        uninformed_factor_correlation,
    )
    from .uninformed_sensitivity import match_factor_weights


DEFAULT_OUT_DIR = Path("artifacts/synthetic_uninformed_dense_baselines")
BASELINE_METHODS = ("expimap", "spectra")


def _spectra_gene_scalings_array(model) -> np.ndarray:
    gene_scalings = model.return_gene_scalings()
    if isinstance(gene_scalings, dict):
        if "global" in gene_scalings:
            gene_scalings = gene_scalings["global"]
        elif gene_scalings:
            gene_scalings = next(iter(gene_scalings.values()))
        else:
            raise ValueError("Spectra returned an empty gene-scaling dictionary.")
    return np.asarray(gene_scalings, dtype=float)


def align_weights_to_prior_mask(weights: pd.DataFrame, prior_mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign soft-mask baseline factors to priors with one-to-one AP matching."""
    from scipy.optimize import linear_sum_assignment

    scores = weights.abs().reindex(columns=prior_mask.columns).to_numpy(dtype=float)
    mask = prior_mask.to_numpy(dtype=bool)
    ap = np.empty((scores.shape[0], mask.shape[0]), dtype=float)
    n_active = mask.sum(axis=1).astype(float)
    positive = n_active > 0
    ranks = np.arange(1, mask.shape[1] + 1, dtype=float)
    for factor_idx in range(scores.shape[0]):
        factor_scores = scores[factor_idx]
        if np.allclose(factor_scores, factor_scores[0]):
            ap[factor_idx] = n_active / mask.shape[1]
            continue
        order = np.argsort(-factor_scores, kind="mergesort")
        sorted_mask = mask[:, order]
        precision_at_rank = np.cumsum(sorted_mask, axis=1) / ranks
        ap[factor_idx] = np.divide(
            (precision_at_rank * sorted_mask).sum(axis=1),
            n_active,
            out=np.zeros(mask.shape[0], dtype=float),
            where=positive,
        )

    factor_indices, pathway_indices = linear_sum_assignment(-ap)
    factor_by_pathway = dict(zip(pathway_indices, factor_indices, strict=True))
    ordered_factor_indices = [factor_by_pathway[pathway_idx] for pathway_idx in range(mask.shape[0])]
    aligned = weights.iloc[ordered_factor_indices].copy()
    aligned.index = prior_mask.index
    assignment = pd.DataFrame(
        {
            "pathway_name": prior_mask.index,
            "factor_name": weights.index[ordered_factor_indices],
            "assignment_average_precision_noisy": [
                float(ap[factor_by_pathway[pathway_idx], pathway_idx]) for pathway_idx in range(mask.shape[0])
            ],
        }
    )
    return aligned, assignment


@dataclass(frozen=True)
class BaselineBenchmarkConfig:
    n_samples: int = 10_000
    n_features: int = 5_000
    n_true_sparse_factors: int = 50
    n_informed_true_priors: int = 40
    n_random_priors: int = 5
    true_dense_grid: tuple[int, ...] = (1, 2, 3, 5, 10)
    fitted_uninformed_grid: tuple[int, ...] = (1, 2, 3, 5, 10)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    methods: tuple[str, ...] = BASELINE_METHODS
    prior_noise_fraction: float = 0.5
    expimap_epochs: int = 500
    expimap_batch_size: int | None = None
    expimap_soft_mask: bool = True
    expimap_recon_loss: str = "mse"
    expimap_hidden_size_1: int = 512
    expimap_hidden_size_2: int = 128
    spectra_epochs: int = 10_000
    spectra_use_gpu: bool = True
    spectra_batch_size: int = 1000
    spectra_overlap_top_n: int = 50
    spectra_overlap_threshold: float = 0.2
    save_model: bool = True
    save_data: bool = True
    skip_existing: bool = True
    heatmap_seed: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train ExpiMap and Spectra baselines on the synthetic uninformed-dense benchmark datasets. "
            "Run this from the mfl_sens_baselines conda environment."
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
    parser.add_argument("--methods", nargs="+", default=list(BASELINE_METHODS), choices=list(BASELINE_METHODS))
    parser.add_argument("--prior-noise-fraction", type=float, default=0.5)
    parser.add_argument("--expimap-epochs", type=int, default=500)
    parser.add_argument("--expimap-batch-size", type=int, default=None)
    parser.add_argument("--expimap-soft-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expimap-recon-loss", type=str, default="mse")
    parser.add_argument("--expimap-hidden-size-1", type=int, default=512)
    parser.add_argument("--expimap-hidden-size-2", type=int, default=128)
    parser.add_argument("--spectra-epochs", type=int, default=10_000)
    parser.add_argument("--spectra-use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spectra-batch-size", type=int, default=1000)
    parser.add_argument("--spectra-overlap-top-n", type=int, default=50)
    parser.add_argument("--spectra-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heatmap-seed", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> BaselineBenchmarkConfig:
    return BaselineBenchmarkConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_true_sparse_factors=args.n_true_sparse_factors,
        n_informed_true_priors=args.n_informed_true_priors,
        n_random_priors=args.n_random_priors,
        true_dense_grid=tuple(args.true_dense_grid),
        fitted_uninformed_grid=tuple(args.fitted_uninformed_grid),
        seeds=tuple(args.seeds),
        methods=tuple(args.methods),
        prior_noise_fraction=args.prior_noise_fraction,
        expimap_epochs=args.expimap_epochs,
        expimap_batch_size=args.expimap_batch_size,
        expimap_soft_mask=args.expimap_soft_mask,
        expimap_recon_loss=args.expimap_recon_loss,
        expimap_hidden_size_1=args.expimap_hidden_size_1,
        expimap_hidden_size_2=args.expimap_hidden_size_2,
        spectra_epochs=args.spectra_epochs,
        spectra_use_gpu=args.spectra_use_gpu,
        spectra_batch_size=args.spectra_batch_size,
        spectra_overlap_top_n=args.spectra_overlap_top_n,
        spectra_overlap_threshold=args.spectra_overlap_threshold,
        save_model=args.save_model,
        save_data=args.save_data,
        skip_existing=args.skip_existing,
        heatmap_seed=args.heatmap_seed,
    )


def benchmark_config(config: BaselineBenchmarkConfig) -> SyntheticBenchmarkConfig:
    return SyntheticBenchmarkConfig(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_true_sparse_factors=config.n_true_sparse_factors,
        n_informed_true_priors=config.n_informed_true_priors,
        n_random_priors=config.n_random_priors,
        true_dense_grid=config.true_dense_grid,
        fitted_uninformed_grid=config.fitted_uninformed_grid,
        seeds=config.seeds,
        prior_noise_fraction=config.prior_noise_fraction,
        save_model=config.save_model,
        save_data=config.save_data,
        skip_existing=config.skip_existing,
        heatmap_seed=config.heatmap_seed,
    )


def _dense_array(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def build_baseline_inputs(
    truth: SyntheticTruth,
    *,
    fitted_uninformed: int,
) -> tuple[np.ndarray, pd.DataFrame, list[str], list[str]]:
    view_name = next(iter(truth.mdata.mod.keys()))
    data = _dense_array(truth.mdata[view_name].X).astype(np.float32, copy=False)
    dense_names = [f"uninformed_dense_{idx:02d}" for idx in range(fitted_uninformed)]
    dense_mask = pd.DataFrame(
        True,
        index=dense_names,
        columns=truth.prior_mask.columns,
    )
    baseline_mask = pd.concat([truth.prior_mask, dense_mask], axis=0).astype(bool)
    return data, baseline_mask, baseline_mask.index.tolist(), dense_names


def train_expimap(
    data: np.ndarray,
    mask: pd.DataFrame,
    *,
    terms: list[str],
    seed: int,
    config: BaselineBenchmarkConfig,
):
    import scarches as sca

    adata = ad.AnnData(data.copy())
    adata.obs["cond"] = "cond"
    adata.varm["I"] = mask.to_numpy(dtype=bool).T
    adata.uns["terms"] = terms

    batch_size = config.expimap_batch_size or adata.n_obs
    model = sca.models.EXPIMAP(
        adata=adata,
        condition_key="cond",
        hidden_layer_sizes=[config.expimap_hidden_size_1, config.expimap_hidden_size_2],
        recon_loss=config.expimap_recon_loss,
        soft_mask=config.expimap_soft_mask,
    )
    early_stopping_kwargs = {
        "early_stopping_metric": "val_unweighted_loss",
        "threshold": 0,
        "patience": 50,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    model.train(
        n_epochs=config.expimap_epochs,
        use_early_stopping=True,
        early_stopping_kwargs=early_stopping_kwargs,
        batch_size=batch_size,
        monitor_only_val=False,
        seed=seed,
    )
    return model


def train_spectra(
    data: np.ndarray,
    mask: pd.DataFrame,
    *,
    terms: list[str],
    feature_names: pd.Index | list[str] | None = None,
    seed: int,
    config: BaselineBenchmarkConfig,
):
    adata = ad.AnnData(data.copy())
    if feature_names is not None:
        adata.var_names = pd.Index(feature_names).astype(str)
    annot = {f"all_{term}": mask.columns[mask.loc[term, :].to_numpy(dtype=bool)].tolist() for term in terms}
    use_gpu = bool(getattr(config, "spectra_use_gpu", True))
    if not use_gpu:
        raise NotImplementedError(
            "Spectra baselines in this repo use a single GPU path only. "
            "Provide global annotations and they will be wrapped internally for Spectra."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Spectra baseline requested GPU execution, but CUDA is not available.")

    # User-facing benchmark semantics are global gene-set annotations only.
    # Spectra GPU does not support use_cell_types=False, so we keep the GPU
    # code path alive with one empty pseudo-cell-type bucket while storing the
    # actual annotations in the nested global block.
    adata.obs["celltype_dummy"] = "ct"
    gene_set_dictionary = {"global": annot, "ct": {}}
    factor_counts = {"global": len(annot), "ct": 0}
    zero_init_scores = {"global": {name: 0.0 for name in annot}, "ct": {}}

    import Spectra.Spectra_gpu as spectra_gpu

    torch.manual_seed(seed)
    original_compute_init_scores = spectra_gpu.compute_init_scores
    original_return_markers = spectra_gpu.return_markers

    def _safe_compute_init_scores(*args, **kwargs):
        try:
            return original_compute_init_scores(*args, **kwargs)
        except ZeroDivisionError:
            return zero_init_scores

    def _fixed_return_markers(factor_matrix, id2word, n_top_vals=100):
        # Upstream initialises with np.zeros (float64); newer pandas refuses
        # string assignment into float columns. Use object dtype instead.
        idx_matrix = np.argsort(factor_matrix, axis=1)[:, ::-1][:, :n_top_vals]
        df = pd.DataFrame(np.empty(idx_matrix.shape, dtype=object))
        for i in range(idx_matrix.shape[0]):
            for j in range(idx_matrix.shape[1]):
                df.iloc[i, j] = id2word[idx_matrix[i, j]]
        return df.values

    spectra_gpu.compute_init_scores = _safe_compute_init_scores
    spectra_gpu.return_markers = _fixed_return_markers
    spectra_kwargs = {
        "adata": adata,
        "gene_set_dictionary": gene_set_dictionary,
        "L": factor_counts,
        "use_highly_variable": False,
        "cell_type_key": "celltype_dummy",
        "batch_size": int(getattr(config, "spectra_batch_size", 1000)),
        "num_epochs": config.spectra_epochs,
        "use_weights": True,
        "kappa": None,
        "use_cell_types": True,
    }

    try:
        return spectra_gpu.est_spectra(**spectra_kwargs)
    finally:
        spectra_gpu.compute_init_scores = original_compute_init_scores
        spectra_gpu.return_markers = original_return_markers


def expimap_arrays(model, *, terms: list[str], feature_names: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    # Use the posterior mean so factors and reconstructions come from the same
    # latent draw-free representation.
    weights = model.model.decoder.L0.expr_L.weight.detach().cpu().numpy().T
    factors = model.get_latent(mean=True)
    y_pred = factors @ weights
    return (
        pd.DataFrame(weights, index=terms, columns=feature_names),
        pd.DataFrame(factors, columns=terms),
        np.asarray(y_pred, dtype=float),
    )


def spectra_arrays(model, *, terms: list[str], feature_names: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    import torch
    from opt_einsum import contract

    # Spectra's public return_factors()/return_cell_scores() accessors expose a
    # post-processed summary that is not the additive matrix factorization used
    # in the reconstruction loss. Recover the raw alpha @ theta decomposition so
    # weights, factors, and reconstruction all live on the same scale.
    internal_model = model.internal_model
    factors_array = (torch.exp(internal_model.alpha) * internal_model.alpha_mask).detach().cpu().numpy()
    theta = torch.cat(
        [
            temp.softmax(dim=1)
            for temp in torch.split(
                internal_model.theta,
                split_size_or_sections=internal_model.L_list,
                dim=1,
            )
        ],
        dim=1,
    )
    gene_scaling = internal_model.gene_scalings.exp() / (1.0 + internal_model.gene_scalings.exp())
    gene_scaling_ = contract("ij,ki->jk", gene_scaling, internal_model.factor_to_celltype)
    weights_array = (theta * (gene_scaling_ + internal_model.delta)).T.detach().cpu().numpy()

    n_terms = len(terms)
    weights = weights_array[:n_terms, :]
    factors = factors_array[:, :n_terms]
    y_pred = factors_array @ weights_array
    return (
        pd.DataFrame(weights, index=terms, columns=feature_names),
        pd.DataFrame(factors, columns=terms),
        np.asarray(y_pred, dtype=float),
    )


def per_factor_r2_table(data: np.ndarray, factors: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ss_tot = float(np.nansum(np.square(data)))
    for factor_name in weights.index:
        if factor_name not in factors.columns:
            rows.append({"factor": factor_name, "matched_variance_explained": np.nan})
            continue
        y_pred = np.outer(
            factors[factor_name].to_numpy(dtype=float),
            weights.loc[factor_name].to_numpy(dtype=float),
        )
        ss_res = float(np.nansum(np.square(data - y_pred)))
        rows.append(
            {
                "factor": factor_name,
                "matched_variance_explained": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def subset_r2(data: np.ndarray, factors: pd.DataFrame, weights: pd.DataFrame, factor_names: list[str]) -> float:
    ss_tot = float(np.nansum(np.square(data)))
    if ss_tot <= 0:
        return np.nan
    if not factor_names:
        y_pred = np.zeros_like(data, dtype=float)
    else:
        y_pred = (
            factors.reindex(columns=factor_names, fill_value=0.0).to_numpy(dtype=float)
            @ weights.reindex(index=factor_names, fill_value=0.0).to_numpy(dtype=float)
        )
    ss_res = float(np.nansum(np.square(data - y_pred)))
    return float(1.0 - ss_res / ss_tot)


def evaluate_baseline(
    *,
    method: str,
    model,
    data: np.ndarray,
    terms: list[str],
    dense_names: list[str],
    prior_mask: pd.DataFrame,
    truth: SyntheticTruth,
    true_dense: int,
    fitted_uninformed: int,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    feature_names = truth.prior_mask.columns
    if method == "expimap":
        weights, factors, y_pred = expimap_arrays(model, terms=terms, feature_names=feature_names)
        weights.to_csv(run_dir / "weights_raw.csv")
        weights, assignment = align_weights_to_prior_mask(weights, prior_mask)
        assignment.to_csv(run_dir / "expimap_aupr_factor_assignment.csv", index=False)
        raw_factors = factors
        aligned_factor_columns = []
        for row in assignment.itertuples(index=False):
            aligned_factor_columns.append(raw_factors.loc[:, str(row.factor_name)].to_numpy(dtype=float))
        factors = pd.DataFrame(np.column_stack(aligned_factor_columns), columns=prior_mask.index)
    elif method == "spectra":
        weights, factors, y_pred = spectra_arrays(model, terms=terms, feature_names=feature_names)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    weights.to_csv(run_dir / "weights.csv")
    factors.to_csv(run_dir / "factors.csv", index=False)
    np.save(run_dir / "reconstruction.npy", y_pred.astype(np.float32, copy=False))

    informed_names = [name for name in terms if name not in dense_names]
    per_factor_r2 = per_factor_r2_table(data, factors, weights)
    per_factor_r2["factor_type"] = per_factor_r2["factor"].map(
        lambda factor: "uninformed"
        if factor in dense_names
        else ("random_prior" if factor in truth.random_prior_names else "noisy_true_prior")
    )
    per_factor_r2.to_csv(run_dir / "per_factor_variance_explained.csv", index=False)

    ss_tot = float(np.nansum(np.square(data)))
    ss_res = float(np.nansum(np.square(data - y_pred)))
    total_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    total_rmse = float(np.sqrt(np.nanmean(np.square(data - y_pred))))
    informed_r2 = subset_r2(data, factors, weights, informed_names)
    uninformed_r2 = subset_r2(data, factors, weights, dense_names)

    uninformed_corr, mean_abs_corr, max_abs_corr = uninformed_factor_correlation(factors, dense_names)
    uninformed_corr.to_csv(run_dir / "uninformed_factor_correlation.csv")

    true_informed_recovery = match_factor_weights(
        truth.true_weights.loc[truth.true_informed_sparse_factors],
        weights.reindex(truth.true_informed_sparse_factors).dropna(axis=0, how="all"),
    )
    hidden_sparse_recovery = match_factor_weights(
        truth.true_weights.loc[truth.true_hidden_sparse_factors],
        weights.reindex(dense_names).dropna(axis=0, how="all"),
    )
    dense_recovery = match_factor_weights(
        truth.true_weights.loc[truth.true_dense_factors],
        weights.reindex(dense_names).dropna(axis=0, how="all"),
    )
    dense_split = dense_split_metrics(dense_recovery["correlation_matrix"])
    true_informed_recovery["correlation_matrix"].to_csv(run_dir / "recovery_true_informed_sparse_correlation.csv")
    hidden_sparse_recovery["correlation_matrix"].to_csv(run_dir / "recovery_hidden_sparse_vs_uninformed_correlation.csv")
    dense_recovery["correlation_matrix"].to_csv(run_dir / "recovery_true_dense_vs_uninformed_correlation.csv")

    prior_usage = summarize_prior_usage(per_factor_r2)
    row = {
        "method": method,
        "true_n_uninformed_factors": int(true_dense),
        "fitted_n_uninformed_factors": int(fitted_uninformed),
        "seed": int(seed),
        "n_informed_priors": int(len(informed_names)),
        "n_random_priors": int(len(truth.random_prior_names)),
        "total_variance_explained": total_r2,
        "total_rmse": total_rmse,
        "informed_variance_explained": float(informed_r2),
        "uninformed_variance_explained": float(uninformed_r2),
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


def save_baseline_model(method: str, model, run_dir: Path) -> None:
    if method == "expimap":
        try:
            model.save(run_dir / "expimap_model", overwrite=True)
        except Exception as exc:  # pragma: no cover - scArches save API varies by version.
            (run_dir / "model_save_warning.txt").write_text(str(exc))
    elif method == "spectra":
        try:
            import torch

            torch.save(model, run_dir / "spectra_model.pt")
        except Exception as exc:  # pragma: no cover - Spectra object pickling may vary by version.
            (run_dir / "model_save_warning.txt").write_text(str(exc))


def run_single_baseline(
    *,
    config: BaselineBenchmarkConfig,
    method: str,
    truth: SyntheticTruth,
    true_dense: int,
    fitted_uninformed: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    run_dir = (
        out_dir
        / "runs"
        / method
        / f"true_dense_{true_dense:02d}"
        / f"fit_uninformed_{fitted_uninformed:02d}"
        / f"seed_{seed:03d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if config.skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text())

    data, mask, terms, dense_names = build_baseline_inputs(truth, fitted_uninformed=fitted_uninformed)
    mask.to_csv(run_dir / "baseline_prior_mask.csv")
    if method == "expimap":
        model = train_expimap(data, mask, terms=terms, seed=seed, config=config)
    elif method == "spectra":
        model = train_spectra(data, mask, terms=terms, feature_names=mask.columns, seed=seed, config=config)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    if config.save_model:
        save_baseline_model(method, model, run_dir)
    return evaluate_baseline(
        method=method,
        model=model,
        data=data,
        terms=terms,
        dense_names=dense_names,
        prior_mask=mask,
        truth=truth,
        true_dense=true_dense,
        fitted_uninformed=fitted_uninformed,
        seed=seed,
        run_dir=run_dir,
    )


def run_benchmark(config: BaselineBenchmarkConfig, *, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))
    synth_config = benchmark_config(config)

    rows: list[dict[str, Any]] = []
    total_runs = (
        len(config.methods)
        * len(config.true_dense_grid)
        * len(config.fitted_uninformed_grid)
        * len(config.seeds)
    )
    run_idx = 0
    for true_dense in config.true_dense_grid:
        for seed in config.seeds:
            truth = generate_synthetic_truth(config=synth_config, true_dense=true_dense, seed=seed)
            if config.save_data:
                data_dir = out_dir / "data" / f"true_dense_{true_dense:02d}" / f"seed_{seed:03d}"
                save_truth_tables(truth=truth, data_dir=data_dir)
            for fitted_uninformed in config.fitted_uninformed_grid:
                for method in config.methods:
                    run_idx += 1
                    print(
                        f"[{run_idx:03d}/{total_runs:03d}] method={method}, true_dense={true_dense}, "
                        f"fit_uninformed={fitted_uninformed}, seed={seed}",
                        flush=True,
                    )
                    rows.append(
                        run_single_baseline(
                            config=config,
                            method=method,
                            truth=truth,
                            true_dense=true_dense,
                            fitted_uninformed=fitted_uninformed,
                            seed=seed,
                            out_dir=out_dir,
                        )
                    )

    summary = pd.DataFrame.from_records(rows).sort_values(
        ["method", "true_n_uninformed_factors", "fitted_n_uninformed_factors", "seed"]
    )
    summary.to_csv(out_dir / "summary_per_run.csv", index=False)
    aggregates = []
    for method, method_df in summary.groupby("method", sort=True):
        method_aggregate = aggregate_summary(method_df.drop(columns=["method"]))
        method_aggregate.insert(0, "method", method)
        aggregates.append(method_aggregate)
        plot_summary(
            summary=method_df,
            aggregate=method_aggregate,
            out_dir=out_dir,
            heatmap_seed=config.heatmap_seed,
            plots_dir=out_dir / "plots" / method,
        )
    aggregate = pd.concat(aggregates, ignore_index=True) if aggregates else pd.DataFrame()
    aggregate.to_csv(out_dir / "summary_by_regime.csv", index=False)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    run_benchmark(config, out_dir=args.out_dir.resolve())


if __name__ == "__main__":
    main()
