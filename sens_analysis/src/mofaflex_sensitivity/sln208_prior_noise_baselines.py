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
    from mofaflex_sensitivity.sln208_prior_noise_refinement import (
        DEFAULT_DATA_PATH,
        best_f1_threshold,
        bool_mask_from_gene_set_stats,
        build_paper_hallmark_mca_gene_set_stats,
        infer_pathway_source,
        noisy_gene_set_stats_from_mask,
        perturb_mask,
        save_refinement_scatter,
        summarize_refinement_table,
        top_k_overlap_metrics,
    )
    from mofaflex_sensitivity.sln208_prior_noise_refinement_plotting import (
        aggregate_with_errorbars,
        save_metric_grid_plot,
    )
    from mofaflex_sensitivity.sln208_prior_noise_refinement_sweep import (
        FOCUS_SOURCE_PANELS,
        SOURCE_SUMMARY_COLUMNS,
        SUMMARY_PANELS,
        _slugify_noise_level,
        save_focus_source_errorbar_outputs,
        save_focus_source_outputs,
        save_summary_errorbar_plot,
        save_summary_plot,
        summarize_by_source,
        summarize_by_source_per_run,
    )
    from mofaflex_sensitivity.synthetic_uninformed_dense_baselines import (
        _spectra_gene_scalings_array,
        train_expimap,
        train_spectra,
    )
    from mofaflex_sensitivity.mca_filter_sensitivity import load_sln208_mdata
else:
    from .mca_filter_sensitivity import load_sln208_mdata
    from .sln208_prior_noise_refinement import (
        DEFAULT_DATA_PATH,
        best_f1_threshold,
        bool_mask_from_gene_set_stats,
        build_paper_hallmark_mca_gene_set_stats,
        infer_pathway_source,
        noisy_gene_set_stats_from_mask,
        perturb_mask,
        save_refinement_scatter,
        summarize_refinement_table,
        top_k_overlap_metrics,
    )
    from .sln208_prior_noise_refinement_plotting import aggregate_with_errorbars, save_metric_grid_plot
    from .sln208_prior_noise_refinement_sweep import (
        FOCUS_SOURCE_PANELS,
        SOURCE_SUMMARY_COLUMNS,
        SUMMARY_PANELS,
        _slugify_noise_level,
        save_focus_source_errorbar_outputs,
        save_focus_source_outputs,
        save_summary_errorbar_plot,
        save_summary_plot,
        summarize_by_source,
        summarize_by_source_per_run,
    )
    from .synthetic_uninformed_dense_baselines import _spectra_gene_scalings_array, train_expimap, train_spectra


DEFAULT_REFERENCE_SWEEP_DIR = Path("artifacts/sln208_prior_noise_refinement_sweep_mca_0to40_5seeds_noise")
DEFAULT_OUT_DIR = Path("artifacts/sln208_prior_noise_baselines_mca_0to40_5seeds_noise")
BASELINE_METHODS = ("expimap", "spectra")


@dataclass(frozen=True)
class Sln208BaselineConfig:
    data_path: Path = DEFAULT_DATA_PATH
    reference_sweep_dir: Path = DEFAULT_REFERENCE_SWEEP_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    methods: tuple[str, ...] = BASELINE_METHODS
    focus_source: str | None = "mca"
    noise_levels: tuple[float, ...] | None = None
    seeds: tuple[int, ...] | None = None
    expimap_epochs: int = 500
    expimap_batch_size: int | None = None
    expimap_soft_mask: bool = True
    expimap_recon_loss: str = "mse"
    expimap_hidden_size_1: int = 512
    expimap_hidden_size_2: int = 128
    spectra_epochs: int = 10_000
    spectra_use_gpu: bool = True
    spectra_batch_size: int = 1000
    spectra_assignment: str = "overlap"
    spectra_overlap_top_n: int = 50
    spectra_overlap_threshold: float = 0.2
    save_model: bool = True
    skip_existing: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train ExpiMap and Spectra RNA-only baselines on the same SLN208 prior-noise masks used "
            "for the MOFA-FLEX refinement sweep."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--reference-sweep-dir", type=Path, default=DEFAULT_REFERENCE_SWEEP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", nargs="+", default=list(BASELINE_METHODS), choices=list(BASELINE_METHODS))
    parser.add_argument("--focus-source", choices=["hallmark", "mca"], default="mca")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Optional subset of reference noise levels to run. Defaults to all levels in the reference sweep.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of reference seeds to run. Defaults to all seeds in the reference sweep.",
    )
    parser.add_argument("--expimap-epochs", type=int, default=500)
    parser.add_argument("--expimap-batch-size", type=int, default=None)
    parser.add_argument("--expimap-soft-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expimap-recon-loss", type=str, default="mse")
    parser.add_argument("--expimap-hidden-size-1", type=int, default=512)
    parser.add_argument("--expimap-hidden-size-2", type=int, default=128)
    parser.add_argument("--spectra-epochs", type=int, default=10_000)
    parser.add_argument("--spectra-use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spectra-batch-size", type=int, default=1000)
    parser.add_argument(
        "--spectra-assignment",
        choices=["aupr", "overlap"],
        default="overlap",
        help=(
            "How to assign unlabeled Spectra factors to input gene sets for evaluation. "
            "'aupr' uses one-to-one average-precision matching; 'overlap' uses Spectra's CPU-style "
            "top-marker overlap labels."
        ),
    )
    parser.add_argument("--spectra-overlap-top-n", type=int, default=50)
    parser.add_argument("--spectra-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def config_from_args(args: argparse.Namespace) -> Sln208BaselineConfig:
    return Sln208BaselineConfig(
        data_path=args.data_path,
        reference_sweep_dir=args.reference_sweep_dir,
        out_dir=args.out_dir,
        methods=tuple(args.methods),
        focus_source=args.focus_source,
        noise_levels=None if args.noise_levels is None else tuple(float(x) for x in args.noise_levels),
        seeds=None if args.seeds is None else tuple(int(x) for x in args.seeds),
        expimap_epochs=args.expimap_epochs,
        expimap_batch_size=args.expimap_batch_size,
        expimap_soft_mask=args.expimap_soft_mask,
        expimap_recon_loss=args.expimap_recon_loss,
        expimap_hidden_size_1=args.expimap_hidden_size_1,
        expimap_hidden_size_2=args.expimap_hidden_size_2,
        spectra_epochs=args.spectra_epochs,
        spectra_use_gpu=args.spectra_use_gpu,
        spectra_batch_size=args.spectra_batch_size,
        spectra_assignment=args.spectra_assignment,
        spectra_overlap_top_n=args.spectra_overlap_top_n,
        spectra_overlap_threshold=args.spectra_overlap_threshold,
        save_model=args.save_model,
        skip_existing=args.skip_existing,
    )


def reference_grid(reference_sweep_dir: Path) -> pd.DataFrame:
    summary_path = reference_sweep_dir / "summary_per_run.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing reference summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = ["fpr", "fnr", "seed", "noise_seed", "noise_level"]
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise ValueError(f"Reference summary is missing columns: {missing}")
    return (
        summary.loc[:, required]
        .drop_duplicates()
        .sort_values(["noise_level", "seed"])
        .reset_index(drop=True)
    )


def _dense_array(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def train_baseline_model(
    *,
    method: str,
    data: np.ndarray,
    noisy_mask: pd.DataFrame,
    feature_names: pd.Index,
    seed: int,
    config: Sln208BaselineConfig,
):
    if method == "expimap":
        return train_expimap(
            data,
            noisy_mask,
            terms=noisy_mask.index.tolist(),
            seed=seed,
            config=config,
        )
    if method == "spectra":
        np.random.seed(seed)
        return train_spectra(
            data,
            noisy_mask,
            terms=noisy_mask.index.tolist(),
            feature_names=feature_names,
            seed=seed,
            config=config,
        )
    raise ValueError(f"Unsupported method: {method}")


def preprocess_baseline_input(method: str, data: np.ndarray) -> tuple[np.ndarray, str]:
    """Return the method-specific RNA input matrix used for training and scoring."""
    if method == "expimap":
        return (data - data.mean(axis=0, keepdims=True)).astype(np.float32, copy=False), "per_feature_centered"
    if method == "spectra":
        return data, "stored_nonnegative_rna"
    raise ValueError(f"Unsupported method: {method}")


def model_weights(method: str, model, *, terms: list[str], var_names: pd.Index) -> pd.DataFrame:
    if method == "expimap":
        weights = model.model.decoder.L0.expr_L.weight.detach().cpu().numpy().T
        return pd.DataFrame(weights, index=terms, columns=var_names)
    elif method == "spectra":
        weights = np.asarray(model.return_factors(), dtype=float)
        return pd.DataFrame(
            weights,
            index=[f"spectra_factor_{idx:03d}" for idx in range(weights.shape[0])],
            columns=var_names,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


def align_weights_to_prior_mask(weights: pd.DataFrame, prior_mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign unlabeled factor loadings to input priors with one-to-one AP matching."""
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
    missing = sorted(set(range(mask.shape[0])) - set(factor_by_pathway))
    if missing:
        raise ValueError(f"Could not assign factors for all pathways; missing pathway indices: {missing}")

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


def spectra_overlap_label_table(
    raw_weights: pd.DataFrame,
    prior_mask: pd.DataFrame,
    *,
    n_top_vals: int = 50,
    overlap_threshold: float = 0.2,
) -> pd.DataFrame:
    """Reproduce Spectra CPU-style marker-overlap labels for GPU Spectra factors."""
    import Spectra.Spectra_util as spectra_util

    top_idx = np.argsort(raw_weights.to_numpy(dtype=float), axis=1)[:, ::-1][:, :n_top_vals]
    markers = pd.DataFrame(raw_weights.columns.to_numpy()[top_idx], index=raw_weights.index)
    gene_set_dictionary = {
        f"all_{pathway_name}": prior_mask.columns[prior_mask.loc[pathway_name].to_numpy(dtype=bool)].tolist()
        for pathway_name in prior_mask.index
    }
    overlap_df = spectra_util.label_marker_genes(markers.to_numpy(), gene_set_dictionary, threshold=overlap_threshold)
    rows = []
    for factor_idx, factor_name in enumerate(raw_weights.index):
        cpu_style_label = str(overlap_df.index[factor_idx])
        matched_pathway = cpu_style_label.removeprefix("all_") if cpu_style_label.startswith("all_") else None
        rows.append(
            {
                "factor_name": factor_name,
                "spectra_cpu_style_label": cpu_style_label,
                "matched_pathway_name": matched_pathway,
                "max_overlap_coefficient": float(overlap_df.iloc[factor_idx].max()),
                "passes_overlap_threshold": bool(matched_pathway is not None),
            }
        )
    return pd.DataFrame.from_records(rows)


def align_weights_by_spectra_overlap_labels(
    weights: pd.DataFrame,
    prior_mask: pd.DataFrame,
    overlap_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign Spectra factors with its CPU-style top-marker overlap labels."""
    zero = np.zeros(weights.shape[1], dtype=float)
    aligned_rows: list[np.ndarray] = []
    assignment_rows: list[dict[str, Any]] = []

    for pathway_name in prior_mask.index:
        candidates = overlap_labels.loc[
            (overlap_labels["passes_overlap_threshold"])
            & (overlap_labels["matched_pathway_name"].astype(str) == str(pathway_name))
        ].copy()

        if candidates.empty:
            aligned_rows.append(zero.copy())
            assignment_rows.append(
                {
                    "pathway_name": pathway_name,
                    "factor_name": None,
                    "assignment_strategy": "spectra_overlap",
                    "assignment_overlap_coefficient": np.nan,
                    "n_candidate_factors": 0,
                    "missing_overlap_label": True,
                }
            )
            continue

        candidates = candidates.sort_values(["max_overlap_coefficient", "factor_name"], ascending=[False, True])
        selected = candidates.iloc[0]
        factor_name = str(selected["factor_name"])
        aligned_rows.append(weights.loc[factor_name].to_numpy(dtype=float))
        assignment_rows.append(
            {
                "pathway_name": pathway_name,
                "factor_name": factor_name,
                "assignment_strategy": "spectra_overlap",
                "assignment_overlap_coefficient": float(selected["max_overlap_coefficient"]),
                "n_candidate_factors": int(len(candidates)),
                "missing_overlap_label": False,
            }
        )

    aligned = pd.DataFrame(aligned_rows, index=prior_mask.index, columns=weights.columns)
    assignment = pd.DataFrame.from_records(assignment_rows)
    return aligned, assignment


def model_reconstruction(method: str, model, *, terms: list[str]) -> np.ndarray:
    if method == "expimap":
        return np.asarray(model.get_y(), dtype=float)
    if method == "spectra":
        weights = np.asarray(model.return_factors(), dtype=float)
        factors = np.asarray(model.return_cell_scores(), dtype=float)
        gene_scalings = _spectra_gene_scalings_array(model)
        return np.asarray(factors @ (weights * gene_scalings), dtype=float)
    raise ValueError(f"Unsupported method: {method}")


def summarize_baseline_refinement(
    *,
    weights: pd.DataFrame,
    true_mask: pd.DataFrame,
    noisy_mask: pd.DataFrame,
    noise_stats: pd.DataFrame,
) -> pd.DataFrame:
    common = [name for name in noisy_mask.index if name in true_mask.index and name in weights.index]
    noise_stats = noise_stats.set_index("pathway_name")

    records: list[dict[str, Any]] = []
    for pathway_name in common:
        scores = weights.loc[pathway_name].abs().reindex(true_mask.columns).to_numpy(dtype=float)
        true_binary = true_mask.loc[pathway_name].to_numpy(dtype=bool)
        noisy_binary = noisy_mask.loc[pathway_name].to_numpy(dtype=bool)

        from sklearn.metrics import average_precision_score

        ap_true = float(average_precision_score(true_binary, scores))
        ap_noisy = float(average_precision_score(noisy_binary, scores))

        precision_true, recall_true, f1_true, threshold_true = best_f1_threshold(true_binary, scores)
        precision_noisy, recall_noisy, f1_noisy, threshold_noisy = best_f1_threshold(noisy_binary, scores)

        k_true = int(true_binary.sum())
        prec_top_true, rec_top_true, jac_top_true = top_k_overlap_metrics(true_binary, scores, k_true)
        prec_top_noisy, rec_top_noisy, jac_top_noisy = top_k_overlap_metrics(noisy_binary, scores, k_true)

        false_positive_mask = noisy_binary & ~true_binary
        false_negative_mask = true_binary & ~noisy_binary
        inactive_mask = ~true_binary

        records.append(
            {
                "pathway_name": pathway_name,
                "source": infer_pathway_source(pathway_name),
                "n_true_active": int(true_binary.sum()),
                "n_noisy_active": int(noisy_binary.sum()),
                "n_false_positives_added": int(noise_stats.loc[pathway_name, "n_false_positives_added"]),
                "n_false_negatives_removed": int(noise_stats.loc[pathway_name, "n_false_negatives_removed"]),
                "average_precision_true": ap_true,
                "average_precision_noisy": ap_noisy,
                "average_precision_gain_true_minus_noisy": ap_true - ap_noisy,
                "best_f1_true": f1_true,
                "best_f1_noisy": f1_noisy,
                "best_f1_gain_true_minus_noisy": f1_true - f1_noisy,
                "best_threshold_true": threshold_true,
                "best_threshold_noisy": threshold_noisy,
                "precision_true": precision_true,
                "recall_true": recall_true,
                "precision_noisy": precision_noisy,
                "recall_noisy": recall_noisy,
                "top_true_size_precision_true": prec_top_true,
                "top_true_size_recall_true": rec_top_true,
                "top_true_size_jaccard_true": jac_top_true,
                "top_true_size_precision_noisy": prec_top_noisy,
                "top_true_size_recall_noisy": rec_top_noisy,
                "top_true_size_jaccard_noisy": jac_top_noisy,
                "mean_abs_weight_true": float(scores[true_binary].mean()) if true_binary.any() else np.nan,
                "mean_abs_weight_noisy": float(scores[noisy_binary].mean()) if noisy_binary.any() else np.nan,
                "mean_abs_weight_false_positives": float(scores[false_positive_mask].mean())
                if false_positive_mask.any()
                else np.nan,
                "mean_abs_weight_false_negatives": float(scores[false_negative_mask].mean())
                if false_negative_mask.any()
                else np.nan,
                "mean_abs_weight_inactive": float(scores[inactive_mask].mean()) if inactive_mask.any() else np.nan,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("average_precision_true", ascending=False).reset_index(drop=True)


def save_baseline_model(method: str, model, run_dir: Path) -> None:
    if method == "expimap":
        try:
            model.save(run_dir / "expimap_model", overwrite=True)
        except Exception as exc:
            (run_dir / "model_save_warning.txt").write_text(str(exc))
    elif method == "spectra":
        try:
            import torch

            torch.save(model, run_dir / "spectra_model.pt")
        except Exception as exc:
            (run_dir / "model_save_warning.txt").write_text(str(exc))


def run_one(
    *,
    method: str,
    data: np.ndarray,
    var_names: pd.Index,
    true_mask: pd.DataFrame,
    true_gene_set_stats: pd.DataFrame,
    run_row: pd.Series,
    config: Sln208BaselineConfig,
) -> dict[str, Any]:
    fpr = float(run_row["fpr"])
    fnr = float(run_row["fnr"])
    seed = int(run_row["seed"])
    noise_seed = int(run_row["noise_seed"])
    noise_level = float(run_row["noise_level"])
    run_dir = config.out_dir / method / f"noise_{_slugify_noise_level(noise_level)}" / f"seed_{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "summary.json"
    if config.skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text())

    noisy_mask, noise_stats = perturb_mask(true_mask, fpr=fpr, fnr=fnr, seed=noise_seed)
    noisy_gene_set_stats = noisy_gene_set_stats_from_mask(true_gene_set_stats, noisy_mask)
    noisy_mask.to_csv(run_dir / "noisy_mask.csv")
    noise_stats.to_csv(run_dir / "noise_stats.csv", index=False)
    noisy_gene_set_stats.to_csv(run_dir / "noisy_gene_set_stats.csv", index=False)

    model_data, input_preprocessing = preprocess_baseline_input(method, data)
    model = train_baseline_model(
        method=method,
        data=model_data,
        noisy_mask=noisy_mask,
        feature_names=var_names,
        seed=seed,
        config=config,
    )
    if config.save_model:
        save_baseline_model(method, model, run_dir)

    weights = model_weights(method, model, terms=noisy_mask.index.tolist(), var_names=var_names)
    if method == "expimap":
        weights.to_csv(run_dir / "weights_raw.csv")
        weights, assignment_df = align_weights_to_prior_mask(weights, noisy_mask)
        assignment_df.to_csv(run_dir / "expimap_aupr_factor_assignment.csv", index=False)
    elif method == "spectra":
        weights.to_csv(run_dir / "weights_raw.csv")
        overlap_labels = spectra_overlap_label_table(
            weights,
            noisy_mask,
            n_top_vals=config.spectra_overlap_top_n,
            overlap_threshold=config.spectra_overlap_threshold,
        )
        overlap_labels.to_csv(run_dir / "spectra_overlap_labels.csv", index=False)
        if config.spectra_assignment == "overlap":
            weights, assignment_df = align_weights_by_spectra_overlap_labels(weights, noisy_mask, overlap_labels)
        else:
            weights, assignment_df = align_weights_to_prior_mask(weights, noisy_mask)
            assignment_df["assignment_strategy"] = "aupr_hungarian"
        assignment_df.to_csv(run_dir / "factor_assignment.csv", index=False)
    weights.to_csv(run_dir / "weights.csv")
    y_pred = model_reconstruction(method, model, terms=noisy_mask.index.tolist())
    ss_res = float(np.nansum(np.square(model_data - y_pred)))
    ss_tot = float(np.nansum(np.square(model_data)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.nanmean(np.square(model_data - y_pred))))

    refinement_df = summarize_baseline_refinement(
        weights=weights,
        true_mask=true_mask,
        noisy_mask=noisy_mask,
        noise_stats=noise_stats,
    )
    refinement_df.to_csv(run_dir / "pathway_refinement.csv", index=False)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_refinement_scatter(refinement_df, path=plots_dir / "refinement_ap_scatter.png")

    summary = {
        "method": method,
        "input_preprocessing": input_preprocessing,
        "data_path": str(config.data_path),
        "reference_sweep_dir": str(config.reference_sweep_dir),
        "out_dir": str(run_dir),
        "fpr": fpr,
        "fnr": fnr,
        "seed": seed,
        "noise_seed": noise_seed,
        "noise_level": noise_level,
        "n_uninformed_factors": 0,
        "n_true_gene_sets": int(len(true_gene_set_stats)),
        "n_hallmark_gene_sets": int((true_gene_set_stats["source"] == "hallmark").sum()),
        "n_mca_gene_sets": int((true_gene_set_stats["source"] == "mca").sum()),
        "rna_r2": r2,
        "rna_rmse": rmse,
        **summarize_refinement_table(refinement_df),
    }
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    (run_dir / "resolved_run.json").write_text(json.dumps({**asdict(config), **summary}, indent=2, sort_keys=True, default=str))
    return summary


def method_summary_outputs(method_dir: Path, *, focus_source: str | None) -> None:
    plots_dir = method_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_per_run = pd.read_csv(method_dir / "summary_per_run.csv")
    summary_df = (
        summary_per_run.groupby("noise_level")[[
            "mean_average_precision_true",
            "mean_average_precision_noisy",
            "fraction_pathways_ap_true_gt_noisy",
            "mean_top_true_size_jaccard_true",
            "mean_ap_gain_true_minus_noisy",
            "mean_best_f1_gain_true_minus_noisy",
            "mean_top_true_size_jaccard_noisy",
        ]]
        .mean()
        .reset_index()
        .sort_values("noise_level")
        .reset_index(drop=True)
    )
    summary_df.to_csv(method_dir / "summary_across_noise_levels.csv", index=False)
    save_summary_plot(summary_df, path=plots_dir / "summary_across_noise_levels.png")
    errorbar_df = save_summary_errorbar_plot(summary_per_run, path=plots_dir / "summary_across_noise_levels_errorbars.png")
    errorbar_df.to_csv(method_dir / "summary_across_noise_levels_with_errorbars.csv", index=False)

    source_df = summarize_by_source(method_dir, summary_df)
    source_per_run_df = summarize_by_source_per_run(method_dir)
    if focus_source is not None and not source_df.empty:
        save_focus_source_outputs(source_df, focus_source=focus_source, out_dir=method_dir, plots_dir=plots_dir)
    if focus_source is not None and not source_per_run_df.empty:
        save_focus_source_errorbar_outputs(
            source_per_run_df,
            focus_source=focus_source,
            out_dir=method_dir,
            plots_dir=plots_dir,
        )


def run_sweep(config: Sln208BaselineConfig) -> pd.DataFrame:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "resolved_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True, default=str))
    grid = reference_grid(config.reference_sweep_dir)
    if config.noise_levels is not None:
        selected_noise = {round(float(value), 10) for value in config.noise_levels}
        grid = grid[grid["noise_level"].map(lambda value: round(float(value), 10)).isin(selected_noise)]
    if config.seeds is not None:
        selected_seeds = {int(seed) for seed in config.seeds}
        grid = grid[grid["seed"].astype(int).isin(selected_seeds)]
    grid = grid.reset_index(drop=True)
    if grid.empty:
        raise ValueError("No reference runs remain after applying --noise-levels/--seeds filters.")

    mdata = load_sln208_mdata(config.data_path)
    rna = mdata.mod["rna"]
    data = _dense_array(rna.X).astype(np.float32, copy=False)
    var_names = pd.Index(rna.var_names)
    true_gene_set_stats = build_paper_hallmark_mca_gene_set_stats(var_names)
    true_mask = bool_mask_from_gene_set_stats(true_gene_set_stats, var_names)
    true_gene_set_stats.to_csv(config.out_dir / "true_gene_set_stats.csv", index=False)
    true_mask.to_csv(config.out_dir / "true_mask.csv")

    rows: list[dict[str, Any]] = []
    total_runs = len(config.methods) * len(grid)
    run_idx = 0
    for method in config.methods:
        for _, run_row in grid.iterrows():
            run_idx += 1
            print(
                f"[{run_idx:03d}/{total_runs:03d}] method={method}, "
                f"noise={float(run_row['noise_level']):.2f}, seed={int(run_row['seed'])}",
                flush=True,
            )
            rows.append(
                run_one(
                    method=method,
                    data=data,
                    var_names=var_names,
                    true_mask=true_mask,
                    true_gene_set_stats=true_gene_set_stats,
                    run_row=run_row,
                    config=config,
                )
            )

        method_df = pd.DataFrame([row for row in rows if row["method"] == method]).sort_values(["noise_level", "seed"])
        method_dir = config.out_dir / method
        method_df.to_csv(method_dir / "summary_per_run.csv", index=False)
        method_summary_outputs(method_dir, focus_source=config.focus_source)

    summary = pd.DataFrame.from_records(rows).sort_values(["method", "noise_level", "seed"]).reset_index(drop=True)
    summary.to_csv(config.out_dir / "summary_per_run.csv", index=False)
    return summary


def main() -> None:
    config = config_from_args(build_parser().parse_args())
    run_sweep(config)


if __name__ == "__main__":
    main()
