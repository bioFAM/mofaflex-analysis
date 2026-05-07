from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.mca_filter_sensitivity import (
        DEFAULT_DATA_PATH,
        MCA_PREFIX,
        PruningStep,
        RunArtifact,
        build_pairwise_jaccard_matrix,
        build_pairwise_spearman_matrix,
        canonicalize_mca_names,
        save_gene_set_count_plot,
        save_mca_recall_plot,
        save_pairwise_heatmap,
        save_reference_jaccard_plot,
        save_selected_pathway_rank_heatmap,
        save_top_pathway_variance_heatmap,
        summarize_run,
        write_summary_text,
    )
    from mofaflex_sensitivity.plot_style import set_house_style
    from mofaflex_sensitivity.sln208_prior_noise_baselines import (
        align_weights_by_spectra_overlap_labels,
        align_weights_to_prior_mask,
        model_reconstruction,
        model_weights,
        preprocess_baseline_input,
        save_baseline_model,
        spectra_overlap_label_table,
    )
    from mofaflex_sensitivity.sln208_prior_noise_refinement import bool_mask_from_gene_set_stats
    from mofaflex_sensitivity.synthetic_uninformed_dense_baselines import (
        _spectra_gene_scalings_array,
        train_expimap,
        train_spectra,
    )
    from mofaflex_sensitivity.mca_filter_sensitivity import load_sln208_mdata
else:
    from .mca_filter_sensitivity import (
        DEFAULT_DATA_PATH,
        MCA_PREFIX,
        PruningStep,
        RunArtifact,
        build_pairwise_jaccard_matrix,
        build_pairwise_spearman_matrix,
        canonicalize_mca_names,
        load_sln208_mdata,
        save_gene_set_count_plot,
        save_mca_recall_plot,
        save_pairwise_heatmap,
        save_reference_jaccard_plot,
        save_selected_pathway_rank_heatmap,
        save_top_pathway_variance_heatmap,
        summarize_run,
        write_summary_text,
    )
    from .plot_style import set_house_style
    from .sln208_prior_noise_baselines import (
        align_weights_by_spectra_overlap_labels,
        align_weights_to_prior_mask,
        model_reconstruction,
        model_weights,
        preprocess_baseline_input,
        save_baseline_model,
        spectra_overlap_label_table,
    )
    from .sln208_prior_noise_refinement import bool_mask_from_gene_set_stats
    from .synthetic_uninformed_dense_baselines import _spectra_gene_scalings_array, train_expimap, train_spectra


DEFAULT_REFERENCE_PRUNING_DIR = Path("artifacts/sln208_mca_pruning_sensitivity")
DEFAULT_OUT_DIR = Path("artifacts/sln208_mca_pruning_sensitivity_baselines")
BASELINE_METHODS = ("expimap", "spectra")


@dataclass(frozen=True)
class Sln208PruningBaselineConfig:
    data_path: Path = DEFAULT_DATA_PATH
    reference_pruning_dir: Path = DEFAULT_REFERENCE_PRUNING_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    methods: tuple[str, ...] = BASELINE_METHODS
    seed: int = 42
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
    top_k_values: tuple[int, ...] = (3, 5, 10)
    selected_mca_rank_pathways: tuple[str, ...] = ()
    expected_mca_pathways: tuple[str, ...] = ()
    save_model: bool = True
    skip_existing: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train ExpiMap and Spectra RNA-only baselines on an existing SLN208 MCA-pruning "
            "sensitivity grid. ExpiMap receives per-gene centered RNA; Spectra receives the "
            "stored nonnegative RNA matrix used as the log_x-like MOFA-FLEX NMF input."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--reference-pruning-dir", type=Path, default=DEFAULT_REFERENCE_PRUNING_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", nargs="+", default=list(BASELINE_METHODS), choices=list(BASELINE_METHODS))
    parser.add_argument("--seed", type=int, default=42)
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
            "How to align Spectra factors to retained priors. 'overlap' uses Spectra's CPU-style "
            "top-marker overlap labels; 'aupr' uses one-to-one average-precision matching."
        ),
    )
    parser.add_argument("--spectra-overlap-top-n", type=int, default=50)
    parser.add_argument("--spectra-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--selected-mca-rank-pathways", nargs="*", default=[])
    parser.add_argument("--expected-mca-pathways", nargs="*", default=[])
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def config_from_args(args: argparse.Namespace) -> Sln208PruningBaselineConfig:
    return Sln208PruningBaselineConfig(
        data_path=args.data_path,
        reference_pruning_dir=args.reference_pruning_dir,
        out_dir=args.out_dir,
        methods=tuple(args.methods),
        seed=args.seed,
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
        top_k_values=tuple(sorted(set([10, *args.top_k_values]))),
        selected_mca_rank_pathways=tuple(args.selected_mca_rank_pathways),
        expected_mca_pathways=tuple(args.expected_mca_pathways),
        save_model=args.save_model,
        skip_existing=args.skip_existing,
    )


def _dense_array(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _parse_features(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(feature) for feature in value]
    if isinstance(value, tuple):
        return [str(feature) for feature in value]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [feature for feature in re.split(r"[,\s]+", text) if feature]
    return [str(feature) for feature in parsed]


def load_retained_gene_sets(path: Path) -> pd.DataFrame:
    gene_sets = pd.read_csv(path)
    required = {"pathway_name", "source", "set_size", "features"}
    missing = required - set(gene_sets.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    gene_sets = gene_sets.copy()
    gene_sets["features"] = gene_sets["features"].map(_parse_features)
    gene_sets["set_size"] = gene_sets["features"].map(len)
    return gene_sets


def load_reference_steps(reference_pruning_dir: Path) -> list[tuple[PruningStep, pd.DataFrame]]:
    summary_path = reference_pruning_dir / "summary.csv"
    gene_sets_dir = reference_pruning_dir / "gene_sets"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing reference pruning summary: {summary_path}")
    if not gene_sets_dir.exists():
        raise FileNotFoundError(f"Missing retained gene-set directory: {gene_sets_dir}")

    summary = pd.read_csv(summary_path).sort_values("step_index")
    steps: list[tuple[PruningStep, pd.DataFrame]] = []
    for row in summary.itertuples(index=False):
        step_label = str(row.step_label)
        gene_set_path = gene_sets_dir / f"{step_label}_retained_gene_sets.csv"
        if not gene_set_path.exists():
            raise FileNotFoundError(f"Missing retained gene-set table for {step_label}: {gene_set_path}")
        pruning_direction = str(getattr(row, "pruning_direction", "smallest"))
        n_removed_smallest = int(getattr(row, "n_removed_smallest", 0))
        n_removed_largest = int(getattr(row, "n_removed_largest", 0))
        step = PruningStep(
            step_index=int(row.step_index),
            n_removed_smallest=n_removed_smallest,
            n_removed_largest=n_removed_largest,
            n_gene_sets=int(row.n_gene_sets),
            pruning_direction=pruning_direction,
        )
        steps.append((step, load_retained_gene_sets(gene_set_path)))
    return steps


def train_baseline_model(
    *,
    method: str,
    data: np.ndarray,
    prior_mask: pd.DataFrame,
    feature_names: pd.Index,
    seed: int,
    config: Sln208PruningBaselineConfig,
):
    if method == "expimap":
        return train_expimap(
            data,
            prior_mask,
            terms=prior_mask.index.tolist(),
            seed=seed,
            config=config,
        )
    if method == "spectra":
        np.random.seed(seed)
        return train_spectra(
            data,
            prior_mask,
            terms=prior_mask.index.tolist(),
            feature_names=feature_names,
            seed=seed,
            config=config,
        )
    raise ValueError(f"Unsupported method: {method}")


def _aligned_spectra_factor_frames(
    *,
    raw_weights: pd.DataFrame,
    model,
    prior_mask: pd.DataFrame,
    config: Sln208PruningBaselineConfig,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_weights.to_csv(run_dir / "weights_raw.csv")
    overlap_labels = spectra_overlap_label_table(
        raw_weights,
        prior_mask,
        n_top_vals=config.spectra_overlap_top_n,
        overlap_threshold=config.spectra_overlap_threshold,
    )
    overlap_labels.to_csv(run_dir / "spectra_overlap_labels.csv", index=False)

    if config.spectra_assignment == "overlap":
        aligned_weights, assignment = align_weights_by_spectra_overlap_labels(
            raw_weights,
            prior_mask,
            overlap_labels,
        )
    else:
        aligned_weights, assignment = align_weights_to_prior_mask(raw_weights, prior_mask)
        assignment["assignment_strategy"] = "aupr_hungarian"
    assignment.to_csv(run_dir / "factor_assignment.csv", index=False)

    raw_factors = pd.DataFrame(
        np.asarray(model.return_cell_scores(), dtype=float),
        columns=raw_weights.index,
    )
    aligned_factor_columns = []
    for row in assignment.itertuples(index=False):
        factor_name = getattr(row, "factor_name")
        if factor_name is None or pd.isna(factor_name) or str(factor_name) not in raw_factors.columns:
            aligned_factor_columns.append(np.zeros(raw_factors.shape[0], dtype=float))
        else:
            aligned_factor_columns.append(raw_factors.loc[:, str(factor_name)].to_numpy(dtype=float))
    aligned_factors = pd.DataFrame(np.column_stack(aligned_factor_columns), columns=prior_mask.index)

    gene_scalings = _spectra_gene_scalings_array(model)
    effective_raw_weights = raw_weights.mul(gene_scalings, axis=1)
    effective_aligned_rows = []
    for row in assignment.itertuples(index=False):
        factor_name = getattr(row, "factor_name")
        if factor_name is None or pd.isna(factor_name) or str(factor_name) not in effective_raw_weights.index:
            effective_aligned_rows.append(np.zeros(effective_raw_weights.shape[1], dtype=float))
        else:
            effective_aligned_rows.append(effective_raw_weights.loc[str(factor_name)].to_numpy(dtype=float))
    effective_aligned_weights = pd.DataFrame(
        effective_aligned_rows,
        index=prior_mask.index,
        columns=prior_mask.columns,
    )
    return aligned_weights, aligned_factors, effective_aligned_weights, assignment


def aligned_factor_frames(
    *,
    method: str,
    model,
    prior_mask: pd.DataFrame,
    var_names: pd.Index,
    config: Sln208PruningBaselineConfig,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terms = prior_mask.index.tolist()
    weights = model_weights(method, model, terms=terms, var_names=var_names)
    if method == "expimap":
        weights.to_csv(run_dir / "weights_raw.csv")
        aligned_weights, assignment = align_weights_to_prior_mask(weights, prior_mask)
        assignment.to_csv(run_dir / "expimap_aupr_factor_assignment.csv", index=False)
        raw_factors = pd.DataFrame(np.asarray(model.get_latent(), dtype=float), columns=terms)
        aligned_factor_columns = []
        for row in assignment.itertuples(index=False):
            factor_name = str(getattr(row, "factor_name"))
            aligned_factor_columns.append(raw_factors.loc[:, factor_name].to_numpy(dtype=float))
        aligned_factors = pd.DataFrame(np.column_stack(aligned_factor_columns), columns=prior_mask.index)
        return aligned_weights, aligned_factors, aligned_weights.copy()
    if method == "spectra":
        aligned_weights, aligned_factors, effective_aligned_weights, _assignment = _aligned_spectra_factor_frames(
            raw_weights=weights,
            model=model,
            prior_mask=prior_mask,
            config=config,
            run_dir=run_dir,
        )
        return aligned_weights, aligned_factors, effective_aligned_weights
    raise ValueError(f"Unsupported method: {method}")


def per_factor_r2_table(data: np.ndarray, factors: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    ss_tot = float(np.nansum(np.square(data)))
    rows: list[dict[str, Any]] = []
    for factor_name in weights.index:
        if factor_name not in factors.columns:
            rows.append({"pathway_name": factor_name, "variance_explained": np.nan})
            continue
        y_pred = np.outer(
            factors[factor_name].to_numpy(dtype=float),
            weights.loc[factor_name].to_numpy(dtype=float),
        )
        ss_res = float(np.nansum(np.square(data - y_pred)))
        rows.append(
            {
                "pathway_name": factor_name,
                "variance_explained": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def build_ranking(
    *,
    data: np.ndarray,
    factors: pd.DataFrame,
    effective_weights: pd.DataFrame,
    gene_set_stats: pd.DataFrame,
) -> pd.DataFrame:
    ranking = per_factor_r2_table(data, factors, effective_weights)
    ranking = ranking.merge(gene_set_stats.drop(columns="features", errors="ignore"), on="pathway_name", how="left")
    missing_source = ranking["source"].isna()
    ranking.loc[missing_source, "source"] = np.where(
        ranking.loc[missing_source, "pathway_name"].str.startswith(MCA_PREFIX),
        "mca",
        "hallmark",
    )
    ranking["is_mca"] = ranking["source"].eq("mca")
    ranking["assigned_pathway"] = ranking["pathway_name"]
    ranking = ranking.sort_values("variance_explained", ascending=False, kind="stable").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def write_method_outputs(
    *,
    method_dir: Path,
    run_artifacts: list[RunArtifact],
    summary_df: pd.DataFrame,
    top_k_values: list[int],
    selected_mca_rank_pathways: list[str],
    primary_source_name: str,
) -> None:
    plots_dir = method_dir / "plots"
    rankings_dir = method_dir / "rankings"
    gene_sets_dir = method_dir / "gene_sets"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir.mkdir(parents=True, exist_ok=True)
    gene_sets_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(method_dir / "summary.csv", index=False)
    pairwise_spearman = build_pairwise_spearman_matrix(run_artifacts)
    pairwise_spearman.to_csv(method_dir / "pairwise_spearman.csv")
    pairwise_jaccard = {k: build_pairwise_jaccard_matrix(run_artifacts, k=k) for k in top_k_values}

    for artifact in run_artifacts:
        artifact.ranking.to_csv(
            rankings_dir / f"{artifact.step.step_label}_pathway_ranks.csv",
            index=False,
        )
        gene_set_export = artifact.gene_set_stats.copy()
        gene_set_export["features"] = gene_set_export["features"].map(json.dumps)
        gene_set_export.to_csv(
            gene_sets_dir / f"{artifact.step.step_label}_retained_gene_sets.csv",
            index=False,
        )

    save_pairwise_heatmap(
        pairwise_spearman,
        title="Pairwise pathway-rank Spearman correlation",
        path=plots_dir / "pairwise_spearman.png",
    )
    for k, matrix in pairwise_jaccard.items():
        matrix.to_csv(method_dir / f"pairwise_jaccard_top{k}.csv")
        save_pairwise_heatmap(
            matrix,
            title=f"Top-{k} pathway Jaccard overlap",
            path=plots_dir / f"pairwise_jaccard_top{k}.png",
        )
    save_gene_set_count_plot(summary_df, path=plots_dir / "n_gene_sets_vs_n_mca_in_top10.png")
    save_mca_recall_plot(summary_df, path=plots_dir / "mca_recall_topk.png")
    save_reference_jaccard_plot(summary_df, k=10, path=plots_dir / "jaccard_top10_vs_full.png")
    save_selected_pathway_rank_heatmap(
        summary_df,
        run_artifacts,
        selected_pathways=selected_mca_rank_pathways,
        path=plots_dir / "selected_mca_pathway_ranks.png",
    )
    save_top_pathway_variance_heatmap(
        summary_df,
        run_artifacts,
        top_k=10,
        path=plots_dir / "top10_pathway_variance_heatmap.png",
    )
    write_summary_text(summary_df, path=method_dir / "robustness_summary.txt", primary_source_name=primary_source_name)


def run_one_step(
    *,
    method: str,
    data: np.ndarray,
    var_names: pd.Index,
    step: PruningStep,
    gene_set_stats: pd.DataFrame,
    config: Sln208PruningBaselineConfig,
) -> tuple[RunArtifact, dict[str, Any]]:
    method_dir = config.out_dir / method
    run_dir = method_dir / "runs" / step.step_label
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    method_dir = config.out_dir / method
    ranking_path = method_dir / "rankings" / f"{step.step_label}_pathway_ranks.csv"
    if config.skip_existing and metrics_path.exists() and ranking_path.exists():
        metrics = json.loads(metrics_path.read_text())
        ranking = pd.read_csv(ranking_path)
        return (
            RunArtifact(
                step=step,
                n_uninformed_factors=0,
                gene_set_stats=gene_set_stats,
                ranking=ranking,
            ),
            metrics,
        )

    prior_mask = bool_mask_from_gene_set_stats(gene_set_stats, var_names)
    prior_mask.to_csv(run_dir / "prior_mask.csv")
    gene_set_export = gene_set_stats.copy()
    gene_set_export["features"] = gene_set_export["features"].map(json.dumps)
    gene_set_export.to_csv(run_dir / "retained_gene_sets.csv", index=False)

    model_data, input_preprocessing = preprocess_baseline_input(method, data)
    model = train_baseline_model(
        method=method,
        data=model_data,
        prior_mask=prior_mask,
        feature_names=var_names,
        seed=config.seed,
        config=config,
    )
    if config.save_model:
        save_baseline_model(method, model, run_dir)

    weights, factors, effective_weights = aligned_factor_frames(
        method=method,
        model=model,
        prior_mask=prior_mask,
        var_names=var_names,
        config=config,
        run_dir=run_dir,
    )
    weights.to_csv(run_dir / "weights.csv")
    effective_weights.to_csv(run_dir / "weights_effective_for_reconstruction.csv")
    factors.to_csv(run_dir / "factors.csv", index=False)

    y_pred = model_reconstruction(method, model, terms=prior_mask.index.tolist())
    ss_tot = float(np.nansum(np.square(model_data)))
    ss_res = float(np.nansum(np.square(model_data - y_pred)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.nanmean(np.square(model_data - y_pred))))

    ranking = build_ranking(
        data=model_data,
        factors=factors,
        effective_weights=effective_weights,
        gene_set_stats=gene_set_stats,
    )
    ranking.to_csv(run_dir / "pathway_ranks.csv", index=False)

    primary_sources = gene_set_stats.loc[gene_set_stats["source"] != "mca", "source"].unique().tolist()
    primary_source_name = primary_sources[0] if primary_sources else "primary"
    metrics = {
        "method": method,
        "input_preprocessing": input_preprocessing,
        "data_path": str(config.data_path),
        "reference_pruning_dir": str(config.reference_pruning_dir),
        "out_dir": str(run_dir),
        "seed": int(config.seed),
        "step_index": int(step.step_index),
        "step_label": step.step_label,
        "pruning_direction": step.pruning_direction,
        "n_removed_smallest": int(step.n_removed_smallest),
        "n_removed_largest": int(step.n_removed_largest),
        "n_gene_sets": int(len(gene_set_stats)),
        "n_primary_gene_sets": int((gene_set_stats["source"] == primary_source_name).sum()),
        "n_mca_gene_sets": int((gene_set_stats["source"] == "mca").sum()),
        "rna_r2": r2,
        "rna_rmse": rmse,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (run_dir / "resolved_run.json").write_text(
        json.dumps({**asdict(config), **metrics}, indent=2, sort_keys=True, default=str)
    )
    return (
        RunArtifact(
            step=step,
            n_uninformed_factors=0,
            gene_set_stats=gene_set_stats,
            ranking=ranking,
        ),
        metrics,
    )


def run_sweep(config: Sln208PruningBaselineConfig) -> pd.DataFrame:
    set_house_style()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "resolved_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True, default=str))

    reference_steps = load_reference_steps(config.reference_pruning_dir)
    if not reference_steps:
        raise ValueError(f"No pruning steps found in {config.reference_pruning_dir}")

    mdata = load_sln208_mdata(config.data_path)
    rna = mdata.mod["rna"]
    data = _dense_array(rna.X).astype(np.float32, copy=False)
    var_names = pd.Index(rna.var_names)

    rows: list[dict[str, Any]] = []
    total_runs = len(config.methods) * len(reference_steps)
    run_idx = 0
    for method in config.methods:
        method_artifacts: list[RunArtifact] = []
        method_rows: list[dict[str, Any]] = []
        for step, gene_set_stats in reference_steps:
            run_idx += 1
            print(
                f"[{run_idx:03d}/{total_runs:03d}] method={method}, {step.step_label}, "
                f"n_gene_sets={step.n_gene_sets}",
                flush=True,
            )
            artifact, metrics = run_one_step(
                method=method,
                data=data,
                var_names=var_names,
                step=step,
                gene_set_stats=gene_set_stats,
                config=config,
            )
            method_artifacts.append(artifact)
            method_rows.append(metrics)
            rows.append(metrics)

        reference_ranking = method_artifacts[0].ranking
        expected_mca_pathways = canonicalize_mca_names(list(config.expected_mca_pathways))
        summary_rows = [
            {
                **summarize_run(
                    artifact,
                    reference_ranking=reference_ranking,
                    expected_mca_pathways=expected_mca_pathways,
                    top_k_values=list(config.top_k_values),
                ),
                **{
                    "method": method,
                    "seed": int(config.seed),
                    "input_preprocessing": method_rows[idx]["input_preprocessing"],
                    "rna_r2": method_rows[idx]["rna_r2"],
                    "rna_rmse": method_rows[idx]["rna_rmse"],
                },
            }
            for idx, artifact in enumerate(method_artifacts)
        ]
        summary_df = pd.DataFrame(summary_rows).sort_values("n_gene_sets", ascending=False).reset_index(drop=True)
        primary_sources = reference_steps[0][1].loc[reference_steps[0][1]["source"] != "mca", "source"].unique().tolist()
        primary_source_name = primary_sources[0] if primary_sources else "primary"
        write_method_outputs(
            method_dir=config.out_dir / method,
            run_artifacts=method_artifacts,
            summary_df=summary_df,
            top_k_values=list(config.top_k_values),
            selected_mca_rank_pathways=list(config.selected_mca_rank_pathways),
            primary_source_name=primary_source_name,
        )
        pd.DataFrame(method_rows).sort_values("step_index").to_csv(
            config.out_dir / method / "reconstruction_summary.csv",
            index=False,
        )

    summary = pd.DataFrame.from_records(rows).sort_values(["method", "step_index"]).reset_index(drop=True)
    summary.to_csv(config.out_dir / "reconstruction_summary.csv", index=False)
    return summary


def main() -> None:
    config = config_from_args(build_parser().parse_args())
    summary = run_sweep(config)
    print(f"Finished SLN208 MCA pruning baseline sweep in {config.out_dir}")
    print(f"n_runs={len(summary)}")


if __name__ == "__main__":
    main()
