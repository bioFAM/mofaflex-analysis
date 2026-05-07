from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.mca_filter_sensitivity import load_sln208_mdata, prettify_pathway_name
    from mofaflex_sensitivity.plot_style import CAT_PALETTE, clean_ax, place_legend, savefig, set_house_style
    from mofaflex_sensitivity.sln208_prior_noise_baselines import align_weights_to_prior_mask
    from mofaflex_sensitivity.uninformed_sensitivity import compute_processed_subset_r2
else:
    from .mca_filter_sensitivity import load_sln208_mdata, prettify_pathway_name
    from .plot_style import CAT_PALETTE, clean_ax, place_legend, savefig, set_house_style
    from .sln208_prior_noise_baselines import align_weights_to_prior_mask
    from .uninformed_sensitivity import compute_processed_subset_r2


DEFAULT_MOFLEX_DIR = Path("artifacts/sln208_mca_pruning_sensitivity")
DEFAULT_BASELINES_DIR = Path("artifacts/sln208_mca_pruning_sensitivity_baselines")
DEFAULT_OUT_DIR = DEFAULT_BASELINES_DIR / "comparison_plots"
METHOD_LABELS = {
    "mofaflex": "MOFA-FLEX",
    "expimap": "ExpiMap",
    "spectra": "Spectra",
}
DEFAULT_EXCLUDED_MCA_PATHWAYS = {
    "MCA::HAN_MACROPHAGE",
    "MCA::HAN_NEUTROPHIL",
}
# Use raw matched program weights for gene-program recovery/AUPR plots.
# Spectra's effective reconstruction weights are still used upstream for RMSE
# and variance-based rankings, where gene scalings are part of reconstruction.
BASELINE_WEIGHTS_FILE = "weights.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MOFA-FLEX, ExpiMap, and Spectra on the SLN208 MCA-pruning sensitivity outputs."
        )
    )
    parser.add_argument("--mofaflex-dir", type=Path, default=DEFAULT_MOFLEX_DIR)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["mofaflex", "expimap", "spectra"],
        default=["mofaflex", "expimap", "spectra"],
    )
    parser.add_argument("--top-k-genes", type=int, nargs="+", default=list(range(20, 201, 10)))
    parser.add_argument(
        "--mca-pathway-order-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV with a pathway_pretty column. When provided, the MCA pathway correlation "
            "facet plot uses this order instead of recomputing a MOFA-FLEX variance order."
        ),
    )
    return parser


def _method_dir(method: str, *, mofaflex_dir: Path, baselines_dir: Path) -> Path:
    return mofaflex_dir if method == "mofaflex" else baselines_dir / method


def _step_sort_key(step_label: str) -> int:
    match = re.search(r"step_(\d+)_", step_label)
    return int(match.group(1)) if match else 10_000


def _parse_feature_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    parsed = json.loads(str(value))
    return [str(item) for item in parsed]


def _load_or_align_expimap_weights(run_dir: Path) -> pd.DataFrame:
    """Return soft-mask ExpiMap weights after one-to-one AP matching to priors."""
    aligned_path = run_dir / "weights_aupr_aligned.csv"
    assignment_path = run_dir / "expimap_aupr_factor_assignment.csv"
    if aligned_path.exists() and assignment_path.exists():
        return pd.read_csv(aligned_path, index_col=0).astype(float)

    weights_path = run_dir / "weights.csv"
    prior_mask_path = run_dir / "prior_mask.csv"
    if not weights_path.exists() or not prior_mask_path.exists():
        raise FileNotFoundError(f"Missing ExpiMap weights or prior mask in {run_dir}")

    prior_mask = pd.read_csv(prior_mask_path, index_col=0).astype(bool)
    raw_weights = pd.read_csv(weights_path, index_col=0).astype(float)
    raw_weights = raw_weights.reindex(columns=prior_mask.columns, fill_value=0.0).fillna(0.0)
    aligned, assignment = align_weights_to_prior_mask(raw_weights, prior_mask)
    aligned.to_csv(aligned_path)
    assignment.to_csv(assignment_path, index=False)
    return aligned


def _load_expimap_aupr_aligned_ranking(run_dir: Path) -> pd.DataFrame:
    """Map existing ExpiMap per-factor variance ranks onto AUPR-aligned prior names."""
    output_path = run_dir / "pathway_ranks_aupr_aligned.csv"
    if output_path.exists():
        return pd.read_csv(output_path)

    _load_or_align_expimap_weights(run_dir)
    assignment = pd.read_csv(run_dir / "expimap_aupr_factor_assignment.csv")
    raw_ranking = pd.read_csv(run_dir / "pathway_ranks.csv")
    retained = pd.read_csv(run_dir / "retained_gene_sets.csv")
    retained_lookup = retained.drop(columns="features", errors="ignore").set_index("pathway_name")
    raw_lookup = raw_ranking.set_index("pathway_name")

    rows = []
    for row in assignment.itertuples(index=False):
        pathway_name = str(row.pathway_name)
        factor_name = str(row.factor_name)
        retained_row = retained_lookup.loc[pathway_name] if pathway_name in retained_lookup.index else pd.Series(dtype=object)
        raw_row = raw_lookup.loc[factor_name] if factor_name in raw_lookup.index else pd.Series(dtype=object)
        source = retained_row.get("source", "mca" if pathway_name.startswith("MCA::") else "hallmark")
        rows.append(
            {
                "pathway_name": pathway_name,
                "variance_explained": float(raw_row.get("variance_explained", np.nan)),
                "source": str(source),
                "set_size": int(retained_row.get("set_size", raw_row.get("set_size", 0))),
                "is_mca": str(source) == "mca",
                "assigned_pathway": pathway_name,
                "matched_factor_name": factor_name,
                "assignment_average_precision_noisy": float(row.assignment_average_precision_noisy),
            }
        )
    ranking = pd.DataFrame.from_records(rows)
    ranking = ranking.sort_values("variance_explained", ascending=False, kind="stable").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(output_path, index=False)
    return ranking


def load_ranking_for_step(method: str, *, method_dir: Path, baselines_dir: Path, step_label: str) -> pd.DataFrame:
    if method == "expimap":
        run_dir = baselines_dir / method / "runs" / step_label
        return _load_expimap_aupr_aligned_ranking(run_dir)

    ranking_path = method_dir / "rankings" / f"{step_label}_pathway_ranks.csv"
    if not ranking_path.exists() and method != "mofaflex":
        ranking_path = baselines_dir / method / "runs" / step_label / "pathway_ranks.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)
    return pd.read_csv(ranking_path)


def load_method_summary(method: str, *, mofaflex_dir: Path, baselines_dir: Path) -> pd.DataFrame:
    method_dir = _method_dir(method, mofaflex_dir=mofaflex_dir, baselines_dir=baselines_dir)
    summary_path = method_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary for {method}: {summary_path}")
    summary = pd.read_csv(summary_path).copy()
    summary["method"] = METHOD_LABELS[method]
    if "n_mca_gene_sets" not in summary.columns:
        raise ValueError(f"{summary_path} does not contain n_mca_gene_sets.")
    if "mca_recall_top10" not in summary.columns:
        summary["mca_recall_top10"] = summary["n_mca_in_top10"] / summary["n_mca_gene_sets"]
    if method == "expimap":
        for idx, row in summary.iterrows():
            step_label = str(row["step_label"])
            ranking = load_ranking_for_step(method, method_dir=method_dir, baselines_dir=baselines_dir, step_label=step_label)
            top10 = ranking.sort_values("rank").head(10)
            top10_pathways = top10["pathway_name"].astype(str).tolist()
            top10_mca = top10.loc[top10["source"].eq("mca"), "pathway_name"].astype(str).tolist()
            n_mca = int(row["n_mca_gene_sets"])
            summary.at[idx, "top_10_pathways"] = json.dumps(top10_pathways)
            summary.at[idx, "top_10_mca_pathways"] = json.dumps(top10_mca)
            summary.at[idx, "n_mca_in_top10"] = len(top10_mca)
            if "n_mca_in_top_10" in summary.columns:
                summary.at[idx, "n_mca_in_top_10"] = len(top10_mca)
            summary.at[idx, "mca_recall_top10"] = len(top10_mca) / n_mca if n_mca else np.nan
            summary.at[idx, "mca_fraction_in_top10"] = len(top10_mca) / 10.0
    summary["top10_fraction_label"] = summary.apply(
        lambda row: f"{int(row['n_mca_in_top10'])}/{int(row['n_mca_gene_sets'])}",
        axis=1,
    )
    return summary.sort_values("step_index").reset_index(drop=True)


def load_prior_mask_for_step(
    *,
    method: str,
    step_label: str,
    columns: pd.Index,
    mofaflex_dir: Path,
    baselines_dir: Path,
) -> pd.DataFrame | None:
    if method == "mofaflex":
        gene_set_path = mofaflex_dir / "gene_sets" / f"{step_label}_retained_gene_sets.csv"
        if not gene_set_path.exists():
            return None
        gene_sets = pd.read_csv(gene_set_path)
        mask = pd.DataFrame(False, index=gene_sets["pathway_name"].astype(str), columns=columns)
        column_set = set(columns)
        for row in gene_sets.itertuples(index=False):
            features = [gene for gene in _parse_feature_list(getattr(row, "features")) if gene in column_set]
            if features:
                mask.loc[str(getattr(row, "pathway_name")), features] = True
        return mask

    prior_mask_path = baselines_dir / method / "runs" / step_label / "prior_mask.csv"
    if not prior_mask_path.exists():
        return None
    prior_mask = pd.read_csv(prior_mask_path, index_col=0).astype(bool)
    common = pd.Index(columns).intersection(prior_mask.columns)
    return prior_mask.loc[:, common]


def average_mca_prior_aupr(weights: pd.DataFrame, prior_mask: pd.DataFrame) -> tuple[float, int]:
    common_columns = weights.columns.intersection(prior_mask.columns)
    if common_columns.empty:
        return np.nan, 0
    mca_names = [name for name in prior_mask.index if str(name).startswith("MCA::") and name in weights.index]
    scores = []
    for pathway_name in mca_names:
        mask = prior_mask.loc[pathway_name, common_columns].to_numpy(dtype=bool)
        if not mask.any():
            continue
        loading_scores = weights.loc[pathway_name, common_columns].abs().to_numpy(dtype=float)
        scores.append(float(average_precision_score(mask, loading_scores)))
    return (float(np.mean(scores)) if scores else np.nan, len(scores))


def mofaflex_rna_rmse_by_step(mofaflex_dir: Path) -> pd.DataFrame:
    import mofaflex as mfl

    resolved_path = mofaflex_dir / "resolved_run.json"
    if not resolved_path.exists():
        return pd.DataFrame(columns=["step_label", "rna_rmse"])
    resolved = json.loads(resolved_path.read_text())
    modalities = resolved.get("modalities")
    mdata = load_sln208_mdata(Path(resolved["data_path"]), modalities=modalities)
    n_rna_entries = int(resolved["n_obs"]) * int(resolved["n_vars_rna"])

    rows = []
    for model_path in sorted((mofaflex_dir / "models").glob("step_*.h5"), key=lambda p: _step_sort_key(p.stem)):
        model = mfl.MOFAFLEX.load(model_path, map_location="cpu")
        weights = model.get_weights(return_type="pandas", ordered=False)["rna"]
        subset_metrics = compute_processed_subset_r2(
            model,
            mdata,
            factor_names=weights.index.tolist(),
        )
        ss_res = float(subset_metrics.get("ss_res_rna", np.nan))
        rows.append(
            {
                "step_label": model_path.stem,
                "rna_rmse": float(np.sqrt(ss_res / n_rna_entries)) if np.isfinite(ss_res) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def load_rmse_by_method(
    method: str,
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
    mofaflex_rmse: pd.DataFrame,
) -> pd.DataFrame:
    if method == "mofaflex":
        return mofaflex_rmse.copy()
    summary_path = baselines_dir / method / "summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(columns=["step_label", "rna_rmse"])
    summary = pd.read_csv(summary_path)
    if "rna_rmse" not in summary.columns:
        return pd.DataFrame(columns=["step_label", "rna_rmse"])
    return summary.loc[:, ["step_label", "rna_rmse"]].copy()


def build_mca_aupr_rmse_table(
    methods: list[str],
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weight_cache: dict[tuple[str, str], pd.DataFrame] = {}
    mofaflex_rmse = mofaflex_rna_rmse_by_step(mofaflex_dir)
    top_pathways = load_top_pathway_table(methods, mofaflex_dir=mofaflex_dir, baselines_dir=baselines_dir)

    for method in methods:
        rmse_by_step = load_rmse_by_method(
            method,
            mofaflex_dir=mofaflex_dir,
            baselines_dir=baselines_dir,
            mofaflex_rmse=mofaflex_rmse,
        ).set_index("step_label")
        method_steps = (
            top_pathways.loc[top_pathways["method_key"].eq(method), ["step_label", "step_index", "n_gene_sets"]]
            .drop_duplicates()
            .sort_values("step_index")
        )
        for row in method_steps.itertuples(index=False):
            weights = load_weight_matrix(
                method=method,
                step_label=str(row.step_label),
                mofaflex_dir=mofaflex_dir,
                baselines_dir=baselines_dir,
                cache=weight_cache,
            )
            if weights is None:
                continue
            prior_mask = load_prior_mask_for_step(
                method=method,
                step_label=str(row.step_label),
                columns=weights.columns,
                mofaflex_dir=mofaflex_dir,
                baselines_dir=baselines_dir,
            )
            if prior_mask is None:
                continue
            mean_aupr, n_mca = average_mca_prior_aupr(weights, prior_mask)
            rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "method_key": method,
                    "step_label": str(row.step_label),
                    "step_index": int(row.step_index),
                    "n_gene_sets": int(row.n_gene_sets),
                    "mean_mca_prior_aupr": mean_aupr,
                    "n_mca_gene_sets": int(n_mca),
                    "rna_rmse": float(rmse_by_step.loc[str(row.step_label), "rna_rmse"])
                    if str(row.step_label) in rmse_by_step.index
                    else np.nan,
                }
            )
    return pd.DataFrame.from_records(rows)


def save_mca_aupr_vs_rmse_plot(summary: pd.DataFrame, *, path: Path) -> None:
    set_house_style()
    plot_df = summary.dropna(subset=["mean_mca_prior_aupr", "rna_rmse"]).copy()
    if plot_df.empty:
        return
    method_order = [label for label in METHOD_LABELS.values() if label in set(plot_df["method"])]
    palette = dict(zip(method_order, sns.color_palette(CAT_PALETTE, n_colors=len(method_order)), strict=False))

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    sns.lineplot(
        data=plot_df,
        x="rna_rmse",
        y="mean_mca_prior_aupr",
        hue="method",
        hue_order=method_order,
        marker="o",
        linewidth=1.6,
        palette=palette,
        ax=ax,
        sort=False,
    )
    for row in plot_df.itertuples(index=False):
        ax.text(
            float(row.rna_rmse),
            float(row.mean_mca_prior_aupr),
            str(int(row.n_gene_sets)),
            fontsize=6.5,
            color=palette.get(str(row.method), "0.25"),
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("RNA RMSE")
    ax.set_ylabel("Mean AUPR vs MCA prior annotations")
    clean_ax(ax)
    place_legend(ax, mode="outside", title=None)
    savefig(fig, path)
    plt.close(fig)


def save_mca_aupr_vs_rmse_mean_sd_plot(summary: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    set_house_style()
    plot_df = summary.dropna(subset=["mean_mca_prior_aupr", "rna_rmse"]).copy()
    if plot_df.empty:
        return pd.DataFrame()

    aggregated = (
        plot_df.groupby("method", sort=False)
        .agg(
            mean_mca_prior_aupr=("mean_mca_prior_aupr", "mean"),
            sd_mca_prior_aupr=("mean_mca_prior_aupr", "std"),
            mean_rna_rmse=("rna_rmse", "mean"),
            sd_rna_rmse=("rna_rmse", "std"),
            n_steps=("step_label", "nunique"),
        )
        .reset_index()
    )

    method_order = [label for label in METHOD_LABELS.values() if label in set(aggregated["method"])]
    palette = dict(zip(method_order, sns.color_palette(CAT_PALETTE, n_colors=len(method_order)), strict=False))

    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    for row in aggregated.itertuples(index=False):
        color = palette.get(str(row.method), "0.25")
        ax.errorbar(
            float(row.mean_rna_rmse),
            float(row.mean_mca_prior_aupr),
            xerr=float(row.sd_rna_rmse) if pd.notna(row.sd_rna_rmse) else 0.0,
            yerr=float(row.sd_mca_prior_aupr) if pd.notna(row.sd_mca_prior_aupr) else 0.0,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            markersize=6.0,
            label=str(row.method),
        )
        ax.text(
            float(row.mean_rna_rmse),
            float(row.mean_mca_prior_aupr),
            f" {row.method}",
            fontsize=8,
            color=color,
            ha="left",
            va="center",
        )

    ax.set_xlabel("Mean RNA RMSE")
    ax.set_ylabel("Mean AUPR vs MCA prior annotations")
    clean_ax(ax)
    place_legend(ax, mode="outside", title=None)
    savefig(fig, path)
    plt.close(fig)
    return aggregated


def save_mca_recall_comparison(summaries: pd.DataFrame, *, path: Path) -> None:
    set_house_style()
    method_order = [label for label in METHOD_LABELS.values() if label in set(summaries["method"])]
    palette = dict(zip(method_order, sns.color_palette(CAT_PALETTE, n_colors=len(method_order)), strict=False))
    fig, ax = plt.subplots(figsize=(5.6, 2.85))
    sns.lineplot(
        data=summaries,
        x="n_gene_sets",
        y="mca_recall_top10",
        hue="method",
        hue_order=method_order,
        marker="o",
        linewidth=2.3,
        palette=palette,
        ax=ax,
    )

    offsets = {
        method: offset
        for method, offset in zip(method_order, np.linspace(0.025, -0.025, num=max(len(method_order), 1)), strict=False)
    }
    for row in summaries.itertuples(index=False):
        y = float(row.mca_recall_top10) + offsets.get(str(row.method), 0.0)
        ax.text(
            float(row.n_gene_sets),
            y,
            str(row.top10_fraction_label),
            color=palette.get(str(row.method), "0.2"),
            fontsize=7.6,
            ha="center",
            va="center",
        )

    ax.set_xlabel("Number of retained informed pathways")
    ax.set_ylabel("MCA recall in top 10")
    ax.set_ylim(-0.04, 1.05)
    ax.invert_xaxis()
    clean_ax(ax)
    place_legend(ax, mode="manual", title=None, anchor=(1.12, 0.5))
    savefig(fig, path)
    plt.close(fig)


def load_top_pathway_table(
    methods: list[str],
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_dir = _method_dir(method, mofaflex_dir=mofaflex_dir, baselines_dir=baselines_dir)
        for ranking_path in sorted((method_dir / "rankings").glob("*_pathway_ranks.csv"), key=lambda p: _step_sort_key(p.name)):
            step_label = ranking_path.name.removesuffix("_pathway_ranks.csv")
            ranking = load_ranking_for_step(method, method_dir=method_dir, baselines_dir=baselines_dir, step_label=step_label)
            ranking = ranking.sort_values("rank")
            if ranking.empty:
                continue
            top = ranking.iloc[0]
            rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "method_key": method,
                    "step_label": step_label,
                    "step_index": _step_sort_key(step_label),
                    "n_gene_sets": int(step_label.rsplit("_n", 1)[1]) if "_n" in step_label else np.nan,
                    "top_pathway": str(top["pathway_name"]),
                    "top_pathway_pretty": prettify_pathway_name(str(top["pathway_name"])),
                    "top_pathway_source": str(top.get("source", "unknown")),
                    "top_pathway_variance_explained": float(top.get("variance_explained", np.nan)),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(["method", "step_index"]).reset_index(drop=True)


def summarize_top_pathway_stability(top_pathways: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for method, group in top_pathways.groupby("method", sort=False):
        counts = group["top_pathway"].value_counts()
        rows.append(
            {
                "method": method,
                "n_steps": int(len(group)),
                "n_unique_top_pathways": int(counts.size),
                "dominant_top_pathway": str(counts.index[0]) if not counts.empty else None,
                "dominant_top_pathway_pretty": prettify_pathway_name(str(counts.index[0])) if not counts.empty else None,
                "dominant_top_pathway_n_steps": int(counts.iloc[0]) if not counts.empty else 0,
            }
        )
    by_method = pd.DataFrame.from_records(rows)
    pivot = top_pathways.pivot_table(index="step_label", columns="method", values="top_pathway", aggfunc="first")
    pivot = pivot.reset_index()
    method_cols = [col for col in pivot.columns if col != "step_label"]
    pivot["all_methods_same_top_pathway"] = pivot.loc[:, method_cols].nunique(axis=1).eq(1) if method_cols else False
    return by_method, pivot


def load_weight_vector(
    *,
    method: str,
    step_label: str,
    pathway_name: str,
    mofaflex_dir: Path,
    baselines_dir: Path,
) -> pd.Series | None:
    if method == "mofaflex":
        import mofaflex as mfl

        model_path = mofaflex_dir / "models" / f"{step_label}.h5"
        if not model_path.exists():
            return None
        model = mfl.MOFAFLEX.load(model_path, map_location="cpu")
        weights = model.get_weights(return_type="pandas", ordered=False)["rna"].astype(float)
    else:
        run_dir = baselines_dir / method / "runs" / step_label
        if method == "expimap":
            weights = _load_or_align_expimap_weights(run_dir)
        else:
            weights_path = run_dir / BASELINE_WEIGHTS_FILE
            if not weights_path.exists():
                return None
            weights = pd.read_csv(weights_path, index_col=0).astype(float)

    if pathway_name not in weights.index:
        return None
    return weights.loc[pathway_name].astype(float)


def load_weight_matrix(
    *,
    method: str,
    step_label: str,
    mofaflex_dir: Path,
    baselines_dir: Path,
    cache: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame | None:
    cache_key = (method, step_label)
    if cache_key in cache:
        return cache[cache_key]

    if method == "mofaflex":
        import mofaflex as mfl

        model_path = mofaflex_dir / "models" / f"{step_label}.h5"
        if not model_path.exists():
            return None
        model = mfl.MOFAFLEX.load(model_path, map_location="cpu")
        weights = model.get_weights(return_type="pandas", ordered=False)["rna"].astype(float)
    else:
        run_dir = baselines_dir / method / "runs" / step_label
        if method == "expimap":
            weights = _load_or_align_expimap_weights(run_dir)
        else:
            weights_path = run_dir / BASELINE_WEIGHTS_FILE
            if not weights_path.exists():
                return None
            weights = pd.read_csv(weights_path, index_col=0).astype(float)

    cache[cache_key] = weights
    return weights


def extract_top_pathway_gene_rankings(
    top_pathways: pd.DataFrame,
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
    max_top_k: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in top_pathways.itertuples(index=False):
        weights = load_weight_vector(
            method=str(row.method_key),
            step_label=str(row.step_label),
            pathway_name=str(row.top_pathway),
            mofaflex_dir=mofaflex_dir,
            baselines_dir=baselines_dir,
        )
        if weights is None:
            continue
        abs_sorted = weights.abs().sort_values(ascending=False, kind="stable")
        for gene_rank, gene_name in enumerate(abs_sorted.index[:max_top_k], start=1):
            signed_weight = float(weights.loc[gene_name])
            records.append(
                {
                    "method": row.method,
                    "method_key": row.method_key,
                    "step_label": row.step_label,
                    "step_index": row.step_index,
                    "n_gene_sets": row.n_gene_sets,
                    "top_pathway": row.top_pathway,
                    "top_pathway_pretty": row.top_pathway_pretty,
                    "gene_rank": int(gene_rank),
                    "gene_name": str(gene_name),
                    "signed_weight": signed_weight,
                    "abs_weight": float(abs(signed_weight)),
                }
            )
    return pd.DataFrame.from_records(records)


def load_mca_pathway_table(
    methods: list[str],
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_dir = _method_dir(method, mofaflex_dir=mofaflex_dir, baselines_dir=baselines_dir)
        for ranking_path in sorted((method_dir / "rankings").glob("*_pathway_ranks.csv"), key=lambda p: _step_sort_key(p.name)):
            step_label = ranking_path.name.removesuffix("_pathway_ranks.csv")
            ranking = load_ranking_for_step(method, method_dir=method_dir, baselines_dir=baselines_dir, step_label=step_label)
            if ranking.empty or "source" not in ranking.columns:
                continue
            mca_rows = ranking.loc[ranking["source"].eq("mca")].copy()
            for row in mca_rows.itertuples(index=False):
                pathway_name = str(getattr(row, "pathway_name"))
                rows.append(
                    {
                        "method": METHOD_LABELS[method],
                        "method_key": method,
                        "step_label": step_label,
                        "step_index": _step_sort_key(step_label),
                        "n_gene_sets": int(step_label.rsplit("_n", 1)[1]) if "_n" in step_label else np.nan,
                        "pathway_name": pathway_name,
                        "pathway_pretty": prettify_pathway_name(pathway_name),
                        "rank": int(getattr(row, "rank", -1)),
                        "variance_explained": float(getattr(row, "variance_explained", np.nan)),
                    }
                )
    return pd.DataFrame.from_records(rows).sort_values(["method", "pathway_name", "step_index"]).reset_index(drop=True)


def extract_mca_pathway_gene_rankings(
    mca_pathways: pd.DataFrame,
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
    max_top_k: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    weight_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for row in mca_pathways.itertuples(index=False):
        weights = load_weight_matrix(
            method=str(row.method_key),
            step_label=str(row.step_label),
            mofaflex_dir=mofaflex_dir,
            baselines_dir=baselines_dir,
            cache=weight_cache,
        )
        if weights is None or row.pathway_name not in weights.index:
            continue
        pathway_weights = weights.loc[row.pathway_name].astype(float)
        abs_sorted = pathway_weights.abs().sort_values(ascending=False, kind="stable")
        for gene_rank, gene_name in enumerate(abs_sorted.index[:max_top_k], start=1):
            signed_weight = float(pathway_weights.loc[gene_name])
            records.append(
                {
                    "method": row.method,
                    "method_key": row.method_key,
                    "step_label": row.step_label,
                    "step_index": row.step_index,
                    "n_gene_sets": row.n_gene_sets,
                    "pathway_name": row.pathway_name,
                    "pathway_pretty": row.pathway_pretty,
                    "pathway_rank": row.rank,
                    "pathway_variance_explained": row.variance_explained,
                    "gene_rank": int(gene_rank),
                    "gene_name": str(gene_name),
                    "signed_weight": signed_weight,
                    "abs_weight": float(abs(signed_weight)),
                }
            )
    return pd.DataFrame.from_records(records)


def _correlation(left: pd.Series, right: pd.Series) -> float:
    if left.empty or right.empty:
        return np.nan
    if float(left.std(ddof=0)) == 0.0 or float(right.std(ddof=0)) == 0.0:
        return np.nan
    return float(left.corr(right, method="pearson"))


def summarize_top_gene_correlations(top_gene_rankings: pd.DataFrame, *, top_k_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, method_df in top_gene_rankings.groupby("method", sort=False):
        step_tables = {
            step_label: step_df.copy()
            for step_label, step_df in method_df.groupby("step_label", sort=False)
        }
        step_labels = (
            method_df[["step_index", "step_label"]]
            .drop_duplicates()
            .sort_values("step_index")["step_label"]
            .tolist()
        )
        pathway_by_step = (
            method_df.sort_values(["step_index", "gene_rank"])
            .groupby("step_label", sort=False)["top_pathway_pretty"]
            .first()
            .to_dict()
        )

        for top_k in top_k_values:
            top_k_tables = {
                step_label: (
                    step_df.loc[step_df["gene_rank"] <= top_k, ["gene_name", "signed_weight"]]
                    .drop_duplicates(subset="gene_name", keep="first")
                    .set_index("gene_name")["signed_weight"]
                    .astype(float)
                )
                for step_label, step_df in step_tables.items()
            }
            pair_rows = []
            for i, left in enumerate(step_labels):
                for right in step_labels[i + 1 :]:
                    gene_union = sorted(set(top_k_tables[left].index) | set(top_k_tables[right].index))
                    left_weights = top_k_tables[left].reindex(gene_union).fillna(0.0)
                    right_weights = top_k_tables[right].reindex(gene_union).fillna(0.0)
                    corr = _correlation(left_weights, right_weights)
                    pair_rows.append(
                        {
                            "left_step": left,
                            "right_step": right,
                            "left_top_pathway": pathway_by_step[left],
                            "right_top_pathway": pathway_by_step[right],
                            "pearson": corr,
                        }
                    )
            pair_df = pd.DataFrame.from_records(pair_rows)
            rows.append(
                {
                    "method": method,
                    "top_k_genes": int(top_k),
                    "comparison": "All models",
                    "mean_pearson": float(pair_df["pearson"].mean()) if not pair_df.empty else np.nan,
                    "n_pairs": int(pair_df["pearson"].notna().sum()) if not pair_df.empty else 0,
                }
            )
            for pathway in dict.fromkeys(pathway_by_step.values()):
                same = pair_df.loc[
                    pair_df["left_top_pathway"].eq(pathway) & pair_df["right_top_pathway"].eq(pathway),
                    "pearson",
                ]
                if same.empty:
                    continue
                rows.append(
                    {
                        "method": method,
                        "top_k_genes": int(top_k),
                        "comparison": pathway,
                        "mean_pearson": float(same.mean()),
                        "n_pairs": int(same.notna().sum()),
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_mca_gene_set_correlations(mca_gene_rankings: pd.DataFrame, *, top_k_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["method", "pathway_name", "pathway_pretty"]
    for (method, pathway_name, pathway_pretty), pathway_df in mca_gene_rankings.groupby(group_cols, sort=False):
        step_tables = {
            step_label: step_df.copy()
            for step_label, step_df in pathway_df.groupby("step_label", sort=False)
        }
        step_labels = (
            pathway_df[["step_index", "step_label"]]
            .drop_duplicates()
            .sort_values("step_index")["step_label"]
            .tolist()
        )
        for top_k in top_k_values:
            top_k_tables = {
                step_label: (
                    step_df.loc[step_df["gene_rank"] <= top_k, ["gene_name", "signed_weight"]]
                    .drop_duplicates(subset="gene_name", keep="first")
                    .set_index("gene_name")["signed_weight"]
                    .astype(float)
                )
                for step_label, step_df in step_tables.items()
            }
            pair_correlations = []
            for i, left in enumerate(step_labels):
                for right in step_labels[i + 1 :]:
                    gene_union = sorted(set(top_k_tables[left].index) | set(top_k_tables[right].index))
                    left_weights = top_k_tables[left].reindex(gene_union).fillna(0.0)
                    right_weights = top_k_tables[right].reindex(gene_union).fillna(0.0)
                    pair_correlations.append(_correlation(left_weights, right_weights))
            finite = pd.Series(pair_correlations, dtype=float).dropna()
            rows.append(
                {
                    "method": method,
                    "pathway_name": pathway_name,
                    "pathway_pretty": pathway_pretty,
                    "top_k_genes": int(top_k),
                    "mean_pearson": float(finite.mean()) if not finite.empty else np.nan,
                    "n_pairs": int(finite.size),
                    "n_steps": int(len(step_labels)),
                }
            )
    return pd.DataFrame.from_records(rows)


def save_top_gene_correlation_plot(summary: pd.DataFrame, *, path: Path) -> None:
    set_house_style()
    if summary.empty:
        return
    plot_df = summary.loc[summary["top_k_genes"] >= 50].copy()
    if plot_df.empty:
        plot_df = summary.copy()
    n_methods = plot_df["method"].nunique()
    g = sns.relplot(
        data=plot_df,
        x="top_k_genes",
        y="mean_pearson",
        hue="comparison",
        col="method",
        col_wrap=min(3, n_methods),
        kind="line",
        marker="o",
        height=3.2,
        aspect=1.15,
        palette=CAT_PALETTE,
    )
    g.set_axis_labels("Top genes per top pathway", "Mean Pearson correlation")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_ylim(-0.05, 1.05)
        clean_ax(ax)
    if g.legend is not None:
        g.legend.set_frame_on(False)
        g.legend.set_title("")
    savefig(g.fig, path)
    plt.close(g.fig)


def save_top_gene_correlation_all_models_plot(summary: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    set_house_style()
    plot_df = summary.loc[
        summary["comparison"].eq("All models") & summary["top_k_genes"].ge(50)
    ].copy()
    if plot_df.empty:
        plot_df = summary.loc[summary["comparison"].eq("All models")].copy()
    if plot_df.empty:
        return plot_df

    method_order = [label for label in METHOD_LABELS.values() if label in set(plot_df["method"])]
    palette = dict(zip(method_order, sns.color_palette(CAT_PALETTE, n_colors=len(method_order)), strict=False))

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    sns.lineplot(
        data=plot_df,
        x="top_k_genes",
        y="mean_pearson",
        hue="method",
        hue_order=method_order,
        marker="o",
        linewidth=1.8,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Top genes from each model's top pathway")
    ax.set_ylabel("Mean Pearson correlation")
    ax.set_ylim(-0.05, 1.05)
    clean_ax(ax)
    place_legend(ax, mode="outside", title=None)
    savefig(fig, path)
    plt.close(fig)
    return plot_df


def mofaflex_mca_pathway_order(mca_pathways: pd.DataFrame) -> list[str]:
    order = (
        mca_pathways.loc[mca_pathways["method_key"].eq("mofaflex")]
        .groupby(["pathway_name", "pathway_pretty"], sort=False)["variance_explained"]
        .mean()
        .reset_index()
        .sort_values("variance_explained", ascending=False, kind="stable")
    )
    return order["pathway_pretty"].tolist()


def save_mca_gene_set_correlation_plot(
    summary: pd.DataFrame,
    *,
    path: Path,
    pathway_order: list[str] | None = None,
) -> pd.DataFrame:
    set_house_style()
    plot_df = summary.loc[summary["top_k_genes"].ge(50)].copy()
    if plot_df.empty:
        plot_df = summary.copy()
    plot_df = plot_df.loc[plot_df["n_pairs"].gt(0) & plot_df["mean_pearson"].notna()].copy()
    if plot_df.empty:
        return plot_df

    if pathway_order is None:
        pathway_order = (
            plot_df.loc[:, ["pathway_pretty", "pathway_name"]]
            .drop_duplicates()
            .sort_values("pathway_pretty")["pathway_pretty"]
            .tolist()
        )
    else:
        pathway_order = [name for name in pathway_order if name in set(plot_df["pathway_pretty"])]
    method_order = [label for label in METHOD_LABELS.values() if label in set(plot_df["method"])]
    plot_df["pathway_pretty"] = pd.Categorical(plot_df["pathway_pretty"], categories=pathway_order, ordered=True)
    plot_df = plot_df.sort_values(["pathway_pretty", "method", "top_k_genes"]).reset_index(drop=True)
    g = sns.relplot(
        data=plot_df,
        x="top_k_genes",
        y="mean_pearson",
        hue="method",
        hue_order=method_order,
        col="pathway_pretty",
        col_order=pathway_order,
        col_wrap=3,
        kind="line",
        marker="o",
        linewidth=1.5,
        height=2.25,
        aspect=1.25,
        palette=CAT_PALETTE,
    )
    g.set_axis_labels("Top genes", "Mean Pearson correlation")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_ylim(-1.05, 1.05)
        clean_ax(ax)
    if g.legend is not None:
        g.legend.set_frame_on(False)
        g.legend.set_title("")
    savefig(g.fig, path)
    plt.close(g.fig)
    return plot_df


def run(args: argparse.Namespace) -> None:
    set_house_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str))

    summaries = pd.concat(
        [
            load_method_summary(method, mofaflex_dir=args.mofaflex_dir, baselines_dir=args.baselines_dir)
            for method in args.methods
        ],
        ignore_index=True,
    )
    summaries.to_csv(args.out_dir / "mca_recall_top10_by_method.csv", index=False)
    save_mca_recall_comparison(summaries, path=args.out_dir / "mca_recall_top10_by_method.png")

    top_pathways = load_top_pathway_table(args.methods, mofaflex_dir=args.mofaflex_dir, baselines_dir=args.baselines_dir)
    top_pathways.to_csv(args.out_dir / "top_pathway_by_step_and_method.csv", index=False)
    by_method, by_step = summarize_top_pathway_stability(top_pathways)
    by_method.to_csv(args.out_dir / "top_pathway_stability_by_method.csv", index=False)
    by_step.to_csv(args.out_dir / "top_pathway_same_across_methods_by_step.csv", index=False)

    top_k_values = sorted(set(int(k) for k in args.top_k_genes))
    top_gene_rankings = extract_top_pathway_gene_rankings(
        top_pathways,
        mofaflex_dir=args.mofaflex_dir,
        baselines_dir=args.baselines_dir,
        max_top_k=max(top_k_values),
    )
    top_gene_rankings.to_csv(args.out_dir / "top_pathway_gene_rankings_by_method.csv", index=False)
    correlation_summary = summarize_top_gene_correlations(top_gene_rankings, top_k_values=top_k_values)
    correlation_summary.to_csv(args.out_dir / "top_pathway_gene_pearson_summary_with_pathways_by_method.csv", index=False)
    save_top_gene_correlation_plot(
        correlation_summary,
        path=args.out_dir / "top_pathway_gene_pearson_summary_with_pathways_by_method.png",
    )
    all_models_correlation = save_top_gene_correlation_all_models_plot(
        correlation_summary,
        path=args.out_dir / "top_pathway_gene_pearson_all_models_by_method.png",
    )
    all_models_correlation.to_csv(
        args.out_dir / "top_pathway_gene_pearson_all_models_by_method.csv",
        index=False,
    )

    mca_pathways = load_mca_pathway_table(args.methods, mofaflex_dir=args.mofaflex_dir, baselines_dir=args.baselines_dir)
    mca_pathways.to_csv(args.out_dir / "mca_pathway_by_step_and_method.csv", index=False)
    filtered_mca_pathways = mca_pathways.loc[
        ~mca_pathways["pathway_name"].isin(DEFAULT_EXCLUDED_MCA_PATHWAYS)
    ].copy()
    filtered_mca_pathways.to_csv(args.out_dir / "mca_pathway_by_step_and_method_filtered.csv", index=False)
    mca_gene_rankings = extract_mca_pathway_gene_rankings(
        filtered_mca_pathways,
        mofaflex_dir=args.mofaflex_dir,
        baselines_dir=args.baselines_dir,
        max_top_k=max(top_k_values),
    )
    mca_gene_rankings.to_csv(args.out_dir / "mca_pathway_gene_rankings_by_method.csv", index=False)
    mca_correlation_summary = summarize_mca_gene_set_correlations(mca_gene_rankings, top_k_values=top_k_values)
    mca_correlation_summary.to_csv(args.out_dir / "mca_pathway_gene_pearson_by_method.csv", index=False)
    if args.mca_pathway_order_file is not None:
        pathway_order = pd.read_csv(args.mca_pathway_order_file)["pathway_pretty"].astype(str).tolist()
    else:
        pathway_order = mofaflex_mca_pathway_order(filtered_mca_pathways)
    pd.DataFrame({"pathway_pretty": pathway_order}).to_csv(
        args.out_dir / "mca_pathway_gene_pearson_mofaflex_order.csv",
        index=False,
    )
    mca_correlation_plot = save_mca_gene_set_correlation_plot(
        mca_correlation_summary,
        path=args.out_dir / "mca_pathway_gene_pearson_by_method.png",
        pathway_order=pathway_order,
    )
    mca_correlation_plot.to_csv(
        args.out_dir / "mca_pathway_gene_pearson_by_method_plotted.csv",
        index=False,
    )

    mca_aupr_rmse = build_mca_aupr_rmse_table(
        args.methods,
        mofaflex_dir=args.mofaflex_dir,
        baselines_dir=args.baselines_dir,
    )
    mca_aupr_rmse.to_csv(args.out_dir / "mca_prior_aupr_vs_rna_rmse_by_method.csv", index=False)
    save_mca_aupr_vs_rmse_plot(
        mca_aupr_rmse,
        path=args.out_dir / "mca_prior_aupr_vs_rna_rmse_by_method.png",
    )
    mca_aupr_rmse_mean_sd = save_mca_aupr_vs_rmse_mean_sd_plot(
        mca_aupr_rmse,
        path=args.out_dir / "mca_prior_aupr_vs_rna_rmse_mean_sd_by_method.png",
    )
    mca_aupr_rmse_mean_sd.to_csv(
        args.out_dir / "mca_prior_aupr_vs_rna_rmse_mean_sd_by_method.csv",
        index=False,
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
