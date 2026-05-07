from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import issparse

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.mca_filter_sensitivity import (
        COLLECTION_PRESETS,
        DEFAULT_DATA_PATH,
        DEFAULT_INIT_FACTORS,
        DEFAULT_INIT_SCALE,
        DEFAULT_LIKELIHOODS,
        DEFAULT_LR,
        DEFAULT_N_PARTICLES,
        DEFAULT_NONNEGATIVE_FACTORS,
        DEFAULT_NONNEGATIVE_WEIGHTS,
        DEFAULT_WEIGHT_PRIOR,
        attach_gene_set_mask,
        fit_model,
        load_collection_gene_set_stats,
    )
    from mofaflex_sensitivity.plot_style import CAT_PALETTE, set_house_style
    from mofaflex_sensitivity.uninformed_sensitivity import (
        aggregate_sweep_metrics,
        compute_processed_subset_r2,
        factor_subset_names,
        matched_pcgse_activity_table,
        matched_pcgse_table,
        summarize_model_sensitivity,
    )
else:
    from .mca_filter_sensitivity import (
        COLLECTION_PRESETS,
        DEFAULT_DATA_PATH,
        DEFAULT_INIT_FACTORS,
        DEFAULT_INIT_SCALE,
        DEFAULT_LIKELIHOODS,
        DEFAULT_LR,
        DEFAULT_N_PARTICLES,
        DEFAULT_NONNEGATIVE_FACTORS,
        DEFAULT_NONNEGATIVE_WEIGHTS,
        DEFAULT_WEIGHT_PRIOR,
        attach_gene_set_mask,
        fit_model,
        load_collection_gene_set_stats,
    )
    from .plot_style import CAT_PALETTE, set_house_style
    from .uninformed_sensitivity import (
        aggregate_sweep_metrics,
        compute_processed_subset_r2,
        factor_subset_names,
        matched_pcgse_activity_table,
        matched_pcgse_table,
        summarize_model_sensitivity,
    )


DEFAULT_OUT_DIR = Path("artifacts/sln208_gene_set_uninformed_sweep")
HALLMARK_PRESET = COLLECTION_PRESETS["hallmark"]
DEFAULT_GENE_SET_SOURCE = "hallmark+mca"
DEFAULT_SEEDS = [42]
DEFAULT_ACTIVE_R2_THRESHOLD = 0.01
DEFAULT_COMPARISON_ACTIVE_R2_THRESHOLD = 0.001
_SWEEP_PALETTE = sns.color_palette(CAT_PALETTE, n_colors=6)
PRIMARY_COLOR = _SWEEP_PALETTE[0]
SECONDARY_COLOR = _SWEEP_PALETTE[1]
TERTIARY_COLOR = _SWEEP_PALETTE[2]
QUATERNARY_COLOR = _SWEEP_PALETTE[3]
FIFTH_COLOR = _SWEEP_PALETTE[4]
SIXTH_COLOR = _SWEEP_PALETTE[5]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SLN208 gene-set uninformed-factor sweep from the notebook as a "
            "plain Python script."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-uninformed-grid", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--active-r2-threshold", type=float, default=DEFAULT_ACTIVE_R2_THRESHOLD)
    parser.add_argument("--save-models", type=_parse_bool, default=True)
    parser.add_argument("--save-processed-subset-r2", type=_parse_bool, default=True)
    return parser


def _parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = text.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {text!r}.")


def aggregate_significant_pathways(significant_df: pd.DataFrame) -> pd.DataFrame:
    if significant_df.empty:
        return pd.DataFrame(
            columns=[
                "n_uninformed_factors",
                "n_matched_pathways_mean",
                "n_significant_pathways_mean",
                "n_significant_pathways_sd",
                "fraction_significant_pathways_mean",
                "fraction_significant_pathways_sd",
                "n_active_pathways_mean",
                "n_active_pathways_sd",
                "fraction_active_pathways_mean",
                "fraction_active_pathways_sd",
                "n_active_significant_pathways_mean",
                "n_active_significant_pathways_sd",
                "fraction_active_significant_pathways_mean",
                "fraction_active_significant_pathways_sd",
            ]
        )
    return (
        significant_df.groupby("n_uninformed_factors", as_index=False)
        .agg(
            n_matched_pathways_mean=("n_matched_pathways", "mean"),
            n_significant_pathways_mean=("n_significant_pathways", "mean"),
            n_significant_pathways_sd=("n_significant_pathways", "std"),
            fraction_significant_pathways_mean=("fraction_significant_pathways", "mean"),
            fraction_significant_pathways_sd=("fraction_significant_pathways", "std"),
            n_active_pathways_mean=("n_active_pathways", "mean"),
            n_active_pathways_sd=("n_active_pathways", "std"),
            fraction_active_pathways_mean=("fraction_active_pathways", "mean"),
            fraction_active_pathways_sd=("fraction_active_pathways", "std"),
            n_active_significant_pathways_mean=("n_active_significant_pathways", "mean"),
            n_active_significant_pathways_sd=("n_active_significant_pathways", "std"),
            fraction_active_significant_pathways_mean=("fraction_active_significant_pathways", "mean"),
            fraction_active_significant_pathways_sd=("fraction_active_significant_pathways", "std"),
        )
        .fillna(0.0)
    )


def plot_main_summary(summary_df: pd.DataFrame, *, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].errorbar(
        summary_df["n_uninformed_factors"],
        summary_df["total_r2_sum_mean"],
        yerr=summary_df.get("total_r2_sum_sd", 0.0),
        marker="o",
    )
    axes[0].axvline(3, color="black", linestyle="--", alpha=0.6)
    axes[0].set_title("Total explained variance")
    axes[0].set_xlabel("Uninformed factors")
    axes[0].set_ylabel("Summed R2")

    if "informed_r2_sum_mean" in summary_df and "uninformed_r2_sum_mean" in summary_df:
        axes[1].errorbar(
            summary_df["n_uninformed_factors"],
            summary_df["informed_r2_sum_mean"],
            yerr=summary_df.get("informed_r2_sum_sd", 0.0),
            marker="o",
            label="Informed",
        )
        axes[1].errorbar(
            summary_df["n_uninformed_factors"],
            summary_df["uninformed_r2_sum_mean"],
            yerr=summary_df.get("uninformed_r2_sum_sd", 0.0),
            marker="o",
            label="Uninformed",
        )
    axes[1].axvline(3, color="black", linestyle="--", alpha=0.6)
    axes[1].set_title("Explained variance by factor type")
    axes[1].set_xlabel("Uninformed factors")
    axes[1].set_ylabel("Summed R2")
    axes[1].legend()

    if "processed_all_r2_total_mean" in summary_df:
        for column, label in [
            ("processed_all_r2_total_mean", "All"),
            ("processed_informed_r2_total_mean", "Informed"),
            ("processed_uninformed_r2_total_mean", "Uninformed"),
        ]:
            if column in summary_df:
                axes[2].plot(summary_df["n_uninformed_factors"], summary_df[column], marker="o", label=label)
        axes[2].legend()
    axes[2].axvline(3, color="black", linestyle="--", alpha=0.6)
    axes[2].set_title("Processed-space subset R2")
    axes[2].set_xlabel("Uninformed factors")
    axes[2].set_ylabel("R2")
    axes[2].set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_significant_pathways(significant_summary_df: pd.DataFrame, *, path: Path) -> None:
    if significant_summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].errorbar(
        significant_summary_df["n_uninformed_factors"],
        significant_summary_df["n_significant_pathways_mean"],
        yerr=significant_summary_df["n_significant_pathways_sd"],
        marker="o",
    )
    axes[0].axvline(3, color="black", linestyle="--", alpha=0.6)
    axes[0].set_title("Matched significant pathways (padj < 0.05)")
    axes[0].set_xlabel("Uninformed factors")
    axes[0].set_ylabel("Count of significant matched pathways")

    axes[1].errorbar(
        significant_summary_df["n_uninformed_factors"],
        significant_summary_df["fraction_significant_pathways_mean"],
        yerr=significant_summary_df["fraction_significant_pathways_sd"],
        marker="o",
    )
    axes[1].axvline(3, color="black", linestyle="--", alpha=0.6)
    axes[1].set_title("Fraction of matched pathways significant")
    axes[1].set_xlabel("Uninformed factors")
    axes[1].set_ylabel("Fraction significant")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_uninformed_share_of_all_explained(metrics_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"n_uninformed_factors", "processed_uninformed_r2_total", "processed_all_r2_total"}
    if not required_columns.issubset(metrics_df.columns):
        return pd.DataFrame(
            columns=[
                "n_uninformed_factors",
                "share_mean",
                "share_sd",
                "processed_uninformed_r2_total_mean",
                "processed_all_r2_total_mean",
            ]
        )

    share_df = metrics_df.loc[:, ["n_uninformed_factors", "processed_uninformed_r2_total", "processed_all_r2_total"]].copy()
    share_df["share"] = share_df["processed_uninformed_r2_total"] / share_df["processed_all_r2_total"]
    summary = (
        share_df.groupby("n_uninformed_factors", as_index=False)
        .agg(
            share_mean=("share", "mean"),
            share_sd=("share", "std"),
            processed_uninformed_r2_total_mean=("processed_uninformed_r2_total", "mean"),
            processed_all_r2_total_mean=("processed_all_r2_total", "mean"),
        )
        .fillna(0.0)
    )
    return summary.sort_values("n_uninformed_factors").reset_index(drop=True)


def plot_uninformed_share_of_all_explained(share_summary_df: pd.DataFrame, *, path: Path) -> None:
    if share_summary_df.empty:
        return

    plot_df = share_summary_df.loc[share_summary_df["n_uninformed_factors"] > 0, :].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = plot_df["n_uninformed_factors"]
    ax.errorbar(
        x,
        plot_df["share_mean"],
        yerr=plot_df["share_sd"],
        marker="o",
        color=PRIMARY_COLOR,
        linewidth=2,
    )
    ax.axhline(0.8, color="black", linestyle="--", alpha=0.65, linewidth=1.2)
    ax.axvline(3, color="black", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.set_xlabel("Uninformed factors")
    ax.set_ylabel("Uninformed share of full-model explained variance")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_title("Explained variance captured by uninformed factors")

    if (plot_df["n_uninformed_factors"] == 3).any():
        row = plot_df.loc[plot_df["n_uninformed_factors"] == 3].iloc[0]
        ax.annotate(
            f"{100.0 * row['share_mean']:.1f}%",
            xy=(3, row["share_mean"]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_bars(
    tradeoff_df: pd.DataFrame,
    *,
    path: Path,
    value_column: str,
    value_label: str,
    title: str,
    count_column: str | None = None,
    count_label: str | None = None,
    count_ylabel: str | None = None,
) -> None:
    if tradeoff_df.empty:
        return

    plot_df = tradeoff_df.sort_values("n_uninformed_factors").reset_index(drop=True)
    x = np.arange(len(plot_df))
    resolved_count_column = (
        count_column
        if count_column is not None
        else (
            "n_active_significant_pathways"
            if "n_active_significant_pathways" in plot_df.columns
            else "n_significant_pathways"
        )
    )
    resolved_count_label = (
        count_label
        if count_label is not None
        else (
            "Active pathways"
            if resolved_count_column == "n_active_significant_pathways"
            else "Significant pathways"
        )
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax.bar(
        x,
        plot_df[value_column],
        color=PRIMARY_COLOR,
        edgecolor="black",
        linewidth=0.6,
        width=0.72,
        zorder=2,
    )
    ax.set_xlabel("Number of uninformed factors (full model)")
    ax.set_ylabel(value_label)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["n_uninformed_factors"].astype(int))

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.annotate(
            f"{row.__getattribute__(value_column):.3f}",
            xy=(idx, row.__getattribute__(value_column)),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="white",
        )

    ax2 = ax.twinx()
    ax2.plot(
        x,
        plot_df[resolved_count_column],
        color=QUATERNARY_COLOR,
        marker="o",
        markersize=5,
        linewidth=1.5,
        linestyle="--",
        zorder=4,
    )
    ax2.scatter(
        x,
        plot_df[resolved_count_column],
        s=40 + 10 * plot_df[resolved_count_column],
        c=plot_df[resolved_count_column],
        cmap="magma",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )
    if count_ylabel is not None:
        ax2.set_ylabel(count_ylabel)
    elif resolved_count_column == "n_active_significant_pathways":
        ax2.set_ylabel("Active pathways (PCGSE padj < 0.05 and r2_rna >= 0.01)")
    elif resolved_count_column == "n_active_pathways":
        ax2.set_ylabel("Active pathways (r2_rna >= 0.01)")
    else:
        ax2.set_ylabel("Significant matched pathways (PCGSE, padj < 0.05)")
    max_sig = float(plot_df[resolved_count_column].max()) if not plot_df.empty else 0.0
    ax2.set_ylim(0.0, max(1.0, max_sig + 2.0))

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax2.annotate(
            str(int(row.__getattribute__(resolved_count_column))),
            xy=(idx, row.__getattribute__(resolved_count_column)),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    bar_proxy = mpl.patches.Patch(facecolor=PRIMARY_COLOR, edgecolor="black", label=value_label)
    line_proxy = mpl.lines.Line2D(
        [0],
        [0],
        color=QUATERNARY_COLOR,
        marker="o",
        linestyle="--",
        label=resolved_count_label,
    )
    ax.legend(
        handles=[bar_proxy, line_proxy],
        loc="upper right",
        frameon=True,
        fontsize=9,
    )
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_factor_r2_split_bars(
    metrics_df: pd.DataFrame,
    *,
    path: Path,
    uninformed_column: str,
    informed_column: str,
    ylabel: str,
    title: str,
    tradeoff_df: pd.DataFrame | None = None,
    count_column: str | None = None,
    count_label: str | None = None,
    count_ylabel: str | None = None,
) -> None:
    if metrics_df.empty:
        return

    plot_df = metrics_df.loc[metrics_df["n_uninformed_factors"] >= 1].sort_values("n_uninformed_factors").reset_index(drop=True)
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax.bar(
        x,
        plot_df[uninformed_column],
        width=0.72,
        color=PRIMARY_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="Uninformed",
    )
    ax.bar(
        x,
        plot_df[informed_column],
        width=0.72,
        bottom=plot_df[uninformed_column],
        color=TERTIARY_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="Informed",
    )
    ax.set_xlabel("Number of uninformed factors (full model)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["n_uninformed_factors"].astype(int))

    legend_handles = [
        mpl.patches.Patch(facecolor=PRIMARY_COLOR, edgecolor="black", label="Uninformed"),
        mpl.patches.Patch(facecolor=TERTIARY_COLOR, edgecolor="black", label="Informed"),
    ]

    if tradeoff_df is not None and count_column is not None and count_column in tradeoff_df.columns:
        overlay_df = (
            tradeoff_df.loc[tradeoff_df["n_uninformed_factors"] >= 1, ["n_uninformed_factors", count_column]]
            .drop_duplicates(subset=["n_uninformed_factors"])
            .sort_values("n_uninformed_factors")
            .reset_index(drop=True)
        )
        ax2 = ax.twinx()
        ax2.plot(
            x,
            overlay_df[count_column],
            color=QUATERNARY_COLOR,
            marker="o",
            markersize=5,
            linewidth=1.5,
            linestyle="--",
            zorder=5,
        )
        ax2.scatter(
            x,
            overlay_df[count_column],
            s=40 + 10 * overlay_df[count_column],
            c=overlay_df[count_column],
            cmap="magma",
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
        )
        for idx, row in enumerate(overlay_df.itertuples(index=False)):
            ax2.annotate(
                str(int(row.__getattribute__(count_column))),
                xy=(idx, row.__getattribute__(count_column)),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        max_sig = float(overlay_df[count_column].max()) if not overlay_df.empty else 0.0
        ax2.set_ylim(0.0, max(1.0, max_sig + 2.0))
        ax2.set_ylabel(count_ylabel or count_label or "Overlay")
        legend_handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                color=QUATERNARY_COLOR,
                marker="o",
                linestyle="--",
                label=count_label or "Overlay",
            )
        )

    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _to_dense_array(arr) -> np.ndarray:
    if issparse(arr):
        return arr.toarray()
    return np.asarray(arr)


def _prepare_reconstruction_payload(model, mdata) -> dict[str, Any]:
    dataset = model._mofaflexdataset(mdata)
    factors_by_group = model.get_factors(return_type="pandas", ordered=False)
    weights_by_view = model.get_weights(return_type="pandas", ordered=False)
    processed_data = dataset.apply(
        lambda adata, group_name, view_name: {
            "y_true": _to_dense_array(dataset.preprocessor(adata.X, slice(None), slice(None), group_name, view_name)[0]),
            "obs_names": adata.obs_names.to_numpy(),
            "var_names": adata.var_names.to_numpy(),
        }
    )
    return {
        "processed_data": processed_data,
        "factors_by_group": factors_by_group,
        "weights_by_view": weights_by_view,
    }


def _compute_subset_r2_from_payload(
    payload: dict[str, Any],
    *,
    factor_names: list[str],
) -> dict[str, float]:
    processed_data = payload["processed_data"]
    factors_by_group = payload["factors_by_group"]
    weights_by_view = payload["weights_by_view"]

    ss_res_total = 0.0
    ss_tot_total = 0.0
    per_view_ss: dict[str, dict[str, float]] = {}

    for group_name, group_views in processed_data.items():
        factor_df = factors_by_group[group_name]
        for view_name, view_payload in group_views.items():
            y_true = np.asarray(view_payload["y_true"], dtype=float)
            obs_names = pd.Index(view_payload["obs_names"])
            var_names = pd.Index(view_payload["var_names"])

            if factor_names:
                factor_subset = factor_df.reindex(index=obs_names, columns=factor_names, fill_value=0.0)
                weight_subset = weights_by_view[view_name].reindex(
                    index=factor_names,
                    columns=var_names,
                    fill_value=0.0,
                )
                y_pred = factor_subset.to_numpy(dtype=float) @ weight_subset.to_numpy(dtype=float)
            else:
                y_pred = np.zeros_like(y_true, dtype=float)

            ss_res = float(np.nansum(np.square(y_true - y_pred)))
            ss_tot = float(np.nansum(np.square(y_true)))
            ss_res_total += ss_res
            ss_tot_total += ss_tot

            per_view_ss.setdefault(view_name, {"ss_res": 0.0, "ss_tot": 0.0})
            per_view_ss[view_name]["ss_res"] += ss_res
            per_view_ss[view_name]["ss_tot"] += ss_tot

    metrics = {
        "r2_total": float(1.0 - ss_res_total / ss_tot_total) if ss_tot_total > 0 else np.nan,
        "ss_res_total": float(ss_res_total),
        "ss_tot_total": float(ss_tot_total),
    }
    for view_name, values in per_view_ss.items():
        metrics[f"r2_{view_name}"] = (
            float(1.0 - values["ss_res"] / values["ss_tot"]) if values["ss_tot"] > 0 else np.nan
        )
    return metrics


def _factor_r2_scores(model, factor_names: list[str]) -> pd.DataFrame:
    r2_by_group = model.get_r2(total=False, ordered=False)
    per_factor = pd.DataFrame(index=pd.Index(model.factor_names, dtype="object"))
    for group_df in r2_by_group.values():
        for view_name in group_df.columns:
            column_name = f"r2_{view_name}"
            if column_name not in per_factor.columns:
                per_factor[column_name] = 0.0
            per_factor.loc[group_df.index, column_name] = (
                per_factor.loc[group_df.index, column_name].to_numpy(dtype=float)
                + group_df.loc[:, view_name].to_numpy(dtype=float)
            )
    per_factor["single_r2_total"] = per_factor.sum(axis=1)
    per_factor = per_factor.loc[factor_names, :].reset_index(names="factor")
    return per_factor.sort_values("single_r2_total", ascending=False).reset_index(drop=True)


def compute_factor_pareto(
    model,
    mdata,
    *,
    factor_names: list[str],
    max_explicit_factors: int | None = None,
    collapse_remaining: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not factor_names:
        return pd.DataFrame(), {}

    payload = _prepare_reconstruction_payload(model, mdata)
    uninformed_factors = factor_subset_names(model, "uninformed")
    informed_factors = factor_subset_names(model, "informed")
    all_factors = factor_subset_names(model, "all")

    per_factor_df = _factor_r2_scores(model, factor_names)

    selected_subset_metrics = _compute_subset_r2_from_payload(payload, factor_names=factor_names)
    uninformed_metrics = _compute_subset_r2_from_payload(payload, factor_names=uninformed_factors)
    informed_metrics = _compute_subset_r2_from_payload(payload, factor_names=informed_factors)
    all_factor_metrics = _compute_subset_r2_from_payload(payload, factor_names=all_factors)

    cumulative_rows = []
    selected_factors: list[str] = []
    previous_cumulative_r2 = 0.0
    selected_subset_total_r2 = selected_subset_metrics["r2_total"]
    all_factor_total_r2 = all_factor_metrics["r2_total"]
    if max_explicit_factors is None or max_explicit_factors <= 0:
        explicit_count = len(per_factor_df)
    else:
        explicit_count = min(max_explicit_factors, len(per_factor_df))

    explicit_df = per_factor_df.iloc[:explicit_count, :].copy()
    for _, row in explicit_df.iterrows():
        selected_factors.append(row["factor"])
        cumulative_metrics = _compute_subset_r2_from_payload(payload, factor_names=selected_factors)
        cumulative_r2_total = cumulative_metrics["r2_total"]
        cumulative_rows.append(
            {
                "order": len(selected_factors),
                "display_label": str(len(selected_factors)),
                "factor": row["factor"],
                "single_r2_total": row["single_r2_total"],
                "single_r2_rna": row.get("r2_rna", np.nan),
                "single_r2_prot": row.get("r2_prot", np.nan),
                "incremental_r2_total": cumulative_r2_total - previous_cumulative_r2,
                "cumulative_r2_total": cumulative_r2_total,
                "cumulative_r2_rna": cumulative_metrics.get("r2_rna", np.nan),
                "cumulative_r2_prot": cumulative_metrics.get("r2_prot", np.nan),
                "cumulative_fraction_of_selected_subset_total": (
                    cumulative_r2_total / selected_subset_total_r2 if selected_subset_total_r2 > 0 else np.nan
                ),
                "cumulative_fraction_of_all_model_total": (
                    cumulative_r2_total / all_factor_total_r2 if all_factor_total_r2 > 0 else np.nan
                ),
            }
        )
        previous_cumulative_r2 = cumulative_r2_total

    remaining_count = len(per_factor_df) - explicit_count
    if collapse_remaining and remaining_count > 0:
        full_factor_order = per_factor_df["factor"].tolist()
        cumulative_metrics = _compute_subset_r2_from_payload(payload, factor_names=full_factor_order)
        cumulative_r2_total = cumulative_metrics["r2_total"]
        cumulative_rows.append(
            {
                "order": len(full_factor_order),
                "display_label": f"Rest ({remaining_count})",
                "factor": "Rest",
                "single_r2_total": np.nan,
                "single_r2_rna": np.nan,
                "single_r2_prot": np.nan,
                "incremental_r2_total": cumulative_r2_total - previous_cumulative_r2,
                "cumulative_r2_total": cumulative_r2_total,
                "cumulative_r2_rna": cumulative_metrics.get("r2_rna", np.nan),
                "cumulative_r2_prot": cumulative_metrics.get("r2_prot", np.nan),
                "cumulative_fraction_of_selected_subset_total": (
                    cumulative_r2_total / selected_subset_total_r2 if selected_subset_total_r2 > 0 else np.nan
                ),
                "cumulative_fraction_of_all_model_total": (
                    cumulative_r2_total / all_factor_total_r2 if all_factor_total_r2 > 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(cumulative_rows), {
        "selected_subset_total_r2": selected_subset_total_r2,
        "uninformed_total_r2": uninformed_metrics["r2_total"],
        "informed_total_r2": informed_metrics["r2_total"],
        "all_factor_total_r2": all_factor_total_r2,
    }


def compute_uninformed_pareto(
    model,
    mdata,
) -> tuple[pd.DataFrame, dict[str, float]]:
    return compute_factor_pareto(
        model,
        mdata,
        factor_names=factor_subset_names(model, "uninformed"),
    )


def compute_all_factor_pareto(
    model,
    mdata,
) -> tuple[pd.DataFrame, dict[str, float]]:
    return compute_factor_pareto(
        model,
        mdata,
        factor_names=factor_subset_names(model, "all"),
        max_explicit_factors=20,
        collapse_remaining=True,
    )


def _resolve_batch_series(mdata) -> pd.Series:
    candidate_columns = [
        "rna:batch",
        "batch",
    ]
    for column_name in candidate_columns:
        if column_name in mdata.obs.columns:
            return mdata.obs[column_name].astype(str)

    for column_name in mdata.obs.columns:
        if column_name.lower().endswith(":batch") or column_name.lower() == "batch":
            return mdata.obs[column_name].astype(str)

    raise KeyError("Could not find a batch column in mdata.obs.")


def _resolve_celltype_series(mdata) -> pd.Series:
    candidate_columns = [
        "rna:cell_types (high)",
        "rna:cell_types",
        "cell_types (high)",
        "cell_types",
    ]
    for column_name in candidate_columns:
        if column_name in mdata.obs.columns:
            return mdata.obs[column_name].astype(str)

    for column_name in mdata.obs.columns:
        lowered = column_name.lower()
        if "cell" in lowered and "type" in lowered:
            return mdata.obs[column_name].astype(str)

    raise KeyError("Could not find a cell type column in mdata.obs.")


def _single_group_factor_scores(model) -> pd.DataFrame:
    factors_by_group = model.get_factors(return_type="pandas", ordered=False)
    if not factors_by_group:
        raise ValueError("No factor scores found in the fitted model.")
    if len(factors_by_group) != 1:
        raise ValueError("Batch prediction pareto currently expects a single-group model.")
    return next(iter(factors_by_group.values())).copy()


def _cross_validated_binary_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_score = np.full(shape=len(y), fill_value=np.nan, dtype=float)

    for train_index, test_index in splitter.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs"),
        )
        model.fit(X[train_index], y[train_index])
        y_score[test_index] = model.predict_proba(X[test_index])[:, 1]

    return float(roc_auc_score(y, y_score))


def _cross_validated_multiclass_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_score = np.full(shape=(len(y), len(classes)), fill_value=np.nan, dtype=float)

    for train_index, test_index in splitter.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, solver="lbfgs"),
        )
        model.fit(X[train_index], y[train_index])
        class_order = model.classes_
        proba = model.predict_proba(X[test_index])
        aligned = np.zeros((len(test_index), len(classes)), dtype=float)
        for class_idx, class_value in enumerate(classes):
            source_idx = int(np.where(class_order == class_value)[0][0])
            aligned[:, class_idx] = proba[:, source_idx]
        y_score[test_index] = aligned

    auc = roc_auc_score(y, y_score, multi_class="ovr", average="macro", labels=classes)
    return float(auc), classes


def compute_batch_prediction_pareto(
    model,
    mdata,
    *,
    factor_names: list[str],
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not factor_names:
        return pd.DataFrame(), {}

    batch_series = _resolve_batch_series(mdata)
    batch_levels = sorted(batch_series.dropna().unique().tolist())
    if len(batch_levels) != 2:
        raise ValueError(
            f"Expected exactly 2 batch labels for binary prediction, found {batch_levels!r}."
        )

    factor_scores = _single_group_factor_scores(model)
    factor_scores = factor_scores.reindex(index=batch_series.index, columns=factor_names)
    y = (batch_series.to_numpy() == batch_levels[1]).astype(int)

    single_factor_rows: list[dict[str, Any]] = []
    for factor_name in factor_names:
        X_single = factor_scores.loc[:, [factor_name]].to_numpy(dtype=float)
        auc_single = _cross_validated_binary_auc(
            X_single,
            y,
            random_state=random_state,
        )
        single_factor_rows.append(
            {
                "factor": factor_name,
                "single_auc": auc_single,
                "single_auc_excess": auc_single - 0.5,
            }
        )

    ranking_df = pd.DataFrame(single_factor_rows).sort_values(
        ["single_auc", "factor"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    ordered_factors = ranking_df["factor"].tolist()
    full_auc = _cross_validated_binary_auc(
        factor_scores.loc[:, ordered_factors].to_numpy(dtype=float),
        y,
        random_state=random_state,
    )

    rows: list[dict[str, Any]] = []
    for order, factor_name in enumerate(ordered_factors, start=1):
        selected = ordered_factors[:order]
        cumulative_auc = _cross_validated_binary_auc(
            factor_scores.loc[:, selected].to_numpy(dtype=float),
            y,
            random_state=random_state,
        )
        single_row = ranking_df.loc[ranking_df["factor"] == factor_name].iloc[0]
        rows.append(
            {
                "order": order,
                "display_label": str(order),
                "factor": factor_name,
                "single_auc": float(single_row["single_auc"]),
                "single_auc_excess": float(single_row["single_auc_excess"]),
                "cumulative_auc": cumulative_auc,
                "cumulative_auc_excess": cumulative_auc - 0.5,
                "cumulative_fraction_of_full_auc_excess": (
                    (cumulative_auc - 0.5) / (full_auc - 0.5) if full_auc > 0.5 else np.nan
                ),
            }
        )

    return pd.DataFrame(rows), {
        "batch_column": str(batch_series.name),
        "batch_levels": batch_levels,
        "n_obs": int(len(batch_series)),
        "full_uninformed_auc": float(full_auc),
        "full_uninformed_auc_excess": float(full_auc - 0.5),
    }


def compute_uninformed_batch_prediction_metrics(
    model,
    mdata,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    factor_names = factor_subset_names(model, "uninformed")
    if not factor_names:
        batch_series = _resolve_batch_series(mdata)
        batch_levels = sorted(batch_series.dropna().unique().tolist())
        return {
            "batch_column": str(batch_series.name),
            "batch_levels": batch_levels,
            "batch_uninformed_auc": np.nan,
            "batch_uninformed_auc_excess": np.nan,
        }

    batch_series = _resolve_batch_series(mdata)
    batch_levels = sorted(batch_series.dropna().unique().tolist())
    if len(batch_levels) != 2:
        raise ValueError(
            f"Expected exactly 2 batch labels for binary prediction, found {batch_levels!r}."
        )

    factor_scores = _single_group_factor_scores(model)
    factor_scores = factor_scores.reindex(index=batch_series.index, columns=factor_names)
    y = (batch_series.to_numpy() == batch_levels[1]).astype(int)
    auc = _cross_validated_binary_auc(
        factor_scores.to_numpy(dtype=float),
        y,
        random_state=random_state,
    )
    return {
        "batch_column": str(batch_series.name),
        "batch_levels": batch_levels,
        "batch_uninformed_auc": float(auc),
        "batch_uninformed_auc_excess": float(auc - 0.5),
    }


def compute_informed_celltype_prediction_metrics(
    model,
    mdata,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    factor_names = factor_subset_names(model, "informed")
    celltype_series = _resolve_celltype_series(mdata)
    valid_mask = celltype_series.notna() & (~celltype_series.astype(str).isin(["nan", "NaN", "None"]))
    celltype_series = celltype_series.loc[valid_mask].astype(str)
    if not factor_names or celltype_series.empty:
        return {
            "celltype_column": str(celltype_series.name),
            "n_celltypes": int(celltype_series.nunique()),
            "celltype_informed_auc": np.nan,
            "celltype_informed_auc_excess": np.nan,
        }

    factor_scores = _single_group_factor_scores(model)
    factor_scores = factor_scores.reindex(index=celltype_series.index, columns=factor_names)
    y_codes, classes = pd.factorize(celltype_series, sort=True)
    auc, _ = _cross_validated_multiclass_auc(
        factor_scores.to_numpy(dtype=float),
        y_codes.astype(int),
        random_state=random_state,
    )
    n_classes = len(classes)
    random_baseline = 1.0 / n_classes if n_classes > 0 else np.nan
    return {
        "celltype_column": str(celltype_series.name),
        "n_celltypes": int(n_classes),
        "celltype_informed_auc": float(auc),
        "celltype_informed_auc_excess": float(auc - random_baseline) if n_classes > 0 else np.nan,
        "celltype_random_baseline_auc": float(random_baseline) if n_classes > 0 else np.nan,
    }


def summarize_batch_prediction_tradeoff(metrics_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = (
        metrics_df.loc[metrics_df["n_uninformed_factors"] >= 1, ["n_uninformed_factors", "batch_uninformed_auc", "batch_uninformed_auc_excess"]]
        .groupby("n_uninformed_factors", as_index=False)
        .agg(
            batch_uninformed_auc_mean=("batch_uninformed_auc", "mean"),
            batch_uninformed_auc_sd=("batch_uninformed_auc", "std"),
            batch_uninformed_auc_excess_mean=("batch_uninformed_auc_excess", "mean"),
            batch_uninformed_auc_excess_sd=("batch_uninformed_auc_excess", "std"),
        )
        .fillna(0.0)
        .sort_values("n_uninformed_factors")
        .reset_index(drop=True)
    )
    max_excess = float(plot_df["batch_uninformed_auc_excess_mean"].max()) if not plot_df.empty else np.nan
    plot_df["fraction_of_best_batch_auc_excess"] = (
        plot_df["batch_uninformed_auc_excess_mean"] / max_excess if max_excess > 0 else np.nan
    )
    return plot_df


def summarize_prediction_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "n_uninformed_factors",
        "batch_uninformed_auc",
        "batch_uninformed_auc_excess",
        "celltype_informed_auc",
        "celltype_informed_auc_excess",
        "celltype_random_baseline_auc",
    ]
    plot_df = (
        metrics_df.loc[metrics_df["n_uninformed_factors"] >= 1, columns]
        .groupby("n_uninformed_factors", as_index=False)
        .agg(
            batch_uninformed_auc_mean=("batch_uninformed_auc", "mean"),
            batch_uninformed_auc_excess_mean=("batch_uninformed_auc_excess", "mean"),
            celltype_informed_auc_mean=("celltype_informed_auc", "mean"),
            celltype_informed_auc_excess_mean=("celltype_informed_auc_excess", "mean"),
            celltype_random_baseline_auc_mean=("celltype_random_baseline_auc", "mean"),
        )
        .sort_values("n_uninformed_factors")
        .reset_index(drop=True)
    )
    return plot_df


def plot_batch_prediction_tradeoff(
    summary_df: pd.DataFrame,
    *,
    path: Path,
    title: str = "Batch predictive power of uninformed subspace across the model sweep",
) -> None:
    if summary_df.empty:
        return

    fig_width = max(7.0, 0.75 * len(summary_df) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    x = np.arange(len(summary_df))

    ax.bar(
        x,
        summary_df["batch_uninformed_auc_mean"],
        color=PRIMARY_COLOR,
        width=0.72,
        label="Batch AUROC from full uninformed subspace",
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.6, label="Random AUROC")
    ax.set_xlabel("Number of uninformed factors (full model)")
    ax.set_ylabel("Batch prediction AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["n_uninformed_factors"].astype(int).astype(str).tolist())
    ax.set_ylim(0.45, min(1.0, max(0.95, float(summary_df["batch_uninformed_auc_mean"].max()) + 0.03)))

    ax2 = ax.twinx()
    ax2.plot(
        x,
        100.0 * summary_df["fraction_of_best_batch_auc_excess"],
        color=QUATERNARY_COLOR,
        marker="o",
        linewidth=2.0,
        label="% of best batch AUROC above random",
    )
    ax2.set_ylabel("% of best batch AUROC above random")
    ax2.set_ylim(0.0, 105.0)

    if (summary_df["n_uninformed_factors"] == 3).any():
        row = summary_df.loc[summary_df["n_uninformed_factors"] == 3].iloc[0]
        x_idx = int(summary_df.index[summary_df["n_uninformed_factors"] == 3][0])
        ax2.annotate(
            f"{100.0 * row['fraction_of_best_batch_auc_excess']:.1f}%",
            xy=(x_idx, 100.0 * row["fraction_of_best_batch_auc_excess"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=QUATERNARY_COLOR,
        )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="center right", frameon=True, fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_batch_celltype_prediction_comparison(
    summary_df: pd.DataFrame,
    *,
    path: Path,
    title: str = "Predictive structure across the model sweep",
) -> None:
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sns.lineplot(
        data=summary_df,
        x="n_uninformed_factors",
        y="batch_uninformed_auc_mean",
        marker="o",
        linewidth=2.0,
        label="Batch from uninformed factors",
        ax=ax,
    )
    sns.lineplot(
        data=summary_df,
        x="n_uninformed_factors",
        y="celltype_informed_auc_mean",
        marker="o",
        linewidth=2.0,
        label="Cell type from informed factors",
        ax=ax,
    )
    baseline = float(summary_df["celltype_random_baseline_auc_mean"].dropna().iloc[0])
    ax.axhline(0.5, color=PRIMARY_COLOR, linestyle="--", linewidth=1.1, alpha=0.5)
    ax.axhline(baseline, color=SECONDARY_COLOR, linestyle="--", linewidth=1.1, alpha=0.5)
    ax.set_xlabel("Number of uninformed factors (full model)")
    ax.set_ylabel("Cross-validated AUROC")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_factor_pareto(
    pareto_df: pd.DataFrame,
    *,
    reference_metrics: dict[str, float],
    path: Path,
    title: str,
    x_label: str,
    bar_label: str,
    cumulative_column: str,
    cumulative_label: str,
    cumulative_ylabel: str,
) -> None:
    if pareto_df.empty:
        return

    fig_width = max(7.0, 0.55 * len(pareto_df) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    x = np.arange(1, len(pareto_df) + 1)
    ax.bar(x, pareto_df["incremental_r2_total"], color=PRIMARY_COLOR, label=bar_label)
    ax.axhline(
        reference_metrics["informed_total_r2"],
        color=SECONDARY_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="Informed-only R2",
    )
    ax.axhline(
        reference_metrics["all_factor_total_r2"],
        color=TERTIARY_COLOR,
        linestyle=":",
        linewidth=1.8,
        label="All-factor R2",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Processed-space R2")
    ax.set_xticks(x)
    xticklabels = pareto_df["display_label"].astype(str).tolist() if "display_label" in pareto_df.columns else [str(value) for value in x]
    ax.set_xticklabels(xticklabels, rotation=0)

    ax2 = ax.twinx()
    cumulative_percent = 100.0 * pareto_df[cumulative_column]
    ax2.plot(
        x,
        cumulative_percent,
        color=QUATERNARY_COLOR,
        marker="o",
        linewidth=2,
        label=cumulative_label,
    )
    ax2.axhline(80.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax2.set_ylabel(cumulative_ylabel)
    ax2.set_ylim(0.0, 105.0)

    if (pareto_df["order"] == 3).any():
        top3_row = pareto_df.loc[pareto_df["order"] == 3].iloc[0]
        x_top3 = pareto_df.index[pareto_df["order"] == 3][0] + 1
        ax2.annotate(
            f"{100.0 * top3_row[cumulative_column]:.1f}%",
            xy=(x_top3, 100.0 * top3_row[cumulative_column]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=10,
        )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="center right",
        frameon=True,
        fontsize=9,
        title_fontsize=9,
        handlelength=1.8,
        borderpad=0.4,
        labelspacing=0.35,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_batch_prediction_pareto(
    pareto_df: pd.DataFrame,
    *,
    reference_metrics: dict[str, Any],
    path: Path,
) -> None:
    if pareto_df.empty:
        return

    fig_width = max(7.0, 0.55 * len(pareto_df) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    x = np.arange(1, len(pareto_df) + 1)

    ax.bar(
        x,
        pareto_df["single_auc_excess"],
        color=PRIMARY_COLOR,
        label="Single-factor batch AUROC above random",
    )
    ax.set_xlabel("Ranked uninformed factors")
    ax.set_ylabel("Single-factor AUROC above 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(pareto_df["display_label"].astype(str).tolist())
    ax.set_ylim(bottom=0.0)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        100.0 * pareto_df["cumulative_fraction_of_full_auc_excess"],
        color=QUATERNARY_COLOR,
        marker="o",
        linewidth=2.0,
        label="Cumulative batch AUROC (% of full 10-factor signal)",
    )
    ax2.set_ylabel("Cumulative batch AUROC (% of full 10-factor signal)")
    ax2.set_ylim(0.0, 105.0)

    if (pareto_df["order"] == 3).any():
        top3_row = pareto_df.loc[pareto_df["order"] == 3].iloc[0]
        x_top3 = pareto_df.index[pareto_df["order"] == 3][0] + 1
        ax2.annotate(
            f"{100.0 * top3_row['cumulative_fraction_of_full_auc_excess']:.1f}%",
            xy=(x_top3, 100.0 * top3_row["cumulative_fraction_of_full_auc_excess"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=QUATERNARY_COLOR,
        )

    full_auc = reference_metrics.get("full_uninformed_auc", np.nan)
    title = "Batch prediction Pareto-style summary from uninformed factors"
    if np.isfinite(full_auc):
        title += f" (full AUROC = {full_auc:.3f})"
    ax.set_title(title)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="center right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    sns.set_theme(style="whitegrid")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    models_dir = out_dir / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    mdata = md.read_h5mu(args.data_path)
    for adata in mdata.mod.values():
        adata.X = adata.X.astype(np.float32)

    gene_set_stats = load_collection_gene_set_stats(
        var_names=mdata.mod["rna"].var_names,
        collection="hallmark",
    )
    mdata = attach_gene_set_mask(mdata, gene_set_stats)

    metadata = {
        "data_path": str(args.data_path),
        "n_obs": int(mdata.n_obs),
        "gene_set_source": DEFAULT_GENE_SET_SOURCE,
        "msigdb_category": HALLMARK_PRESET["msigdb_category"],
        "msigdb_dbver": HALLMARK_PRESET["msigdb_dbver"],
        "gene_set_min_fraction": HALLMARK_PRESET["gene_set_min_fraction"],
        "gene_set_min_count": HALLMARK_PRESET["gene_set_min_count"],
        "gene_set_max_count": HALLMARK_PRESET["gene_set_max_count"],
        "gene_set_similarity_threshold": HALLMARK_PRESET["gene_set_similarity_threshold"],
        "expected_primary_count": HALLMARK_PRESET["expected_primary_count"],
        "expected_mca_count": HALLMARK_PRESET["expected_mca_count"],
        "n_gene_sets": int(len(gene_set_stats)),
        "n_uninformed_grid": args.n_uninformed_grid,
        "seeds": args.seeds,
        "likelihoods": DEFAULT_LIKELIHOODS,
        "weight_prior": DEFAULT_WEIGHT_PRIOR,
        "nonnegative_weights": DEFAULT_NONNEGATIVE_WEIGHTS,
        "nonnegative_factors": DEFAULT_NONNEGATIVE_FACTORS,
        "init_factors": DEFAULT_INIT_FACTORS,
        "init_scale": DEFAULT_INIT_SCALE,
        "lr": DEFAULT_LR,
        "n_particles": DEFAULT_N_PARTICLES,
        "active_r2_threshold": args.active_r2_threshold,
    }
    (out_dir / "resolved_run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    gene_set_stats.assign(features=gene_set_stats["features"].map(json.dumps)).to_csv(
        out_dir / "retained_gene_sets.csv",
        index=False,
    )

    metrics_rows: list[dict[str, Any]] = []
    significant_rows: list[dict[str, Any]] = []
    pareto_df = pd.DataFrame()
    pareto_reference_metrics: dict[str, float] = {}
    all_factor_pareto_df = pd.DataFrame()
    all_factor_pareto_reference_metrics: dict[str, float] = {}
    batch_pareto_df = pd.DataFrame()
    batch_pareto_reference_metrics: dict[str, Any] = {}
    max_uninformed_factors = max(args.n_uninformed_grid) if args.n_uninformed_grid else 0

    for seed in args.seeds:
        for n_uninformed in args.n_uninformed_grid:
            save_path = models_dir / f"seed_{seed:03d}_uninformed_{n_uninformed}.h5" if args.save_models else False
            model = fit_model(
                mdata=mdata,
                n_uninformed_factors=n_uninformed,
                seed=seed,
                save_path=save_path,
            )
            row = summarize_model_sensitivity(
                model,
                seed=seed,
                n_uninformed_factors=n_uninformed,
                alpha=args.alpha,
            ).to_dict()
            row["model_path"] = str(save_path) if args.save_models else ""
            row.update(
                compute_uninformed_batch_prediction_metrics(
                    model,
                    mdata,
                    random_state=seed,
                )
            )
            row.update(
                compute_informed_celltype_prediction_metrics(
                    model,
                    mdata,
                    random_state=seed,
                )
            )

            if args.save_processed_subset_r2:
                for subset_name in ["all", "informed", "uninformed"]:
                    subset_metrics = compute_processed_subset_r2(
                        model,
                        mdata,
                        factor_names=factor_subset_names(model, subset_name),
                    )
                    for key, value in subset_metrics.items():
                        row[f"processed_{subset_name}_{key}"] = value

            metrics_rows.append(row)

            matched = matched_pcgse_activity_table(model).copy()
            n_matched = int(len(matched))
            n_significant = int((matched["padj"] < args.alpha).sum()) if n_matched > 0 else 0
            n_active = int((matched["r2_rna"] >= args.active_r2_threshold).sum()) if n_matched > 0 else 0
            n_active_significant = (
                int(
                    (
                        (matched["padj"] < args.alpha)
                        & (matched["r2_rna"] >= args.active_r2_threshold)
                    ).sum()
                )
                if n_matched > 0
                else 0
            )
            n_active_significant_rna001 = (
                int(
                    (
                        (matched["padj"] < args.alpha)
                        & (matched["r2_rna"] >= DEFAULT_COMPARISON_ACTIVE_R2_THRESHOLD)
                    ).sum()
                )
                if n_matched > 0
                else 0
            )
            n_active_significant_rna_gt0 = (
                int(
                    (
                        (matched["padj"] < args.alpha)
                        & (matched["r2_rna"] > 0.0)
                    ).sum()
                )
                if n_matched > 0
                else 0
            )
            n_active_pathways_rna_prot_gt0 = (
                int(
                    (
                        (matched["r2_rna"] > 0.0)
                        & (matched["r2_prot"] > 0.0)
                    ).sum()
                )
                if n_matched > 0 and {"r2_rna", "r2_prot"}.issubset(matched.columns)
                else 0
            )
            n_active_pathways_rna_prot_gt001 = (
                int(
                    (
                        (matched["r2_rna"] > 0.01)
                        & (matched["r2_prot"] > 0.01)
                    ).sum()
                )
                if n_matched > 0 and {"r2_rna", "r2_prot"}.issubset(matched.columns)
                else 0
            )
            significant_rows.append(
                {
                    "seed": seed,
                    "n_uninformed_factors": n_uninformed,
                    "n_matched_pathways": n_matched,
                    "n_significant_pathways": n_significant,
                    "fraction_significant_pathways": float(n_significant / n_matched) if n_matched > 0 else np.nan,
                    "n_active_pathways": n_active,
                    "fraction_active_pathways": float(n_active / n_matched) if n_matched > 0 else np.nan,
                    "n_active_significant_pathways": n_active_significant,
                    "fraction_active_significant_pathways": (
                        float(n_active_significant / n_matched) if n_matched > 0 else np.nan
                    ),
                    "n_active_significant_pathways_rna001": n_active_significant_rna001,
                    "fraction_active_significant_pathways_rna001": (
                        float(n_active_significant_rna001 / n_matched) if n_matched > 0 else np.nan
                    ),
                    "n_active_significant_pathways_rna_gt0": n_active_significant_rna_gt0,
                    "fraction_active_significant_pathways_rna_gt0": (
                        float(n_active_significant_rna_gt0 / n_matched) if n_matched > 0 else np.nan
                    ),
                    "n_active_pathways_rna_prot_gt0": n_active_pathways_rna_prot_gt0,
                    "fraction_active_pathways_rna_prot_gt0": (
                        float(n_active_pathways_rna_prot_gt0 / n_matched) if n_matched > 0 else np.nan
                    ),
                    "n_active_pathways_rna_prot_gt001": n_active_pathways_rna_prot_gt001,
                    "fraction_active_pathways_rna_prot_gt001": (
                        float(n_active_pathways_rna_prot_gt001 / n_matched) if n_matched > 0 else np.nan
                    ),
                }
            )

            if n_uninformed == max_uninformed_factors:
                pareto_df, pareto_reference_metrics = compute_uninformed_pareto(model, mdata)
                all_factor_pareto_df, all_factor_pareto_reference_metrics = compute_all_factor_pareto(model, mdata)
                batch_pareto_df, batch_pareto_reference_metrics = compute_batch_prediction_pareto(
                    model,
                    mdata,
                    factor_names=factor_subset_names(model, "uninformed"),
                    random_state=seed,
                )

            del model
            gc.collect()

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["n_uninformed_factors", "seed"]).reset_index(drop=True)
    summary_df = aggregate_sweep_metrics(metrics_df)
    if "uninformed_r2_sum_mean" in summary_df and "informed_r2_sum_mean" in summary_df:
        summary_df["explained_r2_other_mean"] = (
            summary_df["total_r2_sum_mean"]
            - summary_df["uninformed_r2_sum_mean"]
            - summary_df["informed_r2_sum_mean"]
        )

    significant_df = pd.DataFrame(significant_rows).sort_values(["n_uninformed_factors", "seed"]).reset_index(drop=True)
    significant_summary_df = aggregate_significant_pathways(significant_df)
    uninformed_share_summary_df = summarize_uninformed_share_of_all_explained(metrics_df)
    batch_prediction_summary_df = summarize_batch_prediction_tradeoff(metrics_df)
    prediction_comparison_df = summarize_prediction_comparison(metrics_df)
    tradeoff_df = (
        metrics_df.merge(
            significant_df[
                [
                    "seed",
                    "n_uninformed_factors",
                    "n_significant_pathways",
                    "fraction_significant_pathways",
                ]
            ],
            on=["seed", "n_uninformed_factors"],
            how="left",
        )
        .sort_values(["n_uninformed_factors", "seed"])
        .reset_index(drop=True)
    )

    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    significant_df.to_csv(out_dir / "significant_pathways.csv", index=False)
    significant_summary_df.to_csv(out_dir / "significant_pathways_summary.csv", index=False)
    uninformed_share_summary_df.to_csv(out_dir / "uninformed_share_of_all_explained.csv", index=False)
    batch_prediction_summary_df.to_csv(out_dir / "batch_prediction_pareto.csv", index=False)
    prediction_comparison_df.to_csv(out_dir / "prediction_comparison.csv", index=False)
    tradeoff_df.to_csv(out_dir / "uninformed_tradeoff_1to10.csv", index=False)

    plot_main_summary(summary_df, path=plots_dir / "summary.png")
    plot_significant_pathways(significant_summary_df, path=plots_dir / "significant_pathways.png")
    plot_uninformed_share_of_all_explained(
        uninformed_share_summary_df,
        path=plots_dir / "uninformed_share_of_all_explained.png",
    )
    if not batch_prediction_summary_df.empty:
        batch_series = _resolve_batch_series(mdata)
        batch_reference = {
            "batch_column": str(batch_series.name),
            "batch_levels": sorted(batch_series.dropna().astype(str).unique().tolist()),
            "best_batch_uninformed_auc": float(batch_prediction_summary_df["batch_uninformed_auc_mean"].max()),
            "best_batch_uninformed_auc_excess": float(batch_prediction_summary_df["batch_uninformed_auc_excess_mean"].max()),
        }
        (out_dir / "batch_prediction_pareto_reference.json").write_text(
            json.dumps(batch_reference, indent=2, sort_keys=True)
        )
        plot_batch_prediction_tradeoff(
            batch_prediction_summary_df,
            path=plots_dir / "batch_prediction_pareto.png",
        )
    if not prediction_comparison_df.empty:
        plot_batch_celltype_prediction_comparison(
            prediction_comparison_df,
            path=plots_dir / "batch_celltype_prediction_comparison.png",
        )
    plot_tradeoff_bars(
        tradeoff_df.loc[tradeoff_df["n_uninformed_factors"] >= 1].copy(),
        path=plots_dir / "uninformed_tradeoff_1to10.png",
        value_column="processed_uninformed_r2_total",
        value_label="Explained variance by uninformed factors (subspace R2)",
        title="Uninformed-factor explained variance and pathway significance",
    )
    plot_tradeoff_bars(
        tradeoff_df.loc[tradeoff_df["n_uninformed_factors"] >= 1].copy(),
        path=plots_dir / "uninformed_tradeoff_1to10_rna.png",
        value_column="processed_uninformed_r2_rna",
        value_label="Explained variance by uninformed factors in RNA (subspace R2)",
        title="RNA explained variance by uninformed factors and pathway significance",
    )
    plot_tradeoff_bars(
        tradeoff_df.loc[tradeoff_df["n_uninformed_factors"] >= 1].copy(),
        path=plots_dir / "uninformed_tradeoff_1to10_active_only.png",
        value_column="processed_uninformed_r2_total",
        value_label="Explained variance by uninformed factors (subspace R2)",
        title="Uninformed-factor explained variance and RNA-active pathways",
        count_column="n_active_pathways",
        count_label="Active pathways",
        count_ylabel="Active pathways (r2_rna >= 0.01)",
    )
    plot_tradeoff_bars(
        tradeoff_df.loc[tradeoff_df["n_uninformed_factors"] >= 1].copy(),
        path=plots_dir / "total_tradeoff_1to10.png",
        value_column="processed_all_r2_total",
        value_label="Total explained variance (subspace R2)",
        title="Total explained variance and pathway significance",
    )
    plot_tradeoff_bars(
        tradeoff_df.loc[tradeoff_df["n_uninformed_factors"] >= 1].copy(),
        path=plots_dir / "total_tradeoff_1to10_active_only.png",
        value_column="processed_all_r2_total",
        value_label="Total explained variance (subspace R2)",
        title="Total explained variance and RNA-active pathways",
        count_column="n_active_pathways",
        count_label="Active pathways",
        count_ylabel="Active pathways (r2_rna >= 0.01)",
    )
    plot_tradeoff_bars(
        tradeoff_df.loc[tradeoff_df["n_uninformed_factors"] >= 1].copy(),
        path=plots_dir / "total_tradeoff_1to10_rna.png",
        value_column="processed_all_r2_rna",
        value_label="Total explained variance in RNA (subspace R2)",
        title="Total RNA explained variance and pathway significance",
    )
    plot_factor_r2_split_bars(
        metrics_df,
        path=plots_dir / "factor_r2_split_1to10_total.png",
        uninformed_column="uninformed_r2_sum",
        informed_column="informed_r2_sum",
        ylabel="Summed factor R2",
        title="Informed and uninformed factor R2 across the model sweep",
    )
    plot_factor_r2_split_bars(
        metrics_df,
        path=plots_dir / "factor_r2_split_1to10.png",
        uninformed_column="processed_uninformed_r2_rna",
        informed_column="processed_informed_r2_rna",
        ylabel="RNA subspace R2",
        title="RNA informed and uninformed subspace R2 across the model sweep",
        tradeoff_df=tradeoff_df,
        count_column="n_active_pathways",
        count_label="Active pathways",
        count_ylabel="Active pathways (r2_rna >= 0.01)",
    )
    plot_factor_r2_split_bars(
        metrics_df,
        path=plots_dir / "factor_r2_split_1to10_pcgse.png",
        uninformed_column="processed_uninformed_r2_rna",
        informed_column="processed_informed_r2_rna",
        ylabel="RNA subspace R2",
        title="RNA informed and uninformed subspace R2 with PCGSE significance",
        tradeoff_df=tradeoff_df,
        count_column="n_significant_pathways",
        count_label="Significant pathways",
        count_ylabel="Significant pathways (PCGSE padj < 0.05)",
    )
    plot_factor_r2_split_bars(
        metrics_df,
        path=plots_dir / "factor_r2_split_1to10_pcgse_rna001.png",
        uninformed_column="processed_uninformed_r2_rna",
        informed_column="processed_informed_r2_rna",
        ylabel="RNA subspace R2",
        title="RNA informed and uninformed subspace R2 with PCGSE and r2_rna >= 0.001",
        tradeoff_df=tradeoff_df,
        count_column="n_active_significant_pathways_rna001",
        count_label="PCGSE + RNA-active pathways",
        count_ylabel="Pathways (PCGSE padj < 0.05 and r2_rna >= 0.001)",
    )
    plot_factor_r2_split_bars(
        metrics_df,
        path=plots_dir / "factor_r2_split_1to10_pcgse_rna_gt0.png",
        uninformed_column="processed_uninformed_r2_rna",
        informed_column="processed_informed_r2_rna",
        ylabel="RNA subspace R2",
        title="RNA informed and uninformed subspace R2 with PCGSE and r2_rna > 0",
        tradeoff_df=tradeoff_df,
        count_column="n_active_significant_pathways_rna_gt0",
        count_label="PCGSE + RNA-active pathways",
        count_ylabel="Pathways (PCGSE padj < 0.05 and r2_rna > 0)",
    )
    plot_factor_r2_split_bars(
        metrics_df,
        path=plots_dir / "factor_r2_split_1to10_rna_prot_gt0.png",
        uninformed_column="processed_uninformed_r2_rna",
        informed_column="processed_informed_r2_rna",
        ylabel="RNA subspace R2",
        title="RNA informed and uninformed subspace R2 with r2_rna > 0 and r2_prot > 0",
        tradeoff_df=tradeoff_df,
        count_column="n_active_pathways_rna_prot_gt0",
        count_label="RNA+Protein-active pathways",
        count_ylabel="Pathways (r2_rna > 0 and r2_prot > 0)",
    )
    plot_factor_r2_split_bars(
        metrics_df,
        path=plots_dir / "factor_r2_split_1to10_rna_prot_gt001.png",
        uninformed_column="processed_uninformed_r2_rna",
        informed_column="processed_informed_r2_rna",
        ylabel="RNA subspace R2",
        title="RNA informed and uninformed subspace R2 with r2_rna > 0.01 and r2_prot > 0.01",
        tradeoff_df=tradeoff_df,
        count_column="n_active_pathways_rna_prot_gt001",
        count_label="RNA+Protein-active pathways",
        count_ylabel="Pathways (r2_rna > 0.01 and r2_prot > 0.01)",
    )
    if not pareto_df.empty:
        pareto_df.to_csv(out_dir / "uninformed_pareto.csv", index=False)
        (out_dir / "uninformed_pareto_reference.json").write_text(
            json.dumps(pareto_reference_metrics, indent=2, sort_keys=True)
        )
        plot_factor_pareto(
            pareto_df,
            reference_metrics=pareto_reference_metrics,
            path=plots_dir / "uninformed_pareto.png",
            title=f"Pareto view of uninformed-factor reconstruction (n={max_uninformed_factors})",
            x_label="Top uninformed factors",
            bar_label="Incremental uninformed R2",
            cumulative_column="cumulative_fraction_of_selected_subset_total",
            cumulative_label="Cumulative % of uninformed explained variance",
            cumulative_ylabel="Cumulative uninformed explained variance (%)",
        )
    if not all_factor_pareto_df.empty:
        all_factor_pareto_df.to_csv(out_dir / "all_factor_pareto.csv", index=False)
        (out_dir / "all_factor_pareto_reference.json").write_text(
            json.dumps(all_factor_pareto_reference_metrics, indent=2, sort_keys=True)
        )
        plot_factor_pareto(
            all_factor_pareto_df,
            reference_metrics=all_factor_pareto_reference_metrics,
            path=plots_dir / "all_factor_pareto.png",
            title=f"Pareto view of all-factor reconstruction (n={max_uninformed_factors})",
            x_label="Top ranked factors",
            bar_label="Incremental factor R2",
            cumulative_column="cumulative_fraction_of_all_model_total",
            cumulative_label="Cumulative % of full-model explained variance",
            cumulative_ylabel="Cumulative full-model explained variance (%)",
        )
    if not batch_pareto_df.empty:
        batch_pareto_df.to_csv(out_dir / "batch_prediction_pareto.csv", index=False)
        (out_dir / "batch_prediction_pareto_reference.json").write_text(
            json.dumps(batch_pareto_reference_metrics, indent=2, sort_keys=True)
        )
        plot_batch_prediction_pareto(
            batch_pareto_df,
            reference_metrics=batch_pareto_reference_metrics,
            path=plots_dir / "batch_prediction_pareto.png",
        )

    return {
        "out_dir": str(out_dir),
        "n_gene_sets": int(len(gene_set_stats)),
        "n_runs": int(len(metrics_df)),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_sweep(args)
    print(f"Finished sweep in {result['out_dir']}")
    print(f"n_gene_sets={result['n_gene_sets']}")
    print(f"n_runs={result['n_runs']}")


if __name__ == "__main__":
    main()
