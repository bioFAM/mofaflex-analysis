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
from sklearn.metrics import roc_auc_score

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.mca_filter_sensitivity import load_sln208_mdata, prettify_pathway_name
    from mofaflex_sensitivity.plot_style import CAT_PALETTE, clean_ax, place_legend, savefig, set_house_style
    from mofaflex_sensitivity.sln208_prior_noise_refinement import DEFAULT_DATA_PATH
    from mofaflex_sensitivity.uninformed_sensitivity import matched_pcgse_activity_table
else:
    from .mca_filter_sensitivity import load_sln208_mdata, prettify_pathway_name
    from .plot_style import CAT_PALETTE, clean_ax, place_legend, savefig, set_house_style
    from .sln208_prior_noise_refinement import DEFAULT_DATA_PATH
    from .uninformed_sensitivity import matched_pcgse_activity_table


DEFAULT_SWEEP_DIR = Path("artifacts/sln208_prior_confidence_refinement_sweep_mca_5seeds")
DEFAULT_OUTPUT_SUBDIR = Path("plots/confidence_factor_quality")
DEFAULT_PCGSE_ALPHA = 0.05

CELL_TYPE_AUROC_MAP = {
    "HAN_MARGINAL ZONE B CELL": ("B",),
    "HAN_T CELL": ("CD4", "CD8"),
    "HAN_DENDRITIC CELL_S100A4 HIGH": ("DC",),
    "HAN_NK CELL": ("NK",),
}


def pretty_mca_name(pathway_name: str) -> str:
    name = str(pathway_name)
    return prettify_pathway_name(name if name.startswith("MCA::") else f"MCA::{name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize SLN208 prior-confidence MCA factor quality, variance, overwrite, and cell-type AUROC."
    )
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cell-type-column", default="cell_types (high)")
    parser.add_argument(
        "--pcgse-alpha",
        type=float,
        default=DEFAULT_PCGSE_ALPHA,
        help="Matched PCGSE adjusted-p-value threshold. MCA factors with padj >= alpha are counted as overwritten.",
    )
    return parser


def confidence_from_path(path: Path) -> float:
    for part in path.parts:
        match = re.fullmatch(r"confidence_(.+)", part)
        if match:
            return float(match.group(1).replace("p", "."))
    raise ValueError(f"Could not parse confidence from path: {path}")


def seed_from_path(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not parse seed from path: {path}")


def pretty_confidence(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def mca_refinement_table(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(sweep_dir.glob("confidence_*/seed_*/pathway_refinement.csv")):
        df = pd.read_csv(path)
        df = df.loc[df["source"].eq("mca")].copy()
        df.insert(0, "annotation_confidence", confidence_from_path(path))
        df.insert(1, "seed", seed_from_path(path))
        df["pathway_pretty"] = df["pathway_name"].map(pretty_mca_name)
        df["prior_drift_score"] = 1.0 - df["top_true_size_jaccard_true"].astype(float)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No pathway_refinement.csv files found under {sweep_dir}")
    return pd.concat(rows, ignore_index=True)


def mca_variance_table(sweep_dir: Path) -> pd.DataFrame:
    path = sweep_dir / "mca_pathway_variance_by_model.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MCA variance table: {path}")
    df = pd.read_csv(path)
    df["pathway_pretty"] = df["pathway_name"].map(pretty_mca_name)
    return df


def load_cell_type_labels(data_path: Path, cell_type_column: str) -> pd.Series:
    mdata = load_sln208_mdata(data_path)
    rna_obs = mdata.mod["rna"].obs
    if cell_type_column not in rna_obs.columns:
        raise KeyError(f"{cell_type_column!r} not found in RNA obs columns: {list(rna_obs.columns)}")
    return rna_obs[cell_type_column].astype(str).reset_index(drop=True)


def model_paths(sweep_dir: Path) -> list[Path]:
    paths = sorted(sweep_dir.glob("confidence_*/seed_*/models/*.h5"))
    if not paths:
        paths = sorted((sweep_dir / "models").glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No saved MOFA-FLEX models found under {sweep_dir}")
    return paths


def mca_pcgse_table(sweep_dir: Path, *, alpha: float) -> pd.DataFrame:
    import mofaflex as mfl

    rows: list[pd.DataFrame] = []
    for path in model_paths(sweep_dir):
        confidence = confidence_from_path(path)
        seed = seed_from_path(path)
        model = mfl.MOFAFLEX.load(path, map_location="cpu")
        pcgse = matched_pcgse_activity_table(model).copy()
        if pcgse.empty:
            continue
        pcgse = pcgse.loc[
            pcgse["view"].astype(str).eq("rna") & pcgse["factor"].astype(str).str.startswith("HAN_")
        ].copy()
        if pcgse.empty:
            continue
        pcgse.insert(0, "annotation_confidence", confidence)
        pcgse.insert(1, "seed", seed)
        pcgse["pathway_name"] = pcgse["factor"].astype(str)
        pcgse["pathway_pretty"] = pcgse["pathway_name"].map(pretty_mca_name)
        pcgse["padj"] = pd.to_numeric(pcgse["padj"], errors="coerce")
        pcgse["p"] = pd.to_numeric(pcgse["p"], errors="coerce")
        pcgse["pcgse_significant"] = pcgse["padj"].lt(float(alpha))
        pcgse["is_overwritten"] = ~pcgse["pcgse_significant"]
        pcgse["pcgse_neglog10_padj"] = -np.log10(pcgse["padj"].clip(lower=np.finfo(float).tiny))
        pcgse["pcgse_alpha"] = float(alpha)
        pcgse["model_path"] = str(path)
        rows.append(pcgse)
    if not rows:
        raise FileNotFoundError(f"No matched MCA PCGSE rows could be read from saved models under {sweep_dir}")
    return pd.concat(rows, ignore_index=True)


def celltype_auroc_table(sweep_dir: Path, labels: pd.Series) -> pd.DataFrame:
    import mofaflex as mfl

    rows: list[dict[str, Any]] = []
    label_values = labels.to_numpy(dtype=str)
    for path in model_paths(sweep_dir):
        confidence = confidence_from_path(path)
        seed = seed_from_path(path)
        model = mfl.MOFAFLEX.load(path, map_location="cpu")
        factors_by_group = model.get_factors(return_type="pandas", ordered=False)
        factors = next(iter(factors_by_group.values())) if isinstance(factors_by_group, dict) else factors_by_group
        factors = factors.reset_index(drop=True)
        if len(factors) != len(label_values):
            raise ValueError(f"Model {path} has {len(factors)} cells, but labels have {len(label_values)} cells.")

        for pathway_name, positive_cell_types in CELL_TYPE_AUROC_MAP.items():
            if pathway_name not in factors.columns:
                continue
            y_true = np.isin(label_values, list(positive_cell_types)).astype(int)
            if y_true.min() == y_true.max():
                auroc = np.nan
            else:
                scores = factors[pathway_name].to_numpy(dtype=float)
                auroc = float(roc_auc_score(y_true, scores))
            rows.append(
                {
                    "annotation_confidence": confidence,
                    "seed": seed,
                    "pathway_name": pathway_name,
                    "pathway_pretty": pretty_mca_name(pathway_name),
                    "positive_cell_types": ",".join(positive_cell_types),
                    "cell_type_auroc": auroc,
                    "cell_type_auroc_oriented": max(auroc, 1.0 - auroc) if np.isfinite(auroc) else np.nan,
                    "n_positive_cells": int(y_true.sum()),
                    "n_cells": int(len(y_true)),
                    "model_path": str(path),
                }
            )
    return pd.DataFrame.from_records(rows)


def aggregate_mean_sd(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    pieces = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for value_col in value_cols:
        agg = grouped[value_col].agg(["mean", "std", "count"]).rename(
            columns={
                "mean": f"{value_col}_mean",
                "std": f"{value_col}_sd",
                "count": f"{value_col}_count",
            }
        )
        pieces.append(agg)
    return pd.concat(pieces, axis=1).reset_index()


def save_metric_lineplot(
    df: pd.DataFrame,
    *,
    path: Path,
    y: str,
    yerr: str | None,
    ylabel: str,
    hue: str | None = None,
    title: str | None = None,
    legend_mode: str = "outside",
) -> None:
    set_house_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    xticks = sorted(df["annotation_confidence"].dropna().unique())
    if hue is None:
        ax.errorbar(
            df["annotation_confidence"],
            df[y],
            yerr=df[yerr] if yerr is not None and yerr in df.columns else None,
            marker="o",
            linewidth=1.4,
            capsize=2.0,
            color=sns.color_palette(CAT_PALETTE, n_colors=1)[0],
        )
    else:
        palette = dict(zip(sorted(df[hue].unique()), sns.color_palette(CAT_PALETTE, n_colors=df[hue].nunique()), strict=False))
        for value, group in df.groupby(hue, sort=False):
            ax.errorbar(
                group["annotation_confidence"],
                group[y],
                yerr=group[yerr] if yerr is not None and yerr in group.columns else None,
                marker="o",
                linewidth=1.3,
                capsize=2.0,
                color=palette[value],
                label=value,
            )
        if legend_mode == "outside_center_right":
            ax.legend(
                title=None,
                frameon=False,
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                borderaxespad=0,
            )
        else:
            place_legend(ax, mode=legend_mode, title=None, loc="center right")
    ax.set_xlabel("Prior confidence")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels([pretty_confidence(value) for value in xticks])
    if title:
        ax.set_title(title)
    clean_ax(ax)
    savefig(fig, path)
    plt.close(fig)


def save_overwrite_count_plot(summary: pd.DataFrame, *, path: Path) -> None:
    save_metric_lineplot(
        summary,
        path=path,
        y="n_pcgse_significant_mean",
        yerr="n_pcgse_significant_sd",
        ylabel="Significant MCA gene programs",
        title=None,
    )


def save_celltype_auroc_plot(summary: pd.DataFrame, *, path: Path) -> None:
    save_metric_lineplot(
        summary,
        path=path,
        y="cell_type_auroc_mean",
        yerr="cell_type_auroc_sd",
        ylabel="Cell-type AUROC",
        hue="pathway_pretty",
        title=None,
        legend_mode="outside_center_right",
    )


def save_pareto_plot(df: pd.DataFrame, *, path: Path, y: str, ylabel: str, size_col: str | None = None) -> None:
    set_house_style()
    plot_df = df.dropna(subset=["variance_explained_mean", y]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    sizes = None
    if size_col is not None and size_col in plot_df.columns:
        sizes = 25 + 160 * (plot_df[size_col] - plot_df[size_col].min()) / max(
            float(plot_df[size_col].max() - plot_df[size_col].min()), 1e-12
        )
    scatter = ax.scatter(
        plot_df["variance_explained_mean"],
        plot_df[y],
        s=sizes if sizes is not None else 55,
        c=plot_df["annotation_confidence"],
        cmap="crest",
        edgecolor="0.25",
        linewidth=0.4,
        alpha=0.9,
    )
    for row in plot_df.itertuples(index=False):
        if row.annotation_confidence in {plot_df["annotation_confidence"].min(), plot_df["annotation_confidence"].max()}:
            ax.text(
                float(row.variance_explained_mean),
                float(getattr(row, y)),
                f" {pretty_confidence(row.annotation_confidence)}",
                fontsize=7,
                ha="left",
                va="center",
            )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Prior confidence")
    ax.set_xlabel("Mean MCA variance explained")
    ax.set_ylabel(ylabel)
    clean_ax(ax)
    savefig(fig, path)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    sweep_dir = args.sweep_dir.resolve()
    out_dir = (args.out_dir or (sweep_dir / DEFAULT_OUTPUT_SUBDIR)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    refinement = mca_refinement_table(sweep_dir)
    variance = mca_variance_table(sweep_dir)
    labels = load_cell_type_labels(args.data_path, args.cell_type_column)
    pcgse = mca_pcgse_table(sweep_dir, alpha=args.pcgse_alpha)
    auroc = celltype_auroc_table(sweep_dir, labels)

    refinement.to_csv(out_dir / "mca_refinement_by_pathway_seed.csv", index=False)
    variance.to_csv(out_dir / "mca_variance_by_pathway_seed.csv", index=False)
    pcgse.to_csv(out_dir / "mca_pcgse_by_pathway_seed.csv", index=False)
    auroc.to_csv(out_dir / "mca_celltype_auroc_by_pathway_seed.csv", index=False)

    quality_summary = aggregate_mean_sd(
        refinement,
        ["annotation_confidence"],
        ["average_precision_true", "top_true_size_jaccard_true", "prior_drift_score"],
    )
    overwrite_counts = (
        pcgse.groupby(["annotation_confidence", "seed"], as_index=False)
        .agg(
            n_overwritten=("is_overwritten", "sum"),
            fraction_overwritten=("is_overwritten", "mean"),
            n_pcgse_significant=("pcgse_significant", "sum"),
            fraction_pcgse_significant=("pcgse_significant", "mean"),
            mean_pcgse_neglog10_padj=("pcgse_neglog10_padj", "mean"),
        )
    )
    overwrite_summary = aggregate_mean_sd(
        overwrite_counts,
        ["annotation_confidence"],
        [
            "n_overwritten",
            "fraction_overwritten",
            "n_pcgse_significant",
            "fraction_pcgse_significant",
            "mean_pcgse_neglog10_padj",
        ],
    )
    variance_summary = aggregate_mean_sd(variance, ["annotation_confidence"], ["variance_explained"])
    celltype_summary = aggregate_mean_sd(
        auroc,
        ["annotation_confidence", "pathway_name", "pathway_pretty", "positive_cell_types"],
        ["cell_type_auroc", "cell_type_auroc_oriented"],
    )
    celltype_overall = aggregate_mean_sd(auroc, ["annotation_confidence"], ["cell_type_auroc", "cell_type_auroc_oriented"])

    quality_summary.to_csv(out_dir / "mca_factor_quality_by_confidence.csv", index=False)
    overwrite_counts.to_csv(out_dir / "mca_overwrite_counts_by_seed.csv", index=False)
    overwrite_summary.to_csv(out_dir / "mca_overwrite_counts_by_confidence.csv", index=False)
    variance_summary.to_csv(out_dir / "mca_variance_by_confidence.csv", index=False)
    celltype_summary.to_csv(out_dir / "mca_celltype_auroc_by_confidence_pathway.csv", index=False)
    celltype_overall.to_csv(out_dir / "mca_celltype_auroc_by_confidence_overall.csv", index=False)

    merged = (
        quality_summary.merge(variance_summary, on="annotation_confidence", how="left")
        .merge(overwrite_summary, on="annotation_confidence", how="left")
        .merge(celltype_overall, on="annotation_confidence", how="left", suffixes=("", "_celltype"))
    )
    merged.to_csv(out_dir / "mca_confidence_quality_variance_overwrite_pareto.csv", index=False)

    save_metric_lineplot(
        quality_summary,
        path=out_dir / "mca_aupr_vs_confidence.png",
        y="average_precision_true_mean",
        yerr="average_precision_true_sd",
        ylabel="Mean MCA AUPR vs prior genes",
    )
    save_metric_lineplot(
        quality_summary,
        path=out_dir / "mca_prior_jaccard_vs_confidence.png",
        y="top_true_size_jaccard_true_mean",
        yerr="top_true_size_jaccard_true_sd",
        ylabel="Top-size Jaccard vs prior genes",
    )
    save_overwrite_count_plot(overwrite_summary, path=out_dir / "mca_overwrite_count_vs_confidence.png")
    save_metric_lineplot(
        variance_summary,
        path=out_dir / "mca_variance_vs_confidence.png",
        y="variance_explained_mean",
        yerr="variance_explained_sd",
        ylabel="Mean MCA variance explained",
    )
    save_celltype_auroc_plot(celltype_summary, path=out_dir / "mca_celltype_auroc_vs_confidence.png")
    save_metric_lineplot(
        celltype_overall,
        path=out_dir / "mca_celltype_auroc_overall_vs_confidence.png",
        y="cell_type_auroc_mean",
        yerr="cell_type_auroc_sd",
        ylabel="Mean cell-type AUROC",
    )
    save_pareto_plot(
        merged,
        path=out_dir / "pareto_mca_aupr_vs_variance_overwrite.png",
        y="average_precision_true_mean",
        ylabel="Mean MCA AUPR vs prior genes",
        size_col="fraction_overwritten_mean",
    )
    save_pareto_plot(
        merged,
        path=out_dir / "pareto_mca_celltype_auroc_vs_variance_overwrite.png",
        y="cell_type_auroc_mean",
        ylabel="Mean mapped cell-type AUROC",
        size_col="fraction_overwritten_mean",
    )

    (out_dir / "resolved_analysis.json").write_text(
        json.dumps(
            {
                "sweep_dir": str(sweep_dir),
                "data_path": str(args.data_path),
                "cell_type_column": args.cell_type_column,
                "cell_type_auroc_map": CELL_TYPE_AUROC_MAP,
                "pcgse_alpha": args.pcgse_alpha,
                "overwrite_definition": (
                    "MCA program is counted as overwritten when its matched RNA PCGSE "
                    f"adjusted p-value is >= {args.pcgse_alpha}."
                ),
                "prior_drift_definition": "1 - top_true_size_jaccard_true; used only as a diagnostic, not as overwrite.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote prior-confidence factor quality analysis to {out_dir}")


if __name__ == "__main__":
    main()
