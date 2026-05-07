from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .plot_style import (
    CAT_PALETTE,
    SEQUENTIAL_CMAP,
    clean_ax,
    savefig,
    set_house_style,
)


def pretty_noise(value: float) -> str:
    return f"{value:.2f}"


def pretty_pathway_name(pathway_name: str) -> str:
    payload = re.sub(r"^(MCA::|HAN[_\s]+)", "", str(pathway_name).strip(), flags=re.IGNORECASE)
    payload = re.sub(r"[_\s]+", " ", payload).strip().title()
    payload = payload.replace("Nk ", "NK ")
    return f"{payload} [M]"


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def aggregate_with_errorbars(df: pd.DataFrame, *, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    aggregated = df.groupby(group_col)[value_cols].agg(["mean", "std", "count"]).reset_index()
    aggregated.columns = [
        group_col if col[0] == group_col else f"{col[0]}_{col[1]}"
        for col in aggregated.columns.to_flat_index()
    ]
    for value_col in value_cols:
        std_col = f"{value_col}_std"
        if std_col in aggregated.columns:
            aggregated[std_col] = aggregated[std_col].fillna(0.0)
    return aggregated.sort_values(group_col).reset_index(drop=True)


def save_metric_grid_plot(
    data_df: pd.DataFrame,
    *,
    x_col: str,
    panels: list[tuple[str, str, str]],
    path: Path,
    title: str,
    xlabel: str = "Noise level (FPR = FNR)",
    errorbars: bool = False,
) -> None:
    if data_df.empty:
        return

    set_house_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    x = data_df[x_col]
    palette = sns.color_palette(CAT_PALETTE, n_colors=len(panels))
    for ax, (column, ylabel, color) in zip(axes.flat, panels, strict=True):
        color = color or palette[0]
        if errorbars:
            ax.errorbar(
                x,
                data_df[f"{column}_mean"],
                yerr=data_df[f"{column}_std"],
                marker="o",
                linewidth=2,
                elinewidth=1.2,
                capsize=3,
                color=color,
            )
        else:
            ax.plot(x, data_df[column], marker="o", linewidth=2, color=color)
        ax.set_ylabel(ylabel)
        clean_ax(ax)
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 1].set_xlabel(xlabel)
    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    savefig(fig, path)
    plt.close(fig)


def save_heatmap(
    matrix: pd.DataFrame,
    *,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (6.0, 5.0),
    cmap: str = SEQUENTIAL_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "Pearson",
) -> None:
    set_house_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_ax(ax)
    fig.tight_layout()
    savefig(fig, path)
    plt.close(fig)
