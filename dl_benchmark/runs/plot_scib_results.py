import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.plot_style import CAT_PALETTE, clean_ax, savefig, set_house_style


def _load_table(path: str) -> pd.DataFrame:
    path = os.path.expanduser(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise ValueError("Input must be .csv or .json")


def _coerce_numeric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out[out[metric].notna()].copy()
    if "Embedding" in out.columns:
        out = out[out["Embedding"] != "Metric Type"].copy()
    return out


def main(args):
    df = _load_table(args.input)
    if args.metric not in df.columns:
        raise KeyError(f"Missing metric column '{args.metric}' in input table.")
    if "Embedding" not in df.columns:
        raise KeyError("Expected 'Embedding' column in scIB benchmark output.")

    df = _coerce_numeric(df, args.metric)
    df = df[["Embedding", args.metric]].copy()
    df = df.sort_values(args.metric, ascending=True).reset_index(drop=True)

    out_png = os.path.expanduser(args.out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    set_house_style()
    fig, ax = plt.subplots(figsize=(max(4.5, 0.55 * len(df)), 3.5))
    bars = sns.barplot(
        data=df,
        x="Embedding",
        y=args.metric,
        hue="Embedding",
        dodge=False,
        legend=False,
        ax=ax,
        palette=CAT_PALETTE,
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(args.metric)
    ax.set_ylim(0, max(1.0, float(df[args.metric].max()) * 1.08))
    ax.tick_params(axis="x", rotation=35)
    clean_ax(ax)

    for patch, val in zip(bars.patches, df[args.metric]):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            patch.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    savefig(fig, out_png)
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="scIB benchmark output (.csv or .json).")
    p.add_argument("--metric", default="Bio conservation")
    p.add_argument("--out-png", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()
    main(args)
