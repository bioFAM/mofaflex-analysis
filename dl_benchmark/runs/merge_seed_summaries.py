import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_csv(path: str) -> pd.DataFrame:
    path = os.path.expanduser(path)
    return pd.read_csv(path)


def _apply_rename(df: pd.DataFrame, old: str | None, new: str | None) -> pd.DataFrame:
    out = df.copy()
    if old and new and "Embedding" in out.columns:
        out["Embedding"] = out["Embedding"].replace({old: new})
    return out


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in {"Embedding", "seed"}]
    grouped = df.groupby("Embedding")[numeric_cols].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = [
        "Embedding" if col[0] == "Embedding" else f"{col[0]}_{col[1]}"
        for col in grouped.columns
    ]
    return grouped


def _plot_metric(summary: pd.DataFrame, metric: str, out_png: str, title: str | None):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    plot_df = summary[["Embedding", mean_col]].copy().rename(columns={mean_col: "mean"})
    plot_df["std"] = summary[std_col].fillna(0.0)
    plot_df = plot_df.sort_values("mean", ascending=True).reset_index(drop=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(plot_df)), 5.2))
    bars = sns.barplot(
        data=plot_df,
        x="Embedding",
        y="mean",
        hue="Embedding",
        dodge=False,
        legend=False,
        ax=ax,
        palette="deep",
    )
    ax.errorbar(
        x=list(range(len(plot_df))),
        y=plot_df["mean"].to_numpy(),
        yerr=plot_df["std"].to_numpy(),
        fmt="none",
        ecolor="black",
        elinewidth=1.3,
        capsize=4,
        capthick=1.3,
        zorder=5,
    )
    ax.set_title(title or f"{metric} across seeds")
    ax.set_xlabel("")
    ax.set_ylabel(f"{metric} (mean ± sd)")
    ymax = float((plot_df["mean"] + plot_df["std"]).max()) if len(plot_df) else 1.0
    ax.set_ylim(0, max(1.0, ymax * 1.1))
    ax.tick_params(axis="x", rotation=25)

    for patch, (_, row) in zip(bars.patches, plot_df.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            patch.get_height() + float(row["std"]) + 0.005,
            f"{row['mean']:.3f}\n±{row['std']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out_png = os.path.expanduser(out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main(args):
    base_combined = _load_csv(args.base_combined)
    add_combined = _load_csv(args.add_combined)
    add_combined = _apply_rename(add_combined, args.rename_from, args.rename_to)

    merged_combined = pd.concat([base_combined, add_combined], ignore_index=True)
    merged_summary = _aggregate(merged_combined)

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    combined_csv = os.path.join(out_dir, "seed_metrics_combined.csv")
    summary_csv = os.path.join(out_dir, "seed_metrics_summary.csv")
    summary_json = os.path.join(out_dir, "seed_metrics_summary.json")

    merged_combined.to_csv(combined_csv, index=False)
    merged_summary.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(merged_summary.to_dict(orient="records"), f, indent=2)

    for metric in args.metric:
        safe_metric = metric.lower().replace(" ", "_").replace("-", "_")
        _plot_metric(
            merged_summary,
            metric=metric,
            out_png=os.path.join(out_dir, f"{safe_metric}_mean_std.png"),
            title=args.title_prefix + metric if args.title_prefix else None,
        )

    print(f"Combined CSV: {combined_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-combined", required=True)
    p.add_argument("--add-combined", required=True)
    p.add_argument("--rename-from", default=None)
    p.add_argument("--rename-to", default=None)
    p.add_argument("--metric", action="append", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--title-prefix", default="")
    args = p.parse_args()
    main(args)
