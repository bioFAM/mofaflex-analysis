import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.plot_style import clean_ax, method_color_map, savefig, set_house_style


def _parse_metric_label_map(items: list[str] | None) -> dict[str, str]:
    mapping = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected metric label mapping as 'metric=label', got: {item}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def _discover_seed_tables(root: str, filename: str) -> list[tuple[int, Path]]:
    root_path = Path(os.path.expanduser(root))
    found = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir() or not child.name.startswith("seed_"):
            continue
        try:
            seed = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        table_path = child / filename
        if table_path.exists():
            found.append((seed, table_path))
    return found


def _load_seed_table(seed: int, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Embedding" not in df.columns:
        raise KeyError(f"Missing 'Embedding' column in {path}")
    out = df.copy()
    out = out[out["Embedding"] != "Metric Type"].copy()
    out["seed"] = int(seed)
    return out


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in {"Embedding", "seed"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in {"Embedding", "seed"}]
    grouped = df.groupby("Embedding")[numeric_cols].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = [
        "Embedding" if col[0] == "Embedding" else f"{col[0]}_{col[1]}"
        for col in grouped.columns
    ]
    return grouped


def _plot_metric(
    summary: pd.DataFrame,
    metric: str,
    out_png: str,
    title: str | None,
    metric_label: str | None = None,
):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    count_col = f"{metric}_count"
    if mean_col not in summary.columns:
        raise KeyError(f"Missing summary column '{mean_col}'")

    plot_df = summary[["Embedding", mean_col]].copy()
    plot_df = plot_df.rename(columns={mean_col: "mean"})
    plot_df["std"] = summary[std_col] if std_col in summary.columns else 0.0
    plot_df["count"] = summary[count_col] if count_col in summary.columns else None
    plot_df["std"] = plot_df["std"].fillna(0.0)
    plot_df = plot_df.sort_values("mean", ascending=True).reset_index(drop=True)
    color_map = method_color_map(sorted(summary["Embedding"].astype(str).tolist()))

    display_metric = metric_label or metric

    set_house_style()
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    bars = sns.barplot(
        data=plot_df,
        x="Embedding",
        y="mean",
        hue="Embedding",
        dodge=False,
        legend=False,
        ax=ax,
        palette=color_map,
    )

    x = range(len(plot_df))
    ax.errorbar(
        x=list(x),
        y=plot_df["mean"].to_numpy(),
        yerr=plot_df["std"].to_numpy(),
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        zorder=5,
    )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(display_metric)
    ymax = float((plot_df["mean"] + plot_df["std"]).max()) if len(plot_df) else 1.0
    ax.set_ylim(0, max(1.0, ymax * 1.1))
    ax.tick_params(axis="x", rotation=35)
    clean_ax(ax)

    for patch, (_, row) in zip(bars.patches, plot_df.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            patch.get_height() + float(row["std"]) + 0.005,
            f"{row['mean']:.3f}\n±{row['std']:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    fig.tight_layout()
    out_png = os.path.expanduser(out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    savefig(fig, out_png)
    plt.close(fig)


def main(args):
    discovered = _discover_seed_tables(args.root, args.input_name)
    if not discovered:
        raise FileNotFoundError(
            f"No seed tables named '{args.input_name}' found under {os.path.expanduser(args.root)}"
        )

    combined = pd.concat(
        [_load_seed_table(seed, path) for seed, path in discovered],
        ignore_index=True,
    )
    combined = _coerce_numeric_columns(combined)
    summary = _aggregate(combined)

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = out_dir / args.combined_name
    summary_csv = out_dir / args.summary_name
    summary_json = out_dir / args.summary_json_name

    combined.to_csv(combined_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    metric_label_map = _parse_metric_label_map(args.metric_label)

    for metric in args.metric:
        safe_metric = metric.lower().replace(" ", "_").replace("-", "_")
        _plot_metric(
            summary=summary,
            metric=metric,
            out_png=str(out_dir / f"{safe_metric}_mean_std.png"),
            title=args.title_prefix + metric if args.title_prefix else None,
            metric_label=metric_label_map.get(metric),
        )

    print(f"Combined CSV: {combined_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Seed sweep root containing seed_* folders.")
    p.add_argument("--input-name", default="sln_208_d1_metric_comparison.csv")
    p.add_argument("--metric", action="append", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--combined-name", default="seed_metrics_combined.csv")
    p.add_argument("--summary-name", default="seed_metrics_summary.csv")
    p.add_argument("--summary-json-name", default="seed_metrics_summary.json")
    p.add_argument("--title-prefix", default="")
    p.add_argument("--metric-label", action="append", default=None, help="Display label mapping 'metric=label'.")
    args = p.parse_args()
    main(args)
