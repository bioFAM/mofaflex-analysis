import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.plot_style import clean_ax, method_color_map, savefig, set_house_style


def _parse_label_offsets(items: list[str] | None) -> dict[str, tuple[float, float]]:
    offsets: dict[str, tuple[float, float]] = {}
    for item in items or []:
        if "=" not in item or "," not in item:
            raise ValueError(f"Expected label offset as 'name=dx,dy', got: {item}")
        name, coords = item.split("=", 1)
        dx_str, dy_str = coords.split(",", 1)
        offsets[name.strip()] = (float(dx_str), float(dy_str))
    return offsets


def _prepare_summary_frame(df: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    x_mean = f"{x_metric}_mean"
    x_std = f"{x_metric}_std"
    y_mean = f"{y_metric}_mean"
    y_std = f"{y_metric}_std"

    required = ["Embedding", x_mean, y_mean]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in summary CSV: {missing}")

    plot_df = df.copy()
    plot_df[x_mean] = pd.to_numeric(plot_df[x_mean], errors="coerce")
    plot_df[y_mean] = pd.to_numeric(plot_df[y_mean], errors="coerce")
    plot_df[x_std] = pd.to_numeric(plot_df.get(x_std, 0.0), errors="coerce").fillna(0.0)
    plot_df[y_std] = pd.to_numeric(plot_df.get(y_std, 0.0), errors="coerce").fillna(0.0)
    plot_df = plot_df.dropna(subset=[x_mean, y_mean]).copy()
    return plot_df


def _prepare_overlay_frame(df: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    required = ["Embedding", x_metric, y_metric]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in overlay CSV: {missing}")

    plot_df = df.copy()
    plot_df[x_metric] = pd.to_numeric(plot_df[x_metric], errors="coerce")
    plot_df[y_metric] = pd.to_numeric(plot_df[y_metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_metric, y_metric]).copy()
    return plot_df


def _relax_annotations(fig, annotations, max_iter: int = 80):
    if not annotations:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for _ in range(max_iter):
        moved = False
        bboxes = [ann.get_window_extent(renderer=renderer).expanded(1.06, 1.18) for ann in annotations]
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                if not bboxes[i].overlaps(bboxes[j]):
                    continue
                ai = annotations[i]
                aj = annotations[j]
                xi, yi = ai.xy
                xj, yj = aj.xy
                dx_i, dy_i = ai.xyann
                dx_j, dy_j = aj.xyann

                # Prefer separating labels vertically, then horizontally.
                if yi <= yj:
                    ai.xyann = (dx_i - 1.0, dy_i - 1.5)
                    aj.xyann = (dx_j + 1.0, dy_j + 1.5)
                else:
                    ai.xyann = (dx_i - 1.0, dy_i + 1.5)
                    aj.xyann = (dx_j + 1.0, dy_j - 1.5)
                moved = True

        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _plot_summary_scatter(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    out_png: str,
    title: str | None,
    x_label: str | None = None,
    y_label: str | None = None,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    relax_labels: bool = True,
    overlay_df: pd.DataFrame | None = None,
):
    x_mean = f"{x_metric}_mean"
    x_std = f"{x_metric}_std"
    y_mean = f"{y_metric}_mean"
    y_std = f"{y_metric}_std"

    plot_df = _prepare_summary_frame(df, x_metric=x_metric, y_metric=y_metric)
    overlay_plot_df = None
    if overlay_df is not None and len(overlay_df) > 0:
        overlay_plot_df = _prepare_overlay_frame(overlay_df, x_metric=x_metric, y_metric=y_metric)

    display_x = x_label or x_metric
    display_y = y_label or y_metric

    set_house_style()
    fig, ax = plt.subplots(figsize=(4.1, 4.1))
    categories = plot_df["Embedding"].astype(str).tolist()
    if overlay_plot_df is not None and len(overlay_plot_df) > 0:
        categories += overlay_plot_df["Embedding"].astype(str).tolist()
    color_map = method_color_map(sorted(dict.fromkeys(categories)))

    annotations = []
    label_offsets = label_offsets or {}

    for _, row in plot_df.iterrows():
        color = color_map[str(row["Embedding"])]
        ax.errorbar(
            x=float(row[x_mean]),
            y=float(row[y_mean]),
            xerr=float(row[x_std]),
            yerr=float(row[y_std]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.5,
            markersize=4.8,
            alpha=0.95,
            zorder=3,
        )
        dx, dy = label_offsets.get(str(row["Embedding"]), (4, 4))
        ann = ax.annotate(
            str(row["Embedding"]),
            (float(row[x_mean]), float(row[y_mean])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
        )
        annotations.append(ann)

    if overlay_plot_df is not None:
        for _, row in overlay_plot_df.iterrows():
            color = color_map[str(row["Embedding"])]
            x_val = float(row[x_metric])
            y_val = float(row[y_metric])
            ax.plot(
                x_val,
                y_val,
                marker="o",
                markersize=4.8,
                color=color,
                linestyle="None",
                zorder=3,
            )
            dx, dy = label_offsets.get(str(row["Embedding"]), (4, 4))
            ann = ax.annotate(
                str(row["Embedding"]),
                (x_val, y_val),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
            )
            annotations.append(ann)

    ax.set_xlabel(display_x)
    ax.set_ylabel(display_y)
    ax.set_title("")
    clean_ax(ax)

    x_min = float((plot_df[x_mean] - plot_df[x_std]).min())
    x_max = float((plot_df[x_mean] + plot_df[x_std]).max())
    y_min = float((plot_df[y_mean] - plot_df[y_std]).min())
    y_max = float((plot_df[y_mean] + plot_df[y_std]).max())
    if overlay_plot_df is not None and len(overlay_plot_df) > 0:
        x_min = min(x_min, float(overlay_plot_df[x_metric].min()))
        x_max = max(x_max, float(overlay_plot_df[x_metric].max()))
        y_min = min(y_min, float(overlay_plot_df[y_metric].min()))
        y_max = max(y_max, float(overlay_plot_df[y_metric].max()))
    x_pad = max(0.01, (x_max - x_min) * 0.08)
    y_pad = max(0.01, (y_max - y_min) * 0.08)
    ax.set_xlim(max(0.0, x_min - x_pad), min(1.0, x_max + x_pad))
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))
    if relax_labels:
        _relax_annotations(fig, annotations)

    fig.tight_layout()
    out_png = os.path.expanduser(out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    savefig(fig, out_png)
    plt.close(fig)


def main(args):
    df = pd.read_csv(os.path.expanduser(args.input))
    overlay_df = None
    if args.overlay_input:
        overlay_df = pd.read_csv(os.path.expanduser(args.overlay_input))
        if args.overlay_name:
            overlay_df = overlay_df[overlay_df["Embedding"] == args.overlay_name].copy()
    _plot_summary_scatter(
        df=df,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        out_png=args.out_png,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        label_offsets=_parse_label_offsets(args.label_offset),
        relax_labels=not args.no_relax_labels,
        overlay_df=overlay_df,
    )
    print(f"Saved scatter plot: {os.path.expanduser(args.out_png)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Seed summary CSV from summarize_seed_sweep.py or merge_seed_summaries.py")
    p.add_argument("--x-metric", default="Bio conservation")
    p.add_argument("--y-metric", default="Corr-Weighted")
    p.add_argument("--out-png", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--x-label", default=None)
    p.add_argument("--y-label", default=None)
    p.add_argument("--label-offset", action="append", default=None, help="Manual text offset 'name=dx,dy'.")
    p.add_argument("--no-relax-labels", action="store_true", help="Disable automatic label relaxation.")
    p.add_argument("--overlay-input", default=None, help="Optional CSV with single-run metrics to overlay as points.")
    p.add_argument("--overlay-name", default=None, help="Optional Embedding name to filter from the overlay CSV.")
    args = p.parse_args()
    main(args)
