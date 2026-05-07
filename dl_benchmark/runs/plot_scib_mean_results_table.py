import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from sklearn.preprocessing import MinMaxScaler


_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"


def _load_seed_tables(sweep_root: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(sweep_root.glob("seed_*")):
        seed_name = seed_dir.name
        csv_path = seed_dir / "sln_208_d1_scib_benchmarker.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        metric_type_row = df[df["Embedding"] == _METRIC_TYPE]
        df = df[df["Embedding"] != _METRIC_TYPE].copy()
        if metric_type_row.empty:
            raise ValueError(f"Missing '{_METRIC_TYPE}' row in {csv_path}")
        metric_type_row = metric_type_row.iloc[0].to_dict()
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["seed"] = seed_name
            rec[_METRIC_TYPE] = metric_type_row["Bio conservation"]
            rows.append(rec)
    if not rows:
        raise ValueError(f"No per-seed scIB tables found under {sweep_root}")
    out = pd.DataFrame(rows)
    for col in out.columns:
        if col not in {"Embedding", "seed", _METRIC_TYPE}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_mean_results_df(seed_df: pd.DataFrame, min_max_scale: bool = False) -> pd.DataFrame:
    detail_metric_cols = [
        c for c in seed_df.columns if c not in {"Embedding", "seed", _METRIC_TYPE, "Bio conservation"}
    ]
    mean_df = seed_df.groupby("Embedding")[detail_metric_cols].mean()
    if min_max_scale:
        mean_df = pd.DataFrame(
            MinMaxScaler().fit_transform(mean_df),
            index=mean_df.index,
            columns=mean_df.columns,
        )
    mean_df = mean_df.transpose()
    mean_df[_METRIC_TYPE] = "Bio conservation"
    mean_df = mean_df.transpose()
    data_rows = mean_df.index != _METRIC_TYPE
    mean_df.loc[data_rows, "Bio conservation"] = mean_df.loc[data_rows, detail_metric_cols].mean(axis=1)
    mean_df.loc[_METRIC_TYPE, "Bio conservation"] = _AGGREGATE_SCORE
    return mean_df


def _build_std_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in seed_df.columns if c not in {"Embedding", "seed", _METRIC_TYPE}]
    std_df = seed_df.groupby("Embedding")[metric_cols].std(ddof=1).fillna(0.0)
    std_df.index.name = "Embedding"
    return std_df.reset_index()


def _plot_results_table(df: pd.DataFrame, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    num_embeds = len(df.index) - 1
    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)
    plot_df = df.drop(_METRIC_TYPE, axis=0)
    plot_df = plot_df.sort_values(by="Bio conservation", ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
    other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.25}},
            cmap=cmap_fn(plot_df[col]),
            group=df.loc[_METRIC_TYPE, col],
            formatter="{:.2f}",
        )
        for col in other_cols
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": mpl.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=df.loc[_METRIC_TYPE, col],
            border="left",
        )
        for col in score_cols
    ]

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
        Table(
            plot_df,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
        fig.savefig(save_dir / "scib_results.svg", facecolor=ax.get_facecolor(), dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    sweep_root = Path(os.path.expanduser(args.sweep_root))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_df = _load_seed_tables(sweep_root)
    seed_df.to_csv(out_dir / "seed_level_results.csv", index=False)
    _build_std_table(seed_df).to_csv(out_dir / "results_std.csv", index=False)

    default_df = _build_mean_results_df(seed_df, min_max_scale=False)
    default_df.to_csv(out_dir / "results_mean.csv")

    _plot_results_table(default_df, out_dir / "default")
    _plot_results_table(default_df, out_dir / "min_max_scale_false")

    scaled_df = _build_mean_results_df(seed_df, min_max_scale=True)
    scaled_df.to_csv(out_dir / "results_mean_minmax_true.csv")
    _plot_results_table(scaled_df, out_dir / "min_max_scale_true")

    print(f"Saved mean-over-seeds scIB-style tables under: {out_dir}")


if __name__ == "__main__":
    main()
