from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.plot_style import CAT_PALETTE, clean_ax, place_legend, savefig, set_house_style
    from mofaflex_sensitivity.sln208_prior_noise_refinement import (
        DEFAULT_DATA_PATH,
        bool_mask_from_gene_set_stats,
        build_paper_hallmark_mca_gene_set_stats,
    )
    from mofaflex_sensitivity.sln208_prior_noise_baselines import (
        _dense_array,
        align_weights_by_spectra_overlap_labels,
        align_weights_to_prior_mask,
        preprocess_baseline_input,
    )
    from mofaflex_sensitivity.mca_filter_sensitivity import load_sln208_mdata
else:
    from .plot_style import CAT_PALETTE, clean_ax, place_legend, savefig, set_house_style
    from .mca_filter_sensitivity import load_sln208_mdata
    from .sln208_prior_noise_baselines import (
        _dense_array,
        align_weights_by_spectra_overlap_labels,
        align_weights_to_prior_mask,
        preprocess_baseline_input,
    )
    from .sln208_prior_noise_refinement import (
        DEFAULT_DATA_PATH,
        bool_mask_from_gene_set_stats,
        build_paper_hallmark_mca_gene_set_stats,
    )


DEFAULT_MOFAFLEX_DIR = Path("artifacts/sln208_prior_noise_refinement_sweep_mca_0to40_5seeds_noise")
DEFAULT_BASELINES_DIR = Path("artifacts/sln208_prior_noise_baselines_mca_0to40_5seeds_noise")
METHOD_LABELS = {
    "mofaflex": "MOFA-FLEX",
    "expimap": "ExpiMap",
    "spectra": "Spectra",
    "topicvi": "TopicVI",
}
METHOD_ORDER = ["MOFA-FLEX", "ExpiMap", "Spectra", "TopicVI"]
METHOD_ORDER_INDEX = {method: idx for idx, method in enumerate(METHOD_ORDER)}
CELL_TYPE_AUPR_MAP = {
    "HAN_MARGINAL ZONE B CELL": ("B",),
    "HAN_T CELL": ("CD4", "CD8"),
    "HAN_DENDRITIC CELL_S100A4 HIGH": ("DC",),
    "HAN_NK CELL": ("NK",),
}
PATHWAY_LABELS = {
    "HAN_MARGINAL ZONE B CELL": "Marginal Zone B Cell [M]",
    "HAN_T CELL": "T Cell [M]",
    "HAN_DENDRITIC CELL_S100A4 HIGH": "Dendritic Cell S100A4 High [M]",
    "HAN_NK CELL": "NK Cell [M]",
}
CELL_TYPE_PATHWAY_ORDER = [PATHWAY_LABELS[name] for name in CELL_TYPE_AUPR_MAP]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot SLN208 prior-noise MCA recovery comparisons across MOFA-FLEX and baselines."
    )
    parser.add_argument("--mofaflex-dir", type=Path, default=DEFAULT_MOFAFLEX_DIR)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mofaflex", "expimap", "spectra", "topicvi"],
        choices=["mofaflex", "expimap", "spectra", "topicvi"],
        help="Methods to include if their MCA per-run summary exists.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def _read_method_summary(method: str, *, mofaflex_dir: Path, baselines_dir: Path) -> pd.DataFrame | None:
    if method == "expimap":
        matched = _read_expimap_aupr_aligned_summary(baselines_dir)
        if matched is not None:
            return matched

    if method == "mofaflex":
        path = mofaflex_dir / "summary_mca_only_per_run.csv"
    else:
        path = baselines_dir / method / "summary_mca_only_per_run.csv"
    label = METHOD_LABELS[method]

    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"noise_level", "seed", "average_precision_true", "average_precision_noisy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.loc[:, ["noise_level", "seed", "average_precision_true", "average_precision_noisy"]].copy()
    df["method"] = label
    return df


def _load_true_mask(baselines_dir: Path) -> pd.DataFrame:
    path = baselines_dir / "true_mask.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0).astype(bool)

    mdata = load_sln208_mdata(DEFAULT_DATA_PATH)
    var_names = pd.Index(mdata.mod["rna"].var_names)
    return bool_mask_from_gene_set_stats(build_paper_hallmark_mca_gene_set_stats(var_names), var_names)


def _load_or_align_expimap_weights(run_dir: Path) -> pd.DataFrame:
    """Return soft-mask ExpiMap weights after one-to-one AP matching to the noisy prior."""
    aligned_path = run_dir / "weights_aupr_aligned.csv"
    assignment_path = run_dir / "expimap_aupr_factor_assignment.csv"
    if aligned_path.exists() and assignment_path.exists():
        return pd.read_csv(aligned_path, index_col=0).astype(float)

    weights_path = run_dir / "weights.csv"
    prior_path = run_dir / "noisy_mask.csv"
    if not weights_path.exists() or not prior_path.exists():
        raise FileNotFoundError(f"Missing ExpiMap weights or noisy mask in {run_dir}")

    weights = pd.read_csv(weights_path, index_col=0).astype(float)
    prior_mask = pd.read_csv(prior_path, index_col=0).astype(bool)
    weights = weights.reindex(columns=prior_mask.columns, fill_value=0.0).fillna(0.0)
    aligned, assignment = align_weights_to_prior_mask(weights, prior_mask)
    aligned.to_csv(aligned_path)
    assignment.to_csv(assignment_path, index=False)
    return aligned


def _read_expimap_aupr_aligned_summary(baselines_dir: Path) -> pd.DataFrame | None:
    """Recompute ExpiMap MCA AUPR after post hoc one-to-one AP factor matching."""
    expimap_dir = baselines_dir / "expimap"
    run_dirs = sorted(path.parent for path in expimap_dir.glob("noise_*/seed_*/weights.csv"))
    if not run_dirs:
        return None

    true_mask = _load_true_mask(baselines_dir)
    mca_pathways = [name for name in true_mask.index if str(name).startswith("HAN_")]
    rows = []
    for run_dir in run_dirs:
        noisy_mask = (
            pd.read_csv(run_dir / "noisy_mask.csv", index_col=0)
            .astype(bool)
            .reindex(columns=true_mask.columns, fill_value=False)
        )
        aligned = _load_or_align_expimap_weights(run_dir).reindex(columns=true_mask.columns, fill_value=0.0).fillna(0.0)
        rows.append(
            {
                "noise_level": _noise_from_run_dir(run_dir),
                "seed": _seed_from_run_dir(run_dir),
                "average_precision_true": _mean_mca_ap(aligned, true_mask, mca_pathways),
                "average_precision_noisy": _mean_mca_ap(aligned, noisy_mask, mca_pathways),
                "method": METHOD_LABELS["expimap"],
            }
        )
    return pd.DataFrame.from_records(rows)


def load_mca_ap_comparison(
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
    methods: list[str],
) -> pd.DataFrame:
    frames = []
    for method in methods:
        frame = _read_method_summary(method, mofaflex_dir=mofaflex_dir, baselines_dir=baselines_dir)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No method summary_mca_only_per_run.csv files were found.")

    df = pd.concat(frames, ignore_index=True)
    df["_method_order"] = df["method"].map(METHOD_ORDER_INDEX).fillna(len(METHOD_ORDER)).astype(int)
    df = df.sort_values(["noise_level", "_method_order", "method", "seed"]).drop(columns="_method_order").reset_index(drop=True)
    df["noise_label"] = df["noise_level"].map(lambda value: f"{float(value):.2f}")
    return df


def save_mca_ap_boxplot(
    df: pd.DataFrame,
    *,
    path: Path,
    metric_col: str,
    ylabel: str,
    figsize: tuple[float, float] = (9.35, 6.2),
) -> None:
    if df.empty:
        return
    if metric_col not in df.columns:
        raise ValueError(f"Missing metric column: {metric_col}")

    set_house_style()
    present_methods = set(df["method"])
    preferred_methods = [method for method in METHOD_ORDER if method in present_methods]
    other_methods = [method for method in dict.fromkeys(df["method"].tolist()) if method not in preferred_methods]
    methods = preferred_methods + other_methods
    noise_levels = sorted(df["noise_level"].unique())
    noise_order = [f"{float(value):.1f}" for value in noise_levels]
    default_colors = sns.color_palette(CAT_PALETTE, n_colors=max(3, len(methods)))
    preferred_palette = {
        "MOFA-FLEX": default_colors[0],
        "ExpiMap": default_colors[1],
        "Spectra": default_colors[2],
    }
    palette = {method: preferred_palette.get(method, default_colors[idx % len(default_colors)]) for idx, method in enumerate(methods)}

    fig, ax = plt.subplots(figsize=figsize)
    base_positions = np.arange(len(noise_levels), dtype=float) * 1.45
    method_offsets = np.linspace(-0.30, 0.30, num=len(methods)) if len(methods) > 1 else np.array([0.0])
    rng = np.random.default_rng(0)
    box_width = 0.36 if len(methods) > 1 else 0.48
    label_fontsize = 15.0
    tick_fontsize = 13.3
    legend_fontsize = 13.0
    legend_title_fontsize = 14.0

    for method, offset in zip(methods, method_offsets, strict=True):
        method_df = df[df["method"] == method]
        values = [
            method_df.loc[method_df["noise_level"] == noise_level, metric_col].to_numpy(dtype=float)
            for noise_level in noise_levels
        ]
        positions = base_positions + offset
        ax.boxplot(
            values,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": palette[method], "edgecolor": "0.25", "linewidth": 0.9},
            medianprops={"color": "0.1", "linewidth": 1.0},
            whiskerprops={"color": "0.35", "linewidth": 0.8},
            capprops={"color": "0.35", "linewidth": 0.8},
        )
        for xpos, vals in zip(positions, values, strict=True):
            if vals.size == 0:
                continue
            jitter = rng.normal(loc=0.0, scale=box_width * 0.13, size=vals.size)
            ax.scatter(
                np.full(vals.size, xpos) + jitter,
                vals,
                s=13,
                color=palette[method],
                alpha=1.0,
                linewidths=0,
                zorder=3,
            )
    ax.set_xticks(base_positions)
    ax.set_xticklabels(noise_order)
    ax.set_xlabel("Noise level (FPR = FNR)", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    clean_ax(ax)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[method],
            markeredgecolor=palette[method],
            markersize=6.5,
            label=method,
        )
        for method in methods
    ]
    ax.legend(
        handles=handles,
        title="Method",
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.12, top=0.98)
    savefig(fig, path)
    plt.close(fig)


def _load_cell_type_labels() -> pd.Series:
    mdata = load_sln208_mdata(DEFAULT_DATA_PATH)
    rna_obs = mdata.mod["rna"].obs
    if "cell_types (high)" not in rna_obs.columns:
        raise KeyError("'cell_types (high)' is missing from RNA obs.")
    return rna_obs["cell_types (high)"].astype(str).reset_index(drop=True)


def _mofaflex_factor_scores(model_path: Path) -> pd.DataFrame:
    import mofaflex as mfl

    cache_path = model_path.parent.parent / "mofaflex_factor_scores.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0).reset_index(drop=True)

    model = mfl.MOFAFLEX.load(model_path, map_location="cpu")
    factors_by_group = model.get_factors(return_type="pandas", ordered=False)
    factors = next(iter(factors_by_group.values())) if isinstance(factors_by_group, dict) else factors_by_group
    factors = factors.reset_index(drop=True)
    factors.to_csv(cache_path)
    return factors


def _expimap_factor_scores(run_dir: Path) -> pd.DataFrame:
    aligned_cache_path = run_dir / "expimap_factor_scores_aupr_aligned.csv"
    if aligned_cache_path.exists():
        return pd.read_csv(aligned_cache_path, index_col=0).reset_index(drop=True)

    raw_cache_path = run_dir / "expimap_factor_scores.csv"
    if raw_cache_path.exists():
        raw_factors = pd.read_csv(raw_cache_path, index_col=0).reset_index(drop=True)
    else:
        mdata = load_sln208_mdata(DEFAULT_DATA_PATH)
        rna = mdata.mod["rna"]
        data = _dense_array(rna.X).astype(np.float32, copy=False)
        model_data, _input_preprocessing = preprocess_baseline_input("expimap", data)

        # ExpiMap was trained from an AnnData built directly from the data array,
        # so the saved model expects numeric var_names ("0", "1", ...), not gene names.
        adata = ad.AnnData(model_data.copy(), obs=rna.obs.copy())
        adata.obs["cond"] = "cond"
        noisy_mask = pd.read_csv(run_dir / "noisy_mask.csv", index_col=0).astype(bool)
        adata.varm["I"] = noisy_mask.to_numpy(dtype=bool).T
        adata.uns["terms"] = noisy_mask.index.tolist()

        import scarches as sca

        model = sca.models.EXPIMAP.load(str(run_dir / "expimap_model"), adata=adata, map_location="cpu")
        factors = np.asarray(model.get_latent(), dtype=float)
        raw_factors = pd.DataFrame(factors, columns=noisy_mask.index).reset_index(drop=True)
        raw_factors.to_csv(raw_cache_path)

    # Match the soft-mask ExpiMap latent dimensions to priors using the same
    # one-to-one AP assignment used for loadings. The raw column labels are not
    # reliable gene-program identities when soft_mask=True.
    _load_or_align_expimap_weights(run_dir)
    assignment = pd.read_csv(run_dir / "expimap_aupr_factor_assignment.csv")
    aligned: dict[str, np.ndarray] = {}
    zero = np.zeros(raw_factors.shape[0], dtype=float)
    for row in assignment.itertuples(index=False):
        pathway_name = str(row.pathway_name)
        factor_name = str(row.factor_name)
        if factor_name in raw_factors.columns:
            aligned[pathway_name] = raw_factors[factor_name].to_numpy(dtype=float)
        else:
            aligned[pathway_name] = zero.copy()
    factors_df = pd.DataFrame(aligned)
    factors_df.to_csv(aligned_cache_path)
    return factors_df


def _spectra_factor_scores(run_dir: Path) -> pd.DataFrame:
    import torch

    cache_path = run_dir / "spectra_factor_scores.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0).reset_index(drop=True)

    model = torch.load(run_dir / "spectra_model.pt", map_location="cpu", weights_only=False)
    raw_scores = np.asarray(model.return_cell_scores(), dtype=float)
    raw_factors = pd.DataFrame(
        raw_scores,
        columns=[f"spectra_factor_{idx:03d}" for idx in range(raw_scores.shape[1])],
    )
    assignment = pd.read_csv(run_dir / "factor_assignment.csv")

    aligned: dict[str, np.ndarray] = {}
    zero = np.zeros(raw_factors.shape[0], dtype=float)
    for row in assignment.itertuples(index=False):
        pathway_name = str(row.pathway_name)
        factor_name = getattr(row, "factor_name")
        if pd.isna(factor_name) or str(factor_name) not in raw_factors.columns:
            aligned[pathway_name] = zero.copy()
        else:
            aligned[pathway_name] = raw_factors[str(factor_name)].to_numpy(dtype=float)
    factors_df = pd.DataFrame(aligned)
    factors_df.to_csv(cache_path)
    return factors_df


def _generate_topicvi_mca_summary(baselines_dir: Path) -> None:
    topicvi_dir = baselines_dir / "topicvi"
    out_path = topicvi_dir / "summary_mca_only_per_run.csv"
    run_dirs = sorted(p.parent for p in topicvi_dir.glob("noise_*/seed_*/pathway_refinement.csv"))
    if not run_dirs:
        return
    rows = []
    for run_dir in run_dirs:
        df = pd.read_csv(run_dir / "pathway_refinement.csv")
        mca_df = df.loc[df["source"] == "mca"]
        if mca_df.empty:
            continue
        rows.append(
            {
                "noise_level": _noise_from_run_dir(run_dir),
                "seed": _seed_from_run_dir(run_dir),
                "average_precision_true": float(mca_df["average_precision_true"].mean()),
                "average_precision_noisy": float(mca_df["average_precision_noisy"].mean()),
            }
        )
    if rows:
        pd.DataFrame.from_records(rows).to_csv(out_path, index=False)


def _topicvi_factor_scores(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "topicvi_factor_scores.csv").reset_index(drop=True)


def _load_rna_ref(data_path: Path = DEFAULT_DATA_PATH) -> np.ndarray:
    import mudata as md

    mdata = md.read_h5mu(data_path)
    X = mdata.mod["rna"].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _rmse(X_hat: np.ndarray, X_ref: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.square(X_hat - X_ref))))


def _mean_gene_pearson(X_hat: np.ndarray, X_ref: np.ndarray) -> float:
    a = X_hat - X_hat.mean(axis=0, keepdims=True)
    b = X_ref - X_ref.mean(axis=0, keepdims=True)
    denom_a = np.sqrt((a ** 2).sum(axis=0))
    denom_b = np.sqrt((b ** 2).sum(axis=0))
    valid = (denom_a > 0) & (denom_b > 0)
    corr = (a[:, valid] * b[:, valid]).sum(axis=0) / (denom_a[valid] * denom_b[valid])
    return float(np.mean(np.clip(corr, -1.0, 1.0)))


def _reconstruction_from_assigned(
    factor_scores_path: Path,
    weights_raw_path: Path,
    assignment_path: Path,
    *,
    factor_col: str,
    pathway_col: str = "pathway_name",
) -> np.ndarray:
    """Reconstruct X_hat using assigned factor scores + raw weights, aligned by assignment."""
    fs = pd.read_csv(factor_scores_path)  # cells × pathways (columns = pathway names)
    w = pd.read_csv(weights_raw_path, index_col=0)  # factors × genes
    fa = pd.read_csv(assignment_path)  # pathway_name, factor_name
    topic_to_pathway = fa.set_index(factor_col)[pathway_col].to_dict()
    # Reorder fs columns to match w index order
    ordered_cols = [topic_to_pathway[t] for t in w.index if t in topic_to_pathway]
    ordered_topics = [t for t in w.index if t in topic_to_pathway]
    Z = fs[ordered_cols].values.astype(np.float32)
    W = w.loc[ordered_topics].values.astype(np.float32)
    return Z @ W


def _mofaflex_reconstruction(model_path: Path, cache_path: Path | None = None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    import mofaflex as mfl

    model = mfl.MOFAFLEX.load(model_path, map_location="cpu")
    Z = next(iter(model.get_factors(return_type="pandas", ordered=False).values())).values
    W = model.get_weights(return_type="pandas")["rna"].values
    X_hat = (Z @ W).astype(np.float32)
    if cache_path is not None:
        np.save(cache_path, X_hat)
    return X_hat


def load_reconstruction_comparison(
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
    methods: list[str],
    data_path: Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    X_ref = _load_rna_ref(data_path)
    reference = load_mca_ap_comparison(mofaflex_dir=mofaflex_dir, baselines_dir=baselines_dir, methods=["mofaflex"])
    grid = reference.loc[:, ["noise_level", "seed"]].drop_duplicates().sort_values(["noise_level", "seed"])

    rows: list[dict[str, object]] = []
    for row in grid.itertuples(index=False):
        noise_level = float(row.noise_level)
        seed = int(row.seed)

        if "mofaflex" in methods:
            model_path = _mofaflex_model_path(mofaflex_dir, noise_level=noise_level, seed=seed)
            cache = mofaflex_dir / f"recon_cache_seed{seed:03d}_noise{_noise_slug(noise_level)}.npy"
            X_hat = _mofaflex_reconstruction(model_path, cache_path=cache)
            rows.append({"method": "MOFA-FLEX", "noise_level": noise_level, "seed": seed,
                         "mean_gene_pearson": _mean_gene_pearson(X_hat, X_ref),
                         "rmse": _rmse(X_hat, X_ref)})

        if "expimap" in methods:
            run_dir = _run_dir_from_noise_seed(baselines_dir, method="expimap", noise_level=noise_level, seed=seed)
            fs_path = run_dir / "expimap_factor_scores.csv"
            if fs_path.exists() and (run_dir / "weights_raw.csv").exists():
                Z = pd.read_csv(fs_path, index_col=0).values.astype(np.float32)
                W = pd.read_csv(run_dir / "weights_raw.csv", index_col=0).values.astype(np.float32)
                X_hat_ep = Z @ W
                rows.append({"method": "ExpiMap", "noise_level": noise_level, "seed": seed,
                             "mean_gene_pearson": _mean_gene_pearson(X_hat_ep, X_ref),
                             "rmse": _rmse(X_hat_ep, X_ref)})

        if "spectra" in methods:
            run_dir = _run_dir_from_noise_seed(baselines_dir, method="spectra", noise_level=noise_level, seed=seed)
            if (run_dir / "spectra_factor_scores.csv").exists():
                X_hat = _reconstruction_from_assigned(
                    run_dir / "spectra_factor_scores.csv",
                    run_dir / "weights_raw.csv",
                    run_dir / "factor_assignment.csv",
                    factor_col="factor_name",
                )
                rows.append({"method": "Spectra", "noise_level": noise_level, "seed": seed,
                             "mean_gene_pearson": _mean_gene_pearson(X_hat, X_ref),
                             "rmse": _rmse(X_hat, X_ref)})

        if "topicvi" in methods:
            run_dir = _run_dir_from_noise_seed(baselines_dir, method="topicvi", noise_level=noise_level, seed=seed)
            expr_cache = run_dir / "topicvi_normalized_expression.npy"
            if expr_cache.exists():
                X_hat = np.load(expr_cache)
                rows.append({"method": "TopicVI", "noise_level": noise_level, "seed": seed,
                             "mean_gene_pearson": _mean_gene_pearson(X_hat, X_ref),
                             "rmse": _rmse(X_hat, X_ref)})

    df = pd.DataFrame.from_records(rows)
    df["noise_label"] = df["noise_level"].map(lambda v: f"{float(v):.2f}")
    df["_method_order"] = df["method"].map(METHOD_ORDER_INDEX).fillna(len(METHOD_ORDER)).astype(int)
    df = df.sort_values(["noise_level", "_method_order", "seed"]).drop(columns="_method_order").reset_index(drop=True)
    return df


def _run_dir_from_noise_seed(root: Path, *, method: str, noise_level: float, seed: int) -> Path:
    return root / method / f"noise_{_noise_slug(noise_level)}" / f"seed_{seed:03d}"


def _noise_slug(noise_level: float) -> str:
    # Keep the folder convention used by the sweep helpers, e.g. 0.05 -> 0p05.
    text = f"{float(noise_level):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _mofaflex_model_path(mofaflex_dir: Path, *, noise_level: float, seed: int) -> Path:
    run_dir = mofaflex_dir / f"noise_{_noise_slug(noise_level)}" / f"seed_{seed:03d}"
    paths = sorted((run_dir / "models").glob("*.h5"))
    if paths:
        return paths[0]
    flat = mofaflex_dir / "models" / f"seed_{seed:03d}_fpr{float(noise_level):0.3f}_fnr{float(noise_level):0.3f}.h5"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"Could not find MOFA-FLEX model for noise={noise_level}, seed={seed}")


def _celltype_rows_from_factors(
    *,
    factors: pd.DataFrame,
    labels: pd.Series,
    method: str,
    noise_level: float,
    seed: int,
) -> list[dict[str, object]]:
    label_values = labels.to_numpy(dtype=str)
    rows: list[dict[str, object]] = []
    for pathway_name, positive_cell_types in CELL_TYPE_AUPR_MAP.items():
        y_true = np.isin(label_values, list(positive_cell_types)).astype(int)
        if pathway_name in factors.columns:
            scores = factors[pathway_name].to_numpy(dtype=float)
            aupr = float(average_precision_score(y_true, scores)) if y_true.min() != y_true.max() else np.nan
        else:
            aupr = np.nan
        rows.append(
            {
                "method": method,
                "noise_level": float(noise_level),
                "seed": int(seed),
                "pathway_name": pathway_name,
                "pathway_pretty": PATHWAY_LABELS[pathway_name],
                "positive_cell_types": ",".join(positive_cell_types),
                "cell_type_aupr": aupr,
                "n_positive_cells": int(y_true.sum()),
                "n_cells": int(len(y_true)),
            }
        )
    return rows


def load_celltype_aupr_comparison(
    *,
    mofaflex_dir: Path,
    baselines_dir: Path,
    methods: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = _load_cell_type_labels()
    reference = load_mca_ap_comparison(mofaflex_dir=mofaflex_dir, baselines_dir=baselines_dir, methods=["mofaflex"])
    grid = reference.loc[:, ["noise_level", "seed"]].drop_duplicates().sort_values(["noise_level", "seed"])

    rows: list[dict[str, object]] = []
    for row in grid.itertuples(index=False):
        noise_level = float(row.noise_level)
        seed = int(row.seed)
        if "mofaflex" in methods:
            model_path = _mofaflex_model_path(mofaflex_dir, noise_level=noise_level, seed=seed)
            rows.extend(
                _celltype_rows_from_factors(
                    factors=_mofaflex_factor_scores(model_path),
                    labels=labels,
                    method="MOFA-FLEX",
                    noise_level=noise_level,
                    seed=seed,
                )
            )
        if "expimap" in methods:
            run_dir = _run_dir_from_noise_seed(baselines_dir, method="expimap", noise_level=noise_level, seed=seed)
            if (run_dir / "expimap_model").exists():
                rows.extend(
                    _celltype_rows_from_factors(
                        factors=_expimap_factor_scores(run_dir),
                        labels=labels,
                        method="ExpiMap",
                        noise_level=noise_level,
                        seed=seed,
                    )
                )
        if "spectra" in methods:
            run_dir = _run_dir_from_noise_seed(baselines_dir, method="spectra", noise_level=noise_level, seed=seed)
            if (run_dir / "spectra_model.pt").exists():
                rows.extend(
                    _celltype_rows_from_factors(
                        factors=_spectra_factor_scores(run_dir),
                        labels=labels,
                        method="Spectra",
                        noise_level=noise_level,
                        seed=seed,
                    )
                )
        if "topicvi" in methods:
            run_dir = _run_dir_from_noise_seed(baselines_dir, method="topicvi", noise_level=noise_level, seed=seed)
            if (run_dir / "topicvi_factor_scores.csv").exists():
                rows.extend(
                    _celltype_rows_from_factors(
                        factors=_topicvi_factor_scores(run_dir),
                        labels=labels,
                        method="TopicVI",
                        noise_level=noise_level,
                        seed=seed,
                    )
                )

    per_pathway = pd.DataFrame.from_records(rows)
    if per_pathway.empty:
        raise FileNotFoundError("No cell-type AUPR rows could be computed from saved models.")
    per_pathway["noise_label"] = per_pathway["noise_level"].map(lambda value: f"{float(value):.2f}")
    per_pathway["_method_order"] = per_pathway["method"].map(METHOD_ORDER_INDEX).fillna(len(METHOD_ORDER)).astype(int)
    per_pathway["pathway_pretty"] = pd.Categorical(
        per_pathway["pathway_pretty"],
        categories=CELL_TYPE_PATHWAY_ORDER,
        ordered=True,
    )
    per_pathway = (
        per_pathway.sort_values(["noise_level", "_method_order", "method", "seed", "pathway_pretty"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )
    overall = (
        per_pathway.groupby(["method", "noise_level", "noise_label", "seed"], as_index=False, observed=True)
        .agg(mean_cell_type_aupr=("cell_type_aupr", "mean"))
    )
    overall["_method_order"] = overall["method"].map(METHOD_ORDER_INDEX).fillna(len(METHOD_ORDER)).astype(int)
    overall = overall.sort_values(["noise_level", "_method_order", "method", "seed"]).drop(columns="_method_order")
    return per_pathway, overall


def save_celltype_aupr_by_pathway_plot(
    df: pd.DataFrame,
    *,
    path: Path,
    pathways: list[str] | None = None,
    col_wrap: int = 2,
    height: float = 3.1,
    aspect: float = 1.35,
) -> None:
    if df.empty:
        return
    set_house_style()
    plot_df = df.copy()
    if pathways is not None:
        plot_df = plot_df.loc[plot_df["pathway_pretty"].isin(pathways)].copy()
        if plot_df.empty:
            raise ValueError("No rows left after filtering cell-type AUPR plot to selected pathways.")
    present_methods = set(df["method"])
    methods = [method for method in METHOD_ORDER if method in present_methods]
    default_colors = sns.color_palette(CAT_PALETTE, n_colors=max(3, len(methods)))
    palette = dict(zip(methods, default_colors[: len(methods)], strict=False))
    if pathways is not None:
        pathway_order = [pathway for pathway in pathways if pathway in set(plot_df["pathway_pretty"])]
        plot_df["pathway_pretty"] = pd.Categorical(plot_df["pathway_pretty"], categories=pathway_order, ordered=True)
        plot_df = plot_df.sort_values(["pathway_pretty", "noise_level", "method"]).reset_index(drop=True)

    g = sns.catplot(
        data=plot_df,
        x="noise_label",
        y="cell_type_aupr",
        hue="method",
        col="pathway_pretty",
        col_wrap=col_wrap,
        kind="box",
        hue_order=methods,
        palette=palette,
        showfliers=False,
        height=height,
        aspect=aspect,
        linewidth=1.05,
        saturation=1.0,
        legend=False,
    )
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        clean_ax(ax)
        ax.set_xlabel("Noise level")
        ax.set_ylabel("Cell-type AUPR")
        ax.xaxis.label.set_size(10.5)
        ax.yaxis.label.set_size(10.5)
        ax.tick_params(axis="x", rotation=0, labelsize=9.2)
        ax.tick_params(axis="y", labelsize=9.2)
        ax.title.set_size(10.5)
        ticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
        if ticklabels and all(label.startswith("0.") for label in ticklabels if label):
            ax.set_xticks(ax.get_xticks(), [label[1:] if label.startswith("0") else label for label in ticklabels])
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[method], markersize=6, label=method)
        for method in methods
    ]
    legend = g.fig.legend(
        handles=handles,
        title="Method",
        frameon=False,
        bbox_to_anchor=(0.842, 0.5),
        loc="center left",
        fontsize=9.2,
        title_fontsize=10.0,
    )
    if legend is not None:
        legend.set_alignment("left")
    g.fig.subplots_adjust(top=0.92, right=0.84, hspace=0.35, wspace=0.18)
    savefig(g.fig, path)
    plt.close(g.fig)


def save_selected2_combined_plot(
    celltype_df: pd.DataFrame,
    pearson_df: pd.DataFrame,
    *,
    path: Path,
    pathways: list[str] | None = None,
) -> None:
    if celltype_df.empty or pearson_df.empty:
        return
    set_house_style()
    pathways = pathways or ["Marginal Zone B Cell [M]", "T Cell [M]"]
    pathways = [str(pathway) for pathway in pathways]

    cell_df = celltype_df.loc[celltype_df["pathway_pretty"].isin(pathways)].copy()
    pearson_plot_df = pearson_df.loc[pearson_df["pathway_pretty"].isin(pathways)].copy()
    if cell_df.empty or pearson_plot_df.empty:
        raise ValueError("No rows left after filtering selected2 combined plot to requested pathways.")

    present_methods = set(cell_df["method"]).intersection(set(pearson_plot_df["method"]))
    methods = [method for method in METHOD_ORDER if method in present_methods]
    default_colors = sns.color_palette(CAT_PALETTE, n_colors=max(3, len(methods)))
    palette = dict(zip(methods, default_colors[: len(methods)], strict=False))

    pathway_order = [pathway for pathway in pathways if pathway in set(cell_df["pathway_pretty"])]
    noise_levels = sorted(cell_df["noise_level"].unique())
    noise_order = [f"{float(value):.1f}" for value in noise_levels]
    base_positions = np.arange(len(noise_levels), dtype=float) * 1.42
    method_offsets = np.linspace(-0.28, 0.28, num=len(methods)) if len(methods) > 1 else np.array([0.0])
    box_width = 0.33 if len(methods) > 1 else 0.45
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 2, figsize=(8.9, 6.7), sharex=False, squeeze=False)

    for col_idx, pathway in enumerate(pathway_order):
        ax = axes[0, col_idx]
        pathway_df = cell_df.loc[cell_df["pathway_pretty"].eq(pathway)].copy()
        for method, offset in zip(methods, method_offsets, strict=True):
            method_df = pathway_df.loc[pathway_df["method"].eq(method)]
            values = [
                method_df.loc[method_df["noise_level"] == noise_level, "cell_type_aupr"].to_numpy(dtype=float)
                for noise_level in noise_levels
            ]
            positions = base_positions + offset
            ax.boxplot(
                values,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={"facecolor": palette[method], "edgecolor": "0.25", "linewidth": 1.05},
                medianprops={"color": "0.1", "linewidth": 1.05},
                whiskerprops={"color": "0.35", "linewidth": 0.85},
                capprops={"color": "0.35", "linewidth": 0.85},
            )
            for xpos, vals in zip(positions, values, strict=True):
                if vals.size == 0:
                    continue
                jitter = rng.normal(loc=0.0, scale=box_width * 0.13, size=vals.size)
                ax.scatter(
                    np.full(vals.size, xpos) + jitter,
                    vals,
                    s=13,
                    color=palette[method],
                    alpha=1.0,
                    linewidths=0,
                    zorder=3,
                )
        ax.set_title(pathway, fontsize=10.5)
        ax.set_xticks(base_positions, noise_order)
        ax.set_xlabel("Noise level", fontsize=10.5)
        ax.set_ylabel("Cell-type AUPR" if col_idx == 0 else "", fontsize=10.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, num=6))
        ax.tick_params(axis="x", labelsize=9.4)
        ax.tick_params(axis="y", labelsize=9.4)
        clean_ax(ax)

    for col_idx, pathway in enumerate(pathway_order):
        ax = axes[1, col_idx]
        pathway_df = pearson_plot_df.loc[pearson_plot_df["pathway_pretty"].eq(pathway)].copy()
        sns.lineplot(
            data=pathway_df,
            x="top_k_genes",
            y="mean_pearson",
            hue="method",
            hue_order=methods,
            marker="o",
            linewidth=2.1,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title("")
        ax.set_xlabel("Top genes", fontsize=10.5)
        ax.set_ylabel("Mean Pearson correlation" if col_idx == 0 else "", fontsize=10.5)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", labelsize=9.4)
        ax.tick_params(axis="y", labelsize=9.4)
        clean_ax(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[method],
            markeredgecolor=palette[method],
            markersize=6.5,
            label=method,
        )
        for method in methods
    ]
    legend = fig.legend(
        handles=handles,
        title="Method",
        frameon=False,
        bbox_to_anchor=(0.885, 0.5),
        loc="center left",
        fontsize=9.2,
        title_fontsize=10.0,
    )
    if legend is not None:
        legend.set_alignment("left")
    fig.subplots_adjust(right=0.84, wspace=0.20, hspace=0.28)
    savefig(fig, path)
    plt.close(fig)


def save_mca_ap_and_selected2_combined_plot(
    mca_df: pd.DataFrame,
    celltype_df: pd.DataFrame,
    pearson_df: pd.DataFrame,
    *,
    path: Path,
    metric_col: str = "average_precision_true",
    ylabel: str = "AUPR (MCA gene programs)",
    pathways: list[str] | None = None,
) -> None:
    if mca_df.empty or celltype_df.empty or pearson_df.empty:
        return
    if metric_col not in mca_df.columns:
        raise ValueError(f"Missing metric column: {metric_col}")

    set_house_style()
    pathways = pathways or ["Marginal Zone B Cell [M]", "T Cell [M]"]
    pathways = [str(pathway) for pathway in pathways]

    cell_df = celltype_df.loc[celltype_df["pathway_pretty"].isin(pathways)].copy()
    pearson_plot_df = pearson_df.loc[pearson_df["pathway_pretty"].isin(pathways)].copy()
    if cell_df.empty or pearson_plot_df.empty:
        raise ValueError("No rows left after filtering combined MCA AP and selected2 plot to requested pathways.")

    present_methods = (
        set(mca_df["method"]).intersection(set(cell_df["method"])).intersection(set(pearson_plot_df["method"]))
    )
    methods = [method for method in METHOD_ORDER if method in present_methods]
    if not methods:
        raise ValueError("No shared methods are available across the MCA AP and selected2 inputs.")

    default_colors = sns.color_palette(CAT_PALETTE, n_colors=max(3, len(methods)))
    preferred_palette = {
        "MOFA-FLEX": default_colors[0],
        "ExpiMap": default_colors[1],
        "Spectra": default_colors[2],
    }
    palette = {method: preferred_palette.get(method, default_colors[idx % len(default_colors)]) for idx, method in enumerate(methods)}

    label_fontsize = 12.6
    tick_fontsize = 12.0
    title_fontsize = 12.6
    legend_fontsize = 11.4
    legend_title_fontsize = 12.2

    noise_levels = sorted(mca_df["noise_level"].unique())
    noise_order = [f"{float(value):.1f}" for value in noise_levels]
    base_positions = np.arange(len(noise_levels), dtype=float) * 1.42
    method_offsets = np.linspace(-0.28, 0.28, num=len(methods)) if len(methods) > 1 else np.array([0.0])
    box_width = 0.33 if len(methods) > 1 else 0.45
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(16.2, 6.7))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[2.0, 1.0, 1.0],
        wspace=0.22,
        hspace=0.28,
    )
    left_ax = fig.add_subplot(gs[:, 0])
    top_left_ax = fig.add_subplot(gs[0, 1])
    top_right_ax = fig.add_subplot(gs[0, 2])
    bottom_left_ax = fig.add_subplot(gs[1, 1])
    bottom_right_ax = fig.add_subplot(gs[1, 2])

    plot_mca_df = mca_df.loc[mca_df["method"].isin(methods)].copy()
    for method, offset in zip(methods, method_offsets, strict=True):
        method_df = plot_mca_df.loc[plot_mca_df["method"].eq(method)]
        values = [
            method_df.loc[method_df["noise_level"] == noise_level, metric_col].to_numpy(dtype=float)
            for noise_level in noise_levels
        ]
        positions = base_positions + offset
        left_ax.boxplot(
            values,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": palette[method], "edgecolor": "0.25", "linewidth": 1.0},
            medianprops={"color": "0.1", "linewidth": 1.05},
            whiskerprops={"color": "0.35", "linewidth": 0.85},
            capprops={"color": "0.35", "linewidth": 0.85},
        )
        for xpos, vals in zip(positions, values, strict=True):
            if vals.size == 0:
                continue
            jitter = rng.normal(loc=0.0, scale=box_width * 0.13, size=vals.size)
            left_ax.scatter(
                np.full(vals.size, xpos) + jitter,
                vals,
                s=18,
                color=palette[method],
                alpha=1.0,
                linewidths=0,
                zorder=3,
            )
    left_ax.set_xticks(base_positions, noise_order)
    left_ax.set_xlabel("Noise level (FPR = FNR)", fontsize=label_fontsize)
    left_ax.set_ylabel(ylabel, fontsize=label_fontsize)
    left_ax.tick_params(axis="x", labelsize=tick_fontsize)
    left_ax.tick_params(axis="y", labelsize=tick_fontsize)
    clean_ax(left_ax)

    pathway_order = [pathway for pathway in pathways if pathway in set(cell_df["pathway_pretty"])]
    top_axes = [top_left_ax, top_right_ax]
    bottom_axes = [bottom_left_ax, bottom_right_ax]
    for col_idx, pathway in enumerate(pathway_order):
        ax = top_axes[col_idx]
        pathway_df = cell_df.loc[cell_df["pathway_pretty"].eq(pathway) & cell_df["method"].isin(methods)].copy()
        for method, offset in zip(methods, method_offsets, strict=True):
            method_df = pathway_df.loc[pathway_df["method"].eq(method)]
            values = [
                method_df.loc[method_df["noise_level"] == noise_level, "cell_type_aupr"].to_numpy(dtype=float)
                for noise_level in noise_levels
            ]
            positions = base_positions + offset
            ax.boxplot(
                values,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={"facecolor": palette[method], "edgecolor": "0.25", "linewidth": 1.0},
                medianprops={"color": "0.1", "linewidth": 1.05},
                whiskerprops={"color": "0.35", "linewidth": 0.85},
                capprops={"color": "0.35", "linewidth": 0.85},
            )
            for xpos, vals in zip(positions, values, strict=True):
                if vals.size == 0:
                    continue
                jitter = rng.normal(loc=0.0, scale=box_width * 0.13, size=vals.size)
                ax.scatter(
                    np.full(vals.size, xpos) + jitter,
                    vals,
                    s=16,
                    color=palette[method],
                    alpha=1.0,
                    linewidths=0,
                    zorder=3,
                )
        ax.set_title(pathway, fontsize=title_fontsize)
        ax.set_xticks(base_positions, noise_order)
        ax.set_xlabel("Noise level", fontsize=label_fontsize)
        ax.set_ylabel("Cell-type AUPR" if col_idx == 0 else "", fontsize=label_fontsize)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, num=6))
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        clean_ax(ax)

    for col_idx, pathway in enumerate(pathway_order):
        ax = bottom_axes[col_idx]
        pathway_df = pearson_plot_df.loc[
            pearson_plot_df["pathway_pretty"].eq(pathway) & pearson_plot_df["method"].isin(methods)
        ].copy()
        sns.lineplot(
            data=pathway_df,
            x="top_k_genes",
            y="mean_pearson",
            hue="method",
            hue_order=methods,
            marker="o",
            linewidth=2.1,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title("")
        ax.set_xlabel("Top genes", fontsize=label_fontsize)
        ax.set_ylabel("Mean Pearson correlation" if col_idx == 0 else "", fontsize=label_fontsize)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        clean_ax(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[method],
            markeredgecolor=palette[method],
            markersize=8.0,
            label=method,
        )
        for method in methods
    ]
    legend = fig.legend(
        handles=handles,
        title="Method",
        frameon=False,
        bbox_to_anchor=(0.982, 0.5),
        loc="center right",
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    if legend is not None:
        legend.set_alignment("left")

    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.11, top=0.97)
    savefig(fig, path)
    plt.close(fig)


def save_mca_ap_and_selected2_toprow_combined_plot(
    mca_df: pd.DataFrame,
    celltype_df: pd.DataFrame,
    *,
    path: Path,
    metric_col: str = "average_precision_true",
    ylabel: str = "AUPR (MCA gene programs)",
    pathways: list[str] | None = None,
) -> None:
    if mca_df.empty or celltype_df.empty:
        return
    if metric_col not in mca_df.columns:
        raise ValueError(f"Missing metric column: {metric_col}")

    set_house_style()
    pathways = pathways or ["Marginal Zone B Cell [M]", "T Cell [M]"]
    pathways = [str(pathway) for pathway in pathways]

    cell_df = celltype_df.loc[celltype_df["pathway_pretty"].isin(pathways)].copy()
    if cell_df.empty:
        raise ValueError("No rows left after filtering combined MCA AP and selected2 top-row plot to requested pathways.")

    present_methods = set(mca_df["method"]).intersection(set(cell_df["method"]))
    methods = [method for method in METHOD_ORDER if method in present_methods]
    if not methods:
        raise ValueError("No shared methods are available across the MCA AP and selected2 top-row inputs.")

    default_colors = sns.color_palette(CAT_PALETTE, n_colors=max(3, len(methods)))
    preferred_palette = {
        "MOFA-FLEX": default_colors[0],
        "ExpiMap": default_colors[1],
        "Spectra": default_colors[2],
    }
    palette = {method: preferred_palette.get(method, default_colors[idx % len(default_colors)]) for idx, method in enumerate(methods)}

    label_fontsize = 12.6
    tick_fontsize = 12.0
    title_fontsize = 12.6
    legend_fontsize = 11.4
    legend_title_fontsize = 12.2

    noise_levels = sorted(mca_df["noise_level"].unique())
    noise_order = [f"{float(value):.1f}" for value in noise_levels]
    base_positions = np.arange(len(noise_levels), dtype=float) * 1.42
    method_offsets = np.linspace(-0.28, 0.28, num=len(methods)) if len(methods) > 1 else np.array([0.0])
    box_width = 0.33 if len(methods) > 1 else 0.45
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(16.2, 3.95))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.0, 1.0, 1.0],
        wspace=0.22,
    )
    left_ax = fig.add_subplot(gs[0, 0])
    top_left_ax = fig.add_subplot(gs[0, 1])
    top_right_ax = fig.add_subplot(gs[0, 2])

    plot_mca_df = mca_df.loc[mca_df["method"].isin(methods)].copy()
    for method, offset in zip(methods, method_offsets, strict=True):
        method_df = plot_mca_df.loc[plot_mca_df["method"].eq(method)]
        values = [
            method_df.loc[method_df["noise_level"] == noise_level, metric_col].to_numpy(dtype=float)
            for noise_level in noise_levels
        ]
        positions = base_positions + offset
        left_ax.boxplot(
            values,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": palette[method], "edgecolor": "0.25", "linewidth": 1.0},
            medianprops={"color": "0.1", "linewidth": 1.05},
            whiskerprops={"color": "0.35", "linewidth": 0.85},
            capprops={"color": "0.35", "linewidth": 0.85},
        )
        for xpos, vals in zip(positions, values, strict=True):
            if vals.size == 0:
                continue
            jitter = rng.normal(loc=0.0, scale=box_width * 0.13, size=vals.size)
            left_ax.scatter(
                np.full(vals.size, xpos) + jitter,
                vals,
                s=18,
                color=palette[method],
                alpha=1.0,
                linewidths=0,
                zorder=3,
            )
    left_ax.set_xticks(base_positions, noise_order)
    left_ax.set_xlabel("Noise level (FPR = FNR)", fontsize=label_fontsize)
    left_ax.set_ylabel(ylabel, fontsize=label_fontsize)
    left_ax.tick_params(axis="x", labelsize=tick_fontsize)
    left_ax.tick_params(axis="y", labelsize=tick_fontsize)
    clean_ax(left_ax)

    pathway_order = [pathway for pathway in pathways if pathway in set(cell_df["pathway_pretty"])]
    top_axes = [top_left_ax, top_right_ax]
    for col_idx, pathway in enumerate(pathway_order):
        ax = top_axes[col_idx]
        pathway_df = cell_df.loc[cell_df["pathway_pretty"].eq(pathway) & cell_df["method"].isin(methods)].copy()
        for method, offset in zip(methods, method_offsets, strict=True):
            method_df = pathway_df.loc[pathway_df["method"].eq(method)]
            values = [
                method_df.loc[method_df["noise_level"] == noise_level, "cell_type_aupr"].to_numpy(dtype=float)
                for noise_level in noise_levels
            ]
            positions = base_positions + offset
            ax.boxplot(
                values,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={"facecolor": palette[method], "edgecolor": "0.25", "linewidth": 1.0},
                medianprops={"color": "0.1", "linewidth": 1.05},
                whiskerprops={"color": "0.35", "linewidth": 0.85},
                capprops={"color": "0.35", "linewidth": 0.85},
            )
            for xpos, vals in zip(positions, values, strict=True):
                if vals.size == 0:
                    continue
                jitter = rng.normal(loc=0.0, scale=box_width * 0.13, size=vals.size)
                ax.scatter(
                    np.full(vals.size, xpos) + jitter,
                    vals,
                    s=16,
                    color=palette[method],
                    alpha=1.0,
                    linewidths=0,
                    zorder=3,
                )
        ax.set_title(pathway, fontsize=title_fontsize)
        ax.set_xticks(base_positions, noise_order)
        ax.set_xlabel("Noise level", fontsize=label_fontsize)
        ax.set_ylabel("Cell-type AUPR" if col_idx == 0 else "", fontsize=label_fontsize)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, num=6))
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        clean_ax(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[method],
            markeredgecolor=palette[method],
            markersize=8.0,
            label=method,
        )
        for method in methods
    ]
    legend = fig.legend(
        handles=handles,
        title="Method",
        frameon=False,
        bbox_to_anchor=(0.982, 0.5),
        loc="center right",
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    if legend is not None:
        legend.set_alignment("left")

    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.18, top=0.90)
    savefig(fig, path)
    plt.close(fig)


def save_reconstruction_pearson_plot(df: pd.DataFrame, *, path: Path) -> None:
    save_mca_ap_boxplot(
        df.rename(columns={"mean_gene_pearson": "average_precision_true"}),
        path=path,
        metric_col="average_precision_true",
        ylabel="Mean gene-wise Pearson (reconstruction)",
    )


def save_reconstruction_rmse_plot(df: pd.DataFrame, *, path: Path) -> None:
    save_mca_ap_boxplot(
        df.rename(columns={"rmse": "average_precision_true"}),
        path=path,
        metric_col="average_precision_true",
        ylabel="Reconstruction RMSE",
    )


def save_mca_aupr_vs_reconstruction_scatter(
    mca_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    *,
    path: Path,
) -> None:
    set_house_style()
    aupr_agg = (
        mca_df.groupby("method")["average_precision_true"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "aupr_mean", "std": "aupr_sd"})
    )
    recon_agg = (
        recon_df.groupby("method")["mean_gene_pearson"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "pearson_mean", "std": "pearson_sd"})
    )
    summary = aupr_agg.join(recon_agg, how="inner").reset_index()
    if summary.empty:
        return

    methods = [m for m in METHOD_ORDER if m in summary["method"].values]
    n_methods = max(3, len(methods))
    default_colors = sns.color_palette(CAT_PALETTE, n_colors=n_methods)
    preferred_palette = {
        "MOFA-FLEX": default_colors[0],
        "ExpiMap": default_colors[1],
        "Spectra": default_colors[2],
    }
    palette = {
        m: preferred_palette.get(m, default_colors[i % n_methods])
        for i, m in enumerate(methods)
    }

    label_fontsize = 13.0
    tick_fontsize = 11.5
    legend_fontsize = 11.5
    legend_title_fontsize = 12.5

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for method in methods:
        row = summary.loc[summary["method"] == method].iloc[0]
        color = palette[method]
        ax.errorbar(
            row["pearson_mean"],
            row["aupr_mean"],
            xerr=row["pearson_sd"],
            yerr=row["aupr_sd"],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=8,
            capsize=4,
            linewidth=1.4,
            label=method,
        )
        label_offsets = {
            "ExpiMap": (6, -12),
            "TopicVI": (6, 8),
        }
        dx, dy = label_offsets.get(method, (6, 0))
        ax.annotate(
            method,
            xy=(row["pearson_mean"], row["aupr_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.5,
            color=color,
            va="center",
        )

    ax.set_xlabel("Mean gene-wise Pearson (reconstruction)", fontsize=label_fontsize)
    ax.set_ylabel("Mean AUPR (MCA gene programs)", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    clean_ax(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=palette[m], markeredgecolor=palette[m],
               markersize=7, label=m)
        for m in methods
    ]
    ax.legend(
        handles=handles,
        title="Method",
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    fig.subplots_adjust(left=0.14, right=0.78, bottom=0.14, top=0.97)
    savefig(fig, path)
    plt.close(fig)


def save_mca_aupr_vs_rmse_scatter(
    mca_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    *,
    path: Path,
) -> None:
    set_house_style()
    aupr_agg = (
        mca_df.groupby("method")["average_precision_true"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "aupr_mean", "std": "aupr_sd"})
    )
    recon_agg = (
        recon_df.groupby("method")["rmse"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "rmse_mean", "std": "rmse_sd"})
    )
    summary = aupr_agg.join(recon_agg, how="inner").reset_index()
    if summary.empty:
        return

    methods = [m for m in METHOD_ORDER if m in summary["method"].values]
    n_methods = max(3, len(methods))
    default_colors = sns.color_palette(CAT_PALETTE, n_colors=n_methods)
    preferred_palette = {
        "MOFA-FLEX": default_colors[0],
        "ExpiMap": default_colors[1],
        "Spectra": default_colors[2],
    }
    palette = {
        m: preferred_palette.get(m, default_colors[i % n_methods])
        for i, m in enumerate(methods)
    }

    label_fontsize = 13.0
    tick_fontsize = 11.5
    legend_fontsize = 11.5
    legend_title_fontsize = 12.5

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for method in methods:
        row = summary.loc[summary["method"] == method].iloc[0]
        color = palette[method]
        ax.errorbar(
            row["rmse_mean"],
            row["aupr_mean"],
            xerr=row["rmse_sd"],
            yerr=row["aupr_sd"],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=8,
            capsize=4,
            linewidth=1.4,
            label=method,
        )
        label_offsets = {
            "ExpiMap": (6, -12),
            "TopicVI": (6, 8),
        }
        dx, dy = label_offsets.get(method, (6, 0))
        ax.annotate(
            method,
            xy=(row["rmse_mean"], row["aupr_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.5,
            color=color,
            va="center",
        )

    ax.invert_xaxis()
    ax.set_xlabel("Mean reconstruction RMSE", fontsize=label_fontsize)
    ax.set_ylabel("Mean AUPR (MCA gene programs)", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    clean_ax(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=palette[m], markeredgecolor=palette[m],
               markersize=7, label=m)
        for m in methods
    ]
    ax.legend(
        handles=handles,
        title="Method",
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    fig.subplots_adjust(left=0.14, right=0.78, bottom=0.14, top=0.97)
    savefig(fig, path)
    plt.close(fig)


def save_celltype_aupr_overall_plot(df: pd.DataFrame, *, path: Path) -> None:
    plot_df = df.rename(columns={"mean_cell_type_aupr": "average_precision_true"})
    save_mca_ap_boxplot(
        plot_df,
        path=path,
        metric_col="average_precision_true",
        ylabel="Mean cell-type AUPR",
    )


def _noise_from_run_dir(run_dir: Path) -> float:
    return float(run_dir.parent.name.replace("noise_", "").replace("p", "."))


def _seed_from_run_dir(run_dir: Path) -> int:
    return int(run_dir.name.replace("seed_", ""))


def _mean_mca_ap(weights: pd.DataFrame, true_mask: pd.DataFrame, mca_pathways: list[str]) -> float:
    scores = []
    common_columns = weights.columns.intersection(true_mask.columns)
    for pathway_name in mca_pathways:
        if pathway_name not in weights.index:
            scores.append(0.0)
            continue
        y_true = true_mask.loc[pathway_name, common_columns].to_numpy(dtype=bool)
        y_score = weights.loc[pathway_name, common_columns].abs().to_numpy(dtype=float)
        scores.append(float(average_precision_score(y_true, y_score)))
    return float(np.mean(scores)) if scores else np.nan


def load_spectra_matching_strategy_comparison(*, baselines_dir: Path) -> pd.DataFrame:
    spectra_dir = baselines_dir / "spectra"
    mdata = load_sln208_mdata(DEFAULT_DATA_PATH)
    var_names = pd.Index(mdata.mod["rna"].var_names)
    true_mask = bool_mask_from_gene_set_stats(build_paper_hallmark_mca_gene_set_stats(var_names), var_names)
    mca_pathways = [name for name in true_mask.index if str(name).startswith("HAN_")]

    rows = []
    for raw_weights_path in sorted(spectra_dir.glob("noise_*/seed_*/weights_raw.csv")):
        run_dir = raw_weights_path.parent
        noisy_mask_path = run_dir / "noisy_mask.csv"
        label_path = run_dir / "spectra_overlap_labels.csv"
        if not noisy_mask_path.exists():
            continue
        noisy_mask = (
            pd.read_csv(noisy_mask_path, index_col=0)
            .astype(bool)
            .reindex(columns=true_mask.columns, fill_value=False)
        )
        raw_weights = (
            pd.read_csv(raw_weights_path, index_col=0)
            .astype(float)
            .reindex(columns=true_mask.columns, fill_value=0.0)
            .fillna(0.0)
        )

        aupr_weights, _assignment = align_weights_to_prior_mask(raw_weights, noisy_mask)
        rows.append(
            {
                "noise_level": _noise_from_run_dir(run_dir),
                "seed": _seed_from_run_dir(run_dir),
                "average_precision_true": _mean_mca_ap(aupr_weights, true_mask, mca_pathways),
                "matching_strategy": "One-to-one AUPR assignment",
            }
        )

        if not label_path.exists():
            continue
        labels = pd.read_csv(label_path)
        overlap_weights, _assignment = align_weights_by_spectra_overlap_labels(raw_weights, noisy_mask, labels)
        rows.append(
            {
                "noise_level": _noise_from_run_dir(run_dir),
                "seed": _seed_from_run_dir(run_dir),
                "average_precision_true": _mean_mca_ap(overlap_weights, true_mask, mca_pathways),
                "matching_strategy": "Spectra overlap label",
            }
        )

    if not rows:
        raise FileNotFoundError(f"No Spectra raw weight tables found under {spectra_dir}")

    return pd.DataFrame.from_records(rows).sort_values(["noise_level", "matching_strategy", "seed"]).reset_index(drop=True)


def save_spectra_matching_strategy_boxplot(df: pd.DataFrame, *, path: Path) -> None:
    plot_df = df.rename(columns={"matching_strategy": "method"})
    save_mca_ap_boxplot(
        plot_df,
        path=path,
        metric_col="average_precision_true",
        ylabel="AUPR (MCA gene programs)",
    )


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir or (args.baselines_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if "topicvi" in args.methods:
        _generate_topicvi_mca_summary(args.baselines_dir)

    df = load_mca_ap_comparison(
        mofaflex_dir=args.mofaflex_dir,
        baselines_dir=args.baselines_dir,
        methods=list(args.methods),
    )
    df.to_csv(out_dir / "mca_ap_true_by_method_per_seed.csv", index=False)
    save_mca_ap_boxplot(
        df,
        path=out_dir / "mca_ap_true_by_method_boxplot.png",
        metric_col="average_precision_true",
        ylabel="AUPR (MCA gene programs)",
    )
    df.to_csv(out_dir / "mca_ap_noisy_by_method_per_seed.csv", index=False)
    save_mca_ap_boxplot(
        df,
        path=out_dir / "mca_ap_noisy_by_method_boxplot.png",
        metric_col="average_precision_noisy",
        ylabel="AUPR vs noisy MCA priors",
    )

    if (args.baselines_dir / "spectra" / "summary_mca_only_per_run.csv").exists():
        spectra_matching_df = load_spectra_matching_strategy_comparison(baselines_dir=args.baselines_dir)
        spectra_matching_df.to_csv(out_dir / "spectra_mca_ap_true_by_matching_strategy_per_seed.csv", index=False)
        save_spectra_matching_strategy_boxplot(
            spectra_matching_df,
            path=out_dir / "spectra_mca_ap_true_by_matching_strategy_boxplot.png",
        )

    recon_df = load_reconstruction_comparison(
        mofaflex_dir=args.mofaflex_dir,
        baselines_dir=args.baselines_dir,
        methods=list(args.methods),
    )
    recon_df.to_csv(out_dir / "reconstruction_mean_gene_pearson_per_seed.csv", index=False)
    save_reconstruction_pearson_plot(recon_df, path=out_dir / "reconstruction_mean_gene_pearson_boxplot.png")
    save_mca_aupr_vs_reconstruction_scatter(
        df, recon_df, path=out_dir / "mca_aupr_vs_reconstruction_mean_gene_pearson_scatter.png"
    )
    save_reconstruction_rmse_plot(recon_df, path=out_dir / "reconstruction_rmse_boxplot.png")
    save_mca_aupr_vs_rmse_scatter(df, recon_df, path=out_dir / "mca_aupr_vs_reconstruction_rmse_scatter.png")

    celltype_by_pathway, celltype_overall = load_celltype_aupr_comparison(
        mofaflex_dir=args.mofaflex_dir,
        baselines_dir=args.baselines_dir,
        methods=list(args.methods),
    )
    celltype_by_pathway.to_csv(out_dir / "mca_celltype_aupr_by_pathway_per_seed.csv", index=False)
    celltype_overall.to_csv(out_dir / "mca_celltype_aupr_overall_per_seed.csv", index=False)
    save_celltype_aupr_by_pathway_plot(
        celltype_by_pathway,
        path=out_dir / "mca_celltype_aupr_by_pathway_boxplot.png",
    )
    save_celltype_aupr_overall_plot(
        celltype_overall,
        path=out_dir / "mca_celltype_aupr_overall_by_method_boxplot.png",
    )


if __name__ == "__main__":
    main()
