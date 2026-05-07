from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.mca_filter_sensitivity import (
        DEFAULT_DATA_PATH,
        DEFAULT_INIT_FACTORS,
        DEFAULT_INIT_SCALE,
        DEFAULT_LIKELIHOODS,
        DEFAULT_LR,
        DEFAULT_MCA_GMT_URL,
        DEFAULT_MODALITIES,
        DEFAULT_NONNEGATIVE_FACTORS,
        DEFAULT_NONNEGATIVE_WEIGHTS,
        DEFAULT_N_PARTICLES,
        DEFAULT_WEIGHT_PRIOR,
        _to_upper_feature_sets,
        attach_gene_set_mask,
        build_feature_set_collection,
        load_sln208_mdata,
        resolve_likelihoods_for_mdata,
    )
    from mofaflex_sensitivity.uninformed_sensitivity import fit_mofaflex_model
    from mofaflex_sensitivity.plot_style import CAT_PALETTE, clean_ax, savefig, set_house_style
else:
    from .mca_filter_sensitivity import (
        DEFAULT_DATA_PATH,
        DEFAULT_INIT_FACTORS,
        DEFAULT_INIT_SCALE,
        DEFAULT_LIKELIHOODS,
        DEFAULT_LR,
        DEFAULT_MCA_GMT_URL,
        DEFAULT_MODALITIES,
        DEFAULT_NONNEGATIVE_FACTORS,
        DEFAULT_NONNEGATIVE_WEIGHTS,
        DEFAULT_N_PARTICLES,
        DEFAULT_WEIGHT_PRIOR,
        _to_upper_feature_sets,
        attach_gene_set_mask,
        build_feature_set_collection,
        load_sln208_mdata,
        resolve_likelihoods_for_mdata,
    )
    from .uninformed_sensitivity import fit_mofaflex_model
    from .plot_style import CAT_PALETTE, clean_ax, savefig, set_house_style


DEFAULT_OUT_DIR = Path("artifacts/sln208_prior_noise_refinement")
DEFAULT_MSIGDB_CATEGORY = "mh.all"
DEFAULT_MSIGDB_DBVER = "2023.2.Mm"
DEFAULT_GENE_SET_MIN_FRACTION = 0.1
DEFAULT_GENE_SET_MIN_COUNT = 15
DEFAULT_GENE_SET_MAX_COUNT = 300
DEFAULT_GENE_SET_SIMILARITY_THRESHOLD = 0.8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Perturb Hallmark+MCA prior annotations on SLN208 with false positives/false negatives, "
            "train MOFA-FLEX on the noisy prior, and score whether learned RNA weights refine back "
            "towards the true prior."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fpr", type=float, default=0.2, help="False-positive rate relative to true active genes.")
    parser.add_argument("--fnr", type=float, default=0.2, help="False-negative rate relative to true active genes.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--noise-seed", type=int, default=42, help="Noise-mask seed.")
    parser.add_argument("--n-uninformed-factors", type=int, default=3)
    parser.add_argument("--annotation-confidence", type=float, default=0.99)
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--early-stopper-patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--n-particles", type=int, default=DEFAULT_N_PARTICLES)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=sorted(DEFAULT_LIKELIHOODS),
        default=list(DEFAULT_MODALITIES),
        help="Modalities to include when fitting MOFA-FLEX.",
    )
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    return parser


def infer_pathway_source(pathway_name: str) -> str:
    upper_name = str(pathway_name).upper()
    if upper_name.startswith("HALLMARK"):
        return "hallmark"
    if upper_name.startswith("HAN_") or upper_name.startswith("MCA"):
        return "mca"
    return "unknown"


def fit_refinement_model(
    *,
    mdata,
    n_uninformed_factors: int,
    annotation_confidence: float,
    seed: int,
    max_epochs: int,
    early_stopper_patience: int,
    lr: float,
    n_particles: int,
    save_path: str | Path | bool = False,
):
    import mofaflex as mfl

    data_options = mfl.DataOptions(
        group_by=None,
        scale_per_group=False,
        annotations_varm_key={"rna": "gene_set_mask"},
        plot_data_overview=False,
    )
    model_options = mfl.ModelOptions(
        n_factors=n_uninformed_factors,
        weight_prior=DEFAULT_WEIGHT_PRIOR,
        likelihoods=resolve_likelihoods_for_mdata(mdata),
        nonnegative_weights=DEFAULT_NONNEGATIVE_WEIGHTS,
        nonnegative_factors=DEFAULT_NONNEGATIVE_FACTORS,
        annotation_confidence=annotation_confidence,
        init_factors=DEFAULT_INIT_FACTORS,
        init_scale=DEFAULT_INIT_SCALE,
    )
    training_options = mfl.TrainingOptions(
        lr=lr,
        max_epochs=max_epochs,
        early_stopper_patience=early_stopper_patience,
        n_particles=n_particles,
        save_path=save_path,
        seed=seed,
    )
    return fit_mofaflex_model(mdata, data_options, model_options, training_options)


def build_paper_hallmark_mca_gene_set_stats(var_names: pd.Index) -> pd.DataFrame:
    import mofaflex as mfl

    with tempfile.NamedTemporaryFile("w", suffix=".gmt", delete=True) as tmp:
        with fsspec.open(DEFAULT_MCA_GMT_URL, mode="rt", client_kwargs={"trust_env": True}) as handle:
            tmp.write(handle.read())
            tmp.flush()
        mca_collection = _to_upper_feature_sets(
            mfl,
            mfl.FeatureSets.from_gmt(tmp.name, name="mca"),
        ).filter(
            var_names,
            min_fraction=DEFAULT_GENE_SET_MIN_FRACTION,
            min_count=DEFAULT_GENE_SET_MIN_COUNT,
            max_count=DEFAULT_GENE_SET_MAX_COUNT,
        )

    hallmark_collection = _to_upper_feature_sets(
        mfl,
        mfl.tl.msigdb_get_features(category=DEFAULT_MSIGDB_CATEGORY, dbver=DEFAULT_MSIGDB_DBVER),
    ).filter(
        var_names,
        min_fraction=DEFAULT_GENE_SET_MIN_FRACTION,
            min_count=DEFAULT_GENE_SET_MIN_COUNT,
            max_count=DEFAULT_GENE_SET_MAX_COUNT,
        )

    gene_set_collection = hallmark_collection | mca_collection
    gene_set_collection = gene_set_collection.merge_similar(
        metric="jaccard",
        similarity_threshold=DEFAULT_GENE_SET_SIMILARITY_THRESHOLD,
        iteratively=True,
    )

    records: list[dict[str, Any]] = []
    for feature_set in gene_set_collection:
        name = str(feature_set.name)
        records.append(
            {
                "pathway_name": name,
                "source": infer_pathway_source(name),
                "set_size": int(len(feature_set)),
                "features": list(feature_set),
            }
        )
    stats = pd.DataFrame.from_records(records)
    return stats.sort_values(["set_size", "pathway_name"], ascending=[True, True]).reset_index(drop=True)


def bool_mask_from_gene_set_stats(gene_set_stats: pd.DataFrame, var_names: pd.Index) -> pd.DataFrame:
    collection = build_feature_set_collection(gene_set_stats)
    mask = collection.to_mask(var_names.tolist())
    return mask.astype(bool)


def perturb_mask(
    true_mask: pd.DataFrame,
    *,
    fpr: float,
    fnr: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    noisy = true_mask.copy().astype(bool)
    records: list[dict[str, Any]] = []

    for pathway_name in true_mask.index:
        true_row = true_mask.loc[pathway_name].to_numpy(dtype=bool)
        active_idx = np.flatnonzero(true_row)
        inactive_idx = np.flatnonzero(~true_row)

        n_fp = min(len(inactive_idx), int(fpr * len(active_idx)))
        n_fn = min(len(active_idx), int(fnr * len(active_idx)))

        fp_idx = rng.choice(inactive_idx, size=n_fp, replace=False) if n_fp > 0 else np.array([], dtype=int)
        fn_idx = rng.choice(active_idx, size=n_fn, replace=False) if n_fn > 0 else np.array([], dtype=int)

        noisy_row = true_row.copy()
        noisy_row[fp_idx] = True
        noisy_row[fn_idx] = False
        noisy.loc[pathway_name] = noisy_row

        records.append(
            {
                "pathway_name": pathway_name,
                "n_true_active": int(true_row.sum()),
                "n_noisy_active": int(noisy_row.sum()),
                "n_false_positives_added": int(len(fp_idx)),
                "n_false_negatives_removed": int(len(fn_idx)),
            }
        )

    return noisy, pd.DataFrame.from_records(records)


def noisy_gene_set_stats_from_mask(true_stats: pd.DataFrame, noisy_mask: pd.DataFrame) -> pd.DataFrame:
    source_by_pathway = true_stats.set_index("pathway_name")["source"].to_dict()
    records: list[dict[str, Any]] = []
    for pathway_name in noisy_mask.index:
        features = noisy_mask.columns[noisy_mask.loc[pathway_name].to_numpy(dtype=bool)].tolist()
        records.append(
            {
                "pathway_name": pathway_name,
                "source": source_by_pathway[pathway_name],
                "set_size": int(len(features)),
                "features": features,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["set_size", "pathway_name"], ascending=[True, True]).reset_index(
        drop=True
    )


def best_f1_threshold(mask: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    thresholds = np.unique(np.quantile(unique_scores, np.linspace(0.0, 1.0, num=min(200, unique_scores.size))))
    best = (0.0, 0.0, 0.0, float(thresholds[0]))
    for threshold in thresholds:
        predicted = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            mask,
            predicted,
            average="binary",
            zero_division=0,
        )
        if f1 > best[2]:
            best = (float(precision), float(recall), float(f1), float(threshold))
    return best


def top_k_overlap_metrics(mask: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float, float]:
    k = int(max(0, min(k, scores.size)))
    if k == 0:
        return 0.0, 0.0, 0.0
    order = np.argsort(scores)[::-1]
    predicted = np.zeros_like(mask, dtype=bool)
    predicted[order[:k]] = True
    tp = float(np.sum(predicted & mask))
    precision = tp / float(np.sum(predicted)) if np.sum(predicted) > 0 else 0.0
    recall = tp / float(np.sum(mask)) if np.sum(mask) > 0 else 0.0
    union = float(np.sum(predicted | mask))
    jaccard = tp / union if union > 0 else 0.0
    return precision, recall, jaccard


def summarize_refinement(
    *,
    model,
    true_mask: pd.DataFrame,
    noisy_mask: pd.DataFrame,
    noise_stats: pd.DataFrame,
) -> pd.DataFrame:
    weights = model.get_weights(return_type="pandas")["rna"].astype(float)
    informed_names = list(
        model.factor_names[model.n_uninformed_factors : model.n_uninformed_factors + model.n_informed_factors]
    )
    common = [name for name in informed_names if name in true_mask.index and name in weights.index]
    noise_stats = noise_stats.set_index("pathway_name")

    records: list[dict[str, Any]] = []
    for pathway_name in common:
        scores = weights.loc[pathway_name].abs().reindex(true_mask.columns).to_numpy(dtype=float)
        true_binary = true_mask.loc[pathway_name].to_numpy(dtype=bool)
        noisy_binary = noisy_mask.loc[pathway_name].to_numpy(dtype=bool)

        ap_true = float(average_precision_score(true_binary, scores))
        ap_noisy = float(average_precision_score(noisy_binary, scores))

        precision_true, recall_true, f1_true, threshold_true = best_f1_threshold(true_binary, scores)
        precision_noisy, recall_noisy, f1_noisy, threshold_noisy = best_f1_threshold(noisy_binary, scores)

        k_true = int(true_binary.sum())
        prec_top_true, rec_top_true, jac_top_true = top_k_overlap_metrics(true_binary, scores, k_true)
        prec_top_noisy, rec_top_noisy, jac_top_noisy = top_k_overlap_metrics(noisy_binary, scores, k_true)

        false_positive_mask = noisy_binary & ~true_binary
        false_negative_mask = true_binary & ~noisy_binary
        inactive_mask = ~true_binary

        records.append(
            {
                "pathway_name": pathway_name,
                "source": infer_pathway_source(pathway_name),
                "n_true_active": int(true_binary.sum()),
                "n_noisy_active": int(noisy_binary.sum()),
                "n_false_positives_added": int(noise_stats.loc[pathway_name, "n_false_positives_added"]),
                "n_false_negatives_removed": int(noise_stats.loc[pathway_name, "n_false_negatives_removed"]),
                "average_precision_true": ap_true,
                "average_precision_noisy": ap_noisy,
                "average_precision_gain_true_minus_noisy": ap_true - ap_noisy,
                "best_f1_true": f1_true,
                "best_f1_noisy": f1_noisy,
                "best_f1_gain_true_minus_noisy": f1_true - f1_noisy,
                "best_threshold_true": threshold_true,
                "best_threshold_noisy": threshold_noisy,
                "precision_true": precision_true,
                "recall_true": recall_true,
                "precision_noisy": precision_noisy,
                "recall_noisy": recall_noisy,
                "top_true_size_precision_true": prec_top_true,
                "top_true_size_recall_true": rec_top_true,
                "top_true_size_jaccard_true": jac_top_true,
                "top_true_size_precision_noisy": prec_top_noisy,
                "top_true_size_recall_noisy": rec_top_noisy,
                "top_true_size_jaccard_noisy": jac_top_noisy,
                "mean_abs_weight_true": float(scores[true_binary].mean()) if true_binary.any() else np.nan,
                "mean_abs_weight_noisy": float(scores[noisy_binary].mean()) if noisy_binary.any() else np.nan,
                "mean_abs_weight_false_positives": float(scores[false_positive_mask].mean())
                if false_positive_mask.any()
                else np.nan,
                "mean_abs_weight_false_negatives": float(scores[false_negative_mask].mean())
                if false_negative_mask.any()
                else np.nan,
                "mean_abs_weight_inactive": float(scores[inactive_mask].mean()) if inactive_mask.any() else np.nan,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("average_precision_true", ascending=False).reset_index(drop=True)


def summarize_refinement_table(refinement_df: pd.DataFrame) -> dict[str, Any]:
    if refinement_df.empty:
        return {
            "n_pathways_evaluated": 0,
        }
    return {
        "n_pathways_evaluated": int(len(refinement_df)),
        "mean_average_precision_true": float(refinement_df["average_precision_true"].mean()),
        "mean_average_precision_noisy": float(refinement_df["average_precision_noisy"].mean()),
        "median_average_precision_true": float(refinement_df["average_precision_true"].median()),
        "median_average_precision_noisy": float(refinement_df["average_precision_noisy"].median()),
        "fraction_pathways_ap_true_gt_noisy": float(
            (refinement_df["average_precision_true"] > refinement_df["average_precision_noisy"]).mean()
        ),
        "fraction_pathways_best_f1_true_gt_noisy": float(
            (refinement_df["best_f1_true"] > refinement_df["best_f1_noisy"]).mean()
        ),
        "mean_ap_gain_true_minus_noisy": float(refinement_df["average_precision_gain_true_minus_noisy"].mean()),
        "mean_best_f1_gain_true_minus_noisy": float(refinement_df["best_f1_gain_true_minus_noisy"].mean()),
        "mean_top_true_size_jaccard_true": float(refinement_df["top_true_size_jaccard_true"].mean()),
        "mean_top_true_size_jaccard_noisy": float(refinement_df["top_true_size_jaccard_noisy"].mean()),
    }


def save_refinement_scatter(refinement_df: pd.DataFrame, *, path: Path) -> None:
    if refinement_df.empty:
        return
    set_house_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    source_colors = dict(zip(["mca", "hallmark"], sns.color_palette(CAT_PALETTE, n_colors=2), strict=True))
    colors = refinement_df["source"].map(source_colors).fillna("0.5")
    ax.scatter(
        refinement_df["average_precision_noisy"],
        refinement_df["average_precision_true"],
        c=colors,
        alpha=0.8,
        s=28,
    )
    lo = float(min(refinement_df["average_precision_noisy"].min(), refinement_df["average_precision_true"].min()))
    hi = float(max(refinement_df["average_precision_noisy"].max(), refinement_df["average_precision_true"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Average precision vs noisy prior")
    ax.set_ylabel("Average precision vs true prior")
    ax.set_title("Pathway refinement after training")
    clean_ax(ax)
    fig.tight_layout()
    savefig(fig, path)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    mdata = load_sln208_mdata(args.data_path, modalities=args.modalities)
    true_gene_set_stats = build_paper_hallmark_mca_gene_set_stats(pd.Index(mdata.mod["rna"].var_names))
    true_mask = bool_mask_from_gene_set_stats(true_gene_set_stats, pd.Index(mdata.mod["rna"].var_names))
    noisy_mask, noise_stats = perturb_mask(true_mask, fpr=args.fpr, fnr=args.fnr, seed=args.noise_seed)
    noisy_gene_set_stats = noisy_gene_set_stats_from_mask(true_gene_set_stats, noisy_mask)

    run_mdata = attach_gene_set_mask(mdata, noisy_gene_set_stats)
    model_filename = getattr(
        args,
        "model_filename",
        f"seed_{args.seed:03d}_fpr{args.fpr:.3f}_fnr{args.fnr:.3f}.h5",
    )
    model_path = models_dir / model_filename
    model = fit_refinement_model(
        mdata=run_mdata,
        n_uninformed_factors=args.n_uninformed_factors,
        annotation_confidence=args.annotation_confidence,
        seed=args.seed,
        max_epochs=args.max_epochs,
        early_stopper_patience=args.early_stopper_patience,
        lr=args.lr,
        n_particles=args.n_particles,
        save_path=model_path if args.save_model else False,
    )

    refinement_df = summarize_refinement(
        model=model,
        true_mask=true_mask,
        noisy_mask=noisy_mask,
        noise_stats=noise_stats,
    )
    summary = summarize_refinement_table(refinement_df)

    true_mask.to_csv(out_dir / "true_mask.csv")
    noisy_mask.to_csv(out_dir / "noisy_mask.csv")
    noise_stats.to_csv(out_dir / "noise_stats.csv", index=False)
    true_gene_set_stats.to_csv(out_dir / "true_gene_set_stats.csv", index=False)
    noisy_gene_set_stats.to_csv(out_dir / "noisy_gene_set_stats.csv", index=False)
    refinement_df.to_csv(out_dir / "pathway_refinement.csv", index=False)
    save_refinement_scatter(refinement_df, path=plots_dir / "refinement_ap_scatter.png")

    payload = {
        "data_path": str(args.data_path),
        "out_dir": str(out_dir),
        "modalities": list(mdata.mod.keys()),
        "fpr": float(args.fpr),
        "fnr": float(args.fnr),
        "seed": int(args.seed),
        "noise_seed": int(args.noise_seed),
        "n_uninformed_factors": int(args.n_uninformed_factors),
        "annotation_confidence": float(args.annotation_confidence),
        "n_true_gene_sets": int(len(true_gene_set_stats)),
        "n_hallmark_gene_sets": int((true_gene_set_stats["source"] == "hallmark").sum()),
        "n_mca_gene_sets": int((true_gene_set_stats["source"] == "mca").sum()),
        "model_path": str(model_path) if args.save_model else "",
        **summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (out_dir / "resolved_run.json").write_text(
        json.dumps(
            {
                "data_path": str(args.data_path),
                "out_dir": str(out_dir),
                "fpr": float(args.fpr),
                "fnr": float(args.fnr),
                "seed": int(args.seed),
                "noise_seed": int(args.noise_seed),
                "n_uninformed_factors": int(args.n_uninformed_factors),
                "annotation_confidence": float(args.annotation_confidence),
                "max_epochs": int(args.max_epochs),
                "early_stopper_patience": int(args.early_stopper_patience),
                "lr": float(args.lr),
                "n_particles": int(args.n_particles),
                "modalities": list(mdata.mod.keys()),
                "likelihoods": resolve_likelihoods_for_mdata(mdata),
                "gene_set_source": "hallmark+mca",
                "msigdb_category": DEFAULT_MSIGDB_CATEGORY,
                "msigdb_dbver": DEFAULT_MSIGDB_DBVER,
                "gene_set_min_fraction": DEFAULT_GENE_SET_MIN_FRACTION,
                "gene_set_min_count": DEFAULT_GENE_SET_MIN_COUNT,
                "gene_set_max_count": DEFAULT_GENE_SET_MAX_COUNT,
                "gene_set_similarity_threshold": DEFAULT_GENE_SET_SIMILARITY_THRESHOLD,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return payload


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
