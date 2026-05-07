from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

from .config import PipelineConfig
from .uninformed_sensitivity import absolute_pearson, fit_mofaflex_model


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    import mofaflex as mfl
    import torch

    output_dir = Path(config.experiment.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = asdict(config)
    resolved_config["experiment"]["device"] = select_device(config.experiment.device, torch, mfl)
    write_json(output_dir / "resolved_config.json", resolved_config)

    rng = np.random.default_rng(config.experiment.seed)
    dg = build_data_generator(config)
    dg.generate(rng=rng, all_combs=config.synthetic.all_combinations)
    if config.synthetic.normalize_generated:
        dg.normalize(with_std=config.synthetic.normalize_with_std)
    dg.get_noisy_mask(
        rng=rng,
        noise_fraction=config.synthetic.noise_fraction,
        informed_view_indices=config.synthetic.informed_view_indices,
    )

    mdata = dg.to_mudata(noisy=True)
    if config.experiment.save_mudata:
        mdata.write(output_dir / "synthetic_data.h5mu")

    feature_offsets = [0, *np.cumsum(config.synthetic.n_features).tolist()]
    vlines = feature_offsets[1:-1]
    save_matrix_heatmap(
        dg._view_factor_mask.astype(int),
        plots_dir / "view_factor_mask.png",
        title="Synthetic view-factor mask",
        center=None,
        yticklabels=[f"view_{idx}" for idx in range(len(config.synthetic.n_features))],
        xticklabels=False,
        vlines=None,
        dpi=config.evaluation.dpi,
    )
    save_matrix_heatmap(
        dg.noisy_w_mask.astype(int) - dg.w_mask.astype(int),
        plots_dir / "noisy_mask_delta.png",
        title="Noisy prior mask minus true mask",
        center=0.0,
        xticklabels=False,
        yticklabels=True,
        vlines=vlines,
        dpi=config.evaluation.dpi,
    )

    model = fit_model(
        config=config,
        mdata=mdata,
        resolved_device=resolved_config["experiment"]["device"],
        save_path=(output_dir / "model.h5") if config.experiment.save_model else False,
    )

    save_plotnine(mfl.pl.factor_correlation(model), plots_dir / "factor_correlation.png", dpi=config.evaluation.dpi)
    save_plotnine(mfl.pl.variance_explained(model), plots_dir / "variance_explained.png", dpi=config.evaluation.dpi)

    truth = collect_ground_truth(model, dg, mdata)
    predictions = collect_predictions(model, sparse_type=config.evaluation.sparse_type)
    evaluation = evaluate_predictions(truth["w_true"], predictions["w_hat"], truth["y_true"], predictions["y_hat"])

    aligned_w_hat = predictions["w_hat"].iloc[evaluation["optimal_order"], :].copy()
    aligned_corr = evaluation["correlation_matrix"].iloc[:, evaluation["optimal_order"]].copy()

    truth["w_true"].to_csv(output_dir / "true_weights.csv")
    truth["w_mask"].to_csv(output_dir / "true_mask.csv")
    truth["noisy_w_mask"].to_csv(output_dir / "noisy_mask.csv")
    predictions["w_hat"].to_csv(output_dir / "estimated_weights.csv")
    aligned_w_hat.to_csv(output_dir / "aligned_estimated_weights.csv")
    truth["y_true"].to_csv(output_dir / "true_reconstructions.csv")
    predictions["y_hat"].to_csv(output_dir / "estimated_reconstructions.csv")
    evaluation["correlation_matrix"].to_csv(output_dir / "weight_correlation_matrix.csv")
    aligned_corr.to_csv(output_dir / "aligned_weight_correlation_matrix.csv")
    evaluation["matched_factor_correlations"].to_csv(output_dir / "matched_factor_correlations.csv", index=False)

    matched_true_mask = truth["w_mask"].iloc[evaluation["matched_true_rows"], :].copy()
    matched_abs_w_hat = predictions["w_hat"].iloc[evaluation["optimal_order"], :].abs().copy()
    matched_true_mask.to_csv(output_dir / "aligned_true_mask.csv")
    matched_abs_w_hat.to_csv(output_dir / "aligned_abs_estimated_weights.csv")

    mask_recovery = evaluate_mask_recovery(matched_true_mask, matched_abs_w_hat)
    mask_recovery["curve"].to_csv(output_dir / "mask_recovery_pr_curve.csv", index=False)

    save_matrix_heatmap(
        truth["w_true"],
        plots_dir / "true_weights.png",
        title="True weights",
        center=0.0,
        xticklabels=False,
        yticklabels=True,
        vlines=vlines,
        dpi=config.evaluation.dpi,
    )
    save_matrix_heatmap(
        predictions["w_hat"],
        plots_dir / "estimated_weights.png",
        title="Estimated weights",
        center=0.0,
        xticklabels=False,
        yticklabels=True,
        vlines=vlines,
        dpi=config.evaluation.dpi,
    )
    save_matrix_heatmap(
        aligned_w_hat,
        plots_dir / "aligned_estimated_weights.png",
        title="Aligned estimated weights",
        center=0.0,
        xticklabels=False,
        yticklabels=True,
        vlines=vlines,
        dpi=config.evaluation.dpi,
    )
    save_matrix_heatmap(
        evaluation["correlation_matrix"],
        plots_dir / "weight_correlation_heatmap.png",
        title="Absolute Pearson correlation between true and estimated weights",
        center=None,
        xticklabels=True,
        yticklabels=True,
        vlines=None,
        dpi=config.evaluation.dpi,
    )
    save_matrix_heatmap(
        matched_true_mask,
        plots_dir / "aligned_true_mask.png",
        title="Aligned true binary mask",
        center=None,
        xticklabels=False,
        yticklabels=True,
        vlines=vlines,
        dpi=config.evaluation.dpi,
    )
    save_matrix_heatmap(
        matched_abs_w_hat,
        plots_dir / "aligned_abs_estimated_weights.png",
        title="Aligned |estimated weights|",
        center=None,
        xticklabels=False,
        yticklabels=True,
        vlines=vlines,
        dpi=config.evaluation.dpi,
    )
    save_precision_recall_plot(
        mask_recovery["curve"],
        aupr_average_precision=mask_recovery["average_precision"],
        aupr_trapezoid=mask_recovery["aupr_trapezoid"],
        path=plots_dir / "mask_recovery_pr_curve.png",
        dpi=config.evaluation.dpi,
    )
    save_matrix_heatmap(
        aligned_corr,
        plots_dir / "aligned_weight_correlation_heatmap.png",
        title="Aligned weight correlation heatmap",
        center=None,
        xticklabels=True,
        yticklabels=True,
        vlines=None,
        dpi=config.evaluation.dpi,
    )

    for factor_idx in config.evaluation.factor_activity_indices:
        if factor_idx >= min(truth["w_true"].shape[0], aligned_w_hat.shape[0]):
            continue
        save_factor_activity_plot(
            factor_idx=factor_idx,
            w_true=truth["w_true"],
            w_hat_aligned=aligned_w_hat,
            w_mask=truth["w_mask"],
            noisy_w_mask=truth["noisy_w_mask"],
            vlines=vlines,
            path=plots_dir / f"factor_activity_{factor_idx:02d}.png",
            dpi=config.evaluation.dpi,
        )

    metrics = {
        "seed": config.experiment.seed,
        "device": resolved_config["experiment"]["device"],
        "n_samples": config.synthetic.n_samples,
        "n_views": len(config.synthetic.n_features),
        "n_features": config.synthetic.n_features,
        "n_true_factors": int(truth["w_true"].shape[0]),
        "n_model_factors": int(predictions["w_hat"].shape[0]),
        "reconstruction_rmse": evaluation["reconstruction_rmse"],
        "mean_matched_abs_weight_correlation": evaluation["mean_matched_abs_weight_correlation"],
        "median_matched_abs_weight_correlation": evaluation["median_matched_abs_weight_correlation"],
        "mask_recovery_average_precision": mask_recovery["average_precision"],
        "mask_recovery_aupr_trapezoid": mask_recovery["aupr_trapezoid"],
        "mask_recovery_positive_fraction": mask_recovery["positive_fraction"],
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def build_data_generator(config: PipelineConfig):
    import mofaflex as mfl

    return mfl.tl.DataGenerator(
        n_features=config.synthetic.n_features,
        n_samples=config.synthetic.n_samples,
        likelihoods=config.synthetic.likelihoods,
        n_fully_shared_factors=config.synthetic.n_fully_shared_factors,
        n_partially_shared_factors=config.synthetic.n_partially_shared_factors,
        n_private_factors=config.synthetic.n_private_factors,
        factor_size_dist=config.synthetic.factor_size_dist,
        nmf=resolve_nmf_by_view(config),
    )


def fit_model(config: PipelineConfig, mdata, resolved_device: str, save_path: Path | None):
    import mofaflex as mfl

    view_names = list(mdata.mod.keys())
    likelihoods = {view_name: likelihood for view_name, likelihood in zip(view_names, config.synthetic.likelihoods, strict=False)}
    nmf_enabled = any(resolve_nmf_by_view(config))

    data_opts = mfl.DataOptions(
        group_by=None,
        scale_per_group=True,
        annotations_varm_key={view_name: "w_mask" for view_name in view_names},
        covariates_obs_key=None,
        covariates_obsm_key=None,
        use_obs=None,
        use_var=None,
        plot_data_overview=False,
    )
    model_opts = mfl.ModelOptions(
        n_factors=config.model.n_factors,
        weight_prior=config.model.weight_prior,
        factor_prior=config.model.factor_prior,
        likelihoods=likelihoods,
        nonnegative_weights=config.model.nonnegative_weights or nmf_enabled,
        nonnegative_factors=config.model.nonnegative_factors or nmf_enabled,
        annotation_confidence=config.model.annotation_confidence,
        init_factors=config.model.init_factors,
        init_scale=config.model.init_scale,
    )
    train_opts = mfl.TrainingOptions(
        device=resolved_device,
        batch_size=config.training.batch_size,
        max_epochs=config.training.max_epochs,
        n_particles=config.training.n_particles,
        lr=config.training.lr,
        early_stopper_patience=config.training.early_stopper_patience,
        save_path=save_path,
        seed=config.experiment.seed,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    return fit_mofaflex_model(mdata, data_opts, model_opts, train_opts)


def collect_ground_truth(model, dg, mdata) -> dict[str, pd.DataFrame]:
    sample_names = model.sample_names[model.group_names[0]]
    factor_names = [f"true_factor_{idx + 1}" for idx in range(dg._ws[0].shape[0])]

    w_masks = []
    noisy_w_masks = []
    ws_true = []
    ys_true = []

    for view_index, view_name in enumerate(model.view_names):
        feature_names = model.feature_names[view_name]
        var_names = mdata[view_name].var_names

        w_masks.append(pd.DataFrame(dg._w_masks[view_index], index=factor_names, columns=var_names).loc[:, feature_names].copy())
        noisy_w_masks.append(
            pd.DataFrame(dg._noisy_w_masks[view_index], index=factor_names, columns=var_names).loc[:, feature_names].copy()
        )
        true_weights = pd.DataFrame(dg._ws[view_index], index=factor_names, columns=var_names).loc[:, feature_names].copy()
        ys_true.append(pd.DataFrame(dg._ys[view_index], index=sample_names, columns=var_names).loc[:, feature_names].copy())
        ws_true.append(true_weights)

    return {
        "w_mask": pd.concat(w_masks, axis=1),
        "noisy_w_mask": pd.concat(noisy_w_masks, axis=1),
        "w_true": pd.concat(ws_true, axis=1),
        "y_true": pd.concat(ys_true, axis=1),
    }


def collect_predictions(model, sparse_type: str) -> dict[str, pd.DataFrame]:
    weights = model.get_weights(return_type="pandas", sparse_type=sparse_type)
    factors = model.get_factors(return_type="pandas", sparse_type=sparse_type)
    group_name = model.group_names[0]
    z_hat = factors[group_name]
    ws_hat = [weights[view_name] for view_name in model.view_names]
    ys_hat = [z_hat @ view_weights for view_weights in ws_hat]

    return {
        "w_hat": pd.concat(ws_hat, axis=1),
        "z_hat": z_hat,
        "y_hat": pd.concat(ys_hat, axis=1),
    }


def evaluate_predictions(
    w_true: pd.DataFrame,
    w_hat: pd.DataFrame,
    y_true: pd.DataFrame,
    y_hat: pd.DataFrame,
) -> dict[str, Any]:
    corr = np.zeros((w_true.shape[0], w_hat.shape[0]), dtype=float)
    for true_idx, true_weights in enumerate(w_true.to_numpy()):
        for hat_idx, estimated_weights in enumerate(w_hat.to_numpy()):
            corr[true_idx, hat_idx] = absolute_pearson(true_weights, estimated_weights)

    row_ind, col_ind = linear_sum_assignment(-corr)
    sort_idx = np.argsort(row_ind)
    matched_rows = row_ind[sort_idx]
    optimal_order = col_ind[sort_idx]
    matched = pd.DataFrame(
        {
            "true_factor": w_true.index.to_numpy()[matched_rows],
            "estimated_factor": w_hat.index.to_numpy()[optimal_order],
            "abs_pearson": corr[matched_rows, optimal_order],
        }
    )

    return {
        "reconstruction_rmse": float(np.sqrt(np.mean(np.square(y_true.to_numpy() - y_hat.to_numpy())))),
        "correlation_matrix": pd.DataFrame(corr, index=w_true.index, columns=w_hat.index),
        "optimal_order": optimal_order.tolist(),
        "matched_true_rows": matched_rows.tolist(),
        "matched_factor_correlations": matched,
        "mean_matched_abs_weight_correlation": float(matched["abs_pearson"].mean()),
        "median_matched_abs_weight_correlation": float(matched["abs_pearson"].median()),
    }


def evaluate_mask_recovery(true_mask: pd.DataFrame, matched_abs_w_hat: pd.DataFrame) -> dict[str, Any]:
    labels = true_mask.to_numpy(dtype=int).ravel()
    scores = matched_abs_w_hat.to_numpy(dtype=float).ravel()
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    curve = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.r_[thresholds, np.nan],
        }
    )
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "aupr_trapezoid": float(auc(recall, precision)),
        "positive_fraction": float(labels.mean()),
        "curve": curve,
    }


def save_matrix_heatmap(
    matrix: pd.DataFrame | np.ndarray,
    path: Path,
    *,
    title: str,
    center: float | None,
    xticklabels: bool,
    yticklabels: bool | list[str],
    vlines: list[int] | None,
    dpi: int,
) -> None:
    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    fig_width = min(max(values.shape[1] / 40.0, 8.0), 24.0)
    fig_height = min(max(values.shape[0] / 6.0, 4.0), 16.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = "RdBu_r" if center is not None else "viridis"
    sns.heatmap(values, ax=ax, cmap=cmap, center=center, xticklabels=xticklabels, yticklabels=yticklabels, cbar=True)
    if isinstance(matrix, pd.DataFrame):
        if xticklabels:
            ax.set_xticklabels(matrix.columns, rotation=90)
        if yticklabels is True:
            ax.set_yticklabels(matrix.index, rotation=0)
    if vlines:
        for xpos in vlines:
            ax.axvline(x=xpos, color="black", linewidth=0.75)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_factor_activity_plot(
    *,
    factor_idx: int,
    w_true: pd.DataFrame,
    w_hat_aligned: pd.DataFrame,
    w_mask: pd.DataFrame,
    noisy_w_mask: pd.DataFrame,
    vlines: list[int],
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(18, 6), sharex=True)
    rows = [
        ("True weights", w_true.iloc[[factor_idx], :].to_numpy(), "RdBu_r", 0.0),
        ("Aligned estimated weights", w_hat_aligned.iloc[[factor_idx], :].to_numpy(), "RdBu_r", 0.0),
        ("True annotation mask", w_mask.iloc[[factor_idx], :].to_numpy().astype(float), "Greys", None),
        ("Noisy annotation mask", noisy_w_mask.iloc[[factor_idx], :].to_numpy().astype(float), "Greys", None),
    ]
    for ax, (label, values, cmap, center) in zip(axes, rows, strict=False):
        sns.heatmap(values, ax=ax, cmap=cmap, center=center, cbar=False, xticklabels=False, yticklabels=[label])
        for xpos in vlines:
            ax.axvline(x=xpos, color="black", linewidth=0.75)
        ax.set_ylabel("")
        ax.set_xlabel("")
    axes[0].set_title(f"Factor activity summary for factor index {factor_idx}")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plotnine(plot, path: Path, dpi: int) -> None:
    plot.save(filename=path, dpi=dpi, verbose=False)


def save_precision_recall_plot(
    curve: pd.DataFrame,
    *,
    aupr_average_precision: float,
    aupr_trapezoid: float,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curve["recall"], curve["precision"], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Mask recovery PR curve\nAP={aupr_average_precision:.4f}, AUPR={aupr_trapezoid:.4f}"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def select_device(preferred: str, torch_module, mofaflex_module) -> str:
    if preferred != "auto":
        return preferred
    if torch_module.cuda.is_available():
        return f"cuda:{mofaflex_module.tl.get_free_gpu_idx()}"
    return "cpu"


def resolve_nmf_by_view(config: PipelineConfig) -> list[bool]:
    n_views = len(config.synthetic.n_features)
    if isinstance(config.synthetic.nmf, bool):
        return [config.synthetic.nmf for _ in range(n_views)]
    return list(config.synthetic.nmf)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
