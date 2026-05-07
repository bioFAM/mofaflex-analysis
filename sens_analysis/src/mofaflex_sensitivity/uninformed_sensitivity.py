from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse


@dataclass
class SweepMetricSummary:
    seed: int
    n_uninformed_factors: int
    n_total_factors: int
    n_informed_factors: int
    total_r2_sum: float
    total_r2_mean: float
    uninformed_r2_sum: float
    informed_r2_sum: float
    uninformed_r2_fraction: float
    informed_r2_fraction: float
    top_uninformed_factor_r2: float
    matched_pcgse_rows: int
    matched_pcgse_significant_fraction: float | None
    matched_pcgse_median_neglog10_padj: float | None
    matched_pcgse_min_padj: float | None
    hidden_factor_recovery_mean_abs_corr: float | None = None
    hidden_factor_recovery_median_abs_corr: float | None = None
    observed_factor_recovery_mean_abs_corr: float | None = None
    observed_factor_recovery_median_abs_corr: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_mofaflex_model(mdata, data_options, model_options, training_options):
    import mofaflex as mfl
    import mofaflex._core.mofaflex as mfl_core

    # Small or degenerate runs can fail in the post-fit PCGSE diagnostic even
    # though the model itself finished training successfully.
    original_pcgse_test = mfl_core.pcgse_test

    def _safe_pcgse_test(*args, **kwargs):
        try:
            return original_pcgse_test(*args, **kwargs)
        except ValueError as exc:
            if "No objects to concatenate" in str(exc):
                return None
            raise

    mfl_core.pcgse_test = _safe_pcgse_test
    try:
        return mfl.MOFAFLEX(mdata, data_options, model_options, training_options)
    finally:
        mfl_core.pcgse_test = original_pcgse_test


def summarize_model_sensitivity(model, *, seed: int, n_uninformed_factors: int, alpha: float = 0.05) -> SweepMetricSummary:
    r2_total = model.get_r2(total=True)
    r2_by_group = model.get_r2(total=False, ordered=False)

    uninformed_names = list(model.factor_names[: model.n_uninformed_factors])
    informed_names = list(
        model.factor_names[model.n_uninformed_factors : model.n_uninformed_factors + model.n_informed_factors]
    )

    uninformed_r2_sum = 0.0
    informed_r2_sum = 0.0
    top_uninformed_factor_r2 = 0.0

    for group_df in r2_by_group.values():
        if uninformed_names:
            uninformed_df = group_df.loc[uninformed_names, :]
            uninformed_r2_sum += float(uninformed_df.to_numpy().sum())
            top_uninformed_factor_r2 = max(top_uninformed_factor_r2, float(uninformed_df.sum(axis=1).max()))
        if informed_names:
            informed_r2_sum += float(group_df.loc[informed_names, :].to_numpy().sum())

    total_r2_sum = float(r2_total.to_numpy().sum())
    total_r2_mean = float(r2_total.to_numpy().mean())
    uninformed_r2_fraction = float(uninformed_r2_sum / total_r2_sum) if total_r2_sum > 0 else 0.0
    informed_r2_fraction = float(informed_r2_sum / total_r2_sum) if total_r2_sum > 0 else 0.0

    matched_pcgse = matched_pcgse_table(model)
    if matched_pcgse.empty:
        matched_pcgse_significant_fraction = None
        matched_pcgse_median_neglog10_padj = None
        matched_pcgse_min_padj = None
    else:
        clipped_padj = matched_pcgse["padj"].clip(lower=1e-300)
        matched_pcgse_significant_fraction = float((matched_pcgse["padj"] < alpha).mean())
        matched_pcgse_median_neglog10_padj = float((-np.log10(clipped_padj)).median())
        matched_pcgse_min_padj = float(matched_pcgse["padj"].min())

    return SweepMetricSummary(
        seed=seed,
        n_uninformed_factors=n_uninformed_factors,
        n_total_factors=int(model.n_factors),
        n_informed_factors=int(model.n_informed_factors),
        total_r2_sum=total_r2_sum,
        total_r2_mean=total_r2_mean,
        uninformed_r2_sum=float(uninformed_r2_sum),
        informed_r2_sum=float(informed_r2_sum),
        uninformed_r2_fraction=uninformed_r2_fraction,
        informed_r2_fraction=informed_r2_fraction,
        top_uninformed_factor_r2=float(top_uninformed_factor_r2),
        matched_pcgse_rows=int(len(matched_pcgse)),
        matched_pcgse_significant_fraction=matched_pcgse_significant_fraction,
        matched_pcgse_median_neglog10_padj=matched_pcgse_median_neglog10_padj,
        matched_pcgse_min_padj=matched_pcgse_min_padj,
    )


def matched_pcgse_table(model) -> pd.DataFrame:
    pcgse = model.get_significant_factor_annotations()
    if not pcgse:
        return pd.DataFrame(
            columns=["view", "factor", "annotation", "p", "padj", "sign", "t"]
        )

    matched = []
    for view_name, view_df in pcgse.items():
        view_matched = view_df.loc[view_df["factor"] == view_df["annotation"], :].copy()
        if view_matched.empty:
            continue
        view_matched.insert(0, "view", view_name)
        matched.append(view_matched)

    if not matched:
        return pd.DataFrame(columns=["view", "factor", "annotation", "p", "padj", "sign", "t"])
    return pd.concat(matched, ignore_index=True)


def factor_total_r2_table(model) -> pd.DataFrame:
    r2_by_group = model.get_r2(total=False, ordered=False)
    per_factor_df = pd.DataFrame(index=pd.Index(model.factor_names, dtype="object"))
    for group_df in r2_by_group.values():
        for view_name in group_df.columns:
            column_name = f"r2_{view_name}"
            if column_name not in per_factor_df.columns:
                per_factor_df[column_name] = 0.0
            per_factor_df.loc[group_df.index, column_name] = (
                per_factor_df.loc[group_df.index, column_name].to_numpy(dtype=float)
                + group_df.loc[:, view_name].to_numpy(dtype=float)
            )
    if per_factor_df.empty:
        return pd.DataFrame(columns=["factor", "matched_variance_explained"])
    r2_columns = [col for col in per_factor_df.columns if col.startswith("r2_")]
    per_factor_df["matched_variance_explained"] = per_factor_df.loc[:, r2_columns].sum(axis=1) if r2_columns else 0.0
    return per_factor_df.rename_axis("factor").reset_index().sort_values("factor", kind="stable").reset_index(drop=True)


def matched_pcgse_activity_table(model) -> pd.DataFrame:
    matched = matched_pcgse_table(model).copy()
    if matched.empty:
        return matched.assign(matched_variance_explained=pd.Series(dtype=float))
    factor_r2 = factor_total_r2_table(model)
    return matched.merge(factor_r2, on="factor", how="left").fillna({"matched_variance_explained": 0.0})


def factor_subset_names(model, subset: str) -> list[str]:
    subset_lower = subset.lower()
    if subset_lower == "all":
        return list(model.factor_names)
    if subset_lower == "uninformed":
        return list(model.factor_names[: model.n_uninformed_factors])
    if subset_lower == "informed":
        start = model.n_uninformed_factors
        stop = model.n_uninformed_factors + model.n_informed_factors
        return list(model.factor_names[start:stop])
    raise ValueError(f"Unknown factor subset {subset!r}.")


def _to_dense_array(arr) -> np.ndarray:
    if issparse(arr):
        return arr.toarray()
    return np.asarray(arr)


def compute_processed_subset_r2(
    model,
    mdata,
    *,
    factor_names: list[str],
) -> dict[str, float]:
    dataset = model._mofaflexdataset(mdata)
    factors_by_group = model.get_factors(return_type="pandas", ordered=False)
    weights_by_view = model.get_weights(return_type="pandas", ordered=False)

    processed_data = dataset.apply(
        lambda adata, group_name, view_name: {
            "y_true": _to_dense_array(dataset.preprocessor(adata.X, slice(None), slice(None), group_name, view_name)[0]),
            "obs_names": adata.obs_names.to_numpy(),
            "var_names": adata.var_names.to_numpy(),
        }
    )

    ss_res_total = 0.0
    ss_tot_total = 0.0
    per_view_ss: dict[str, dict[str, float]] = {}

    for group_name, group_views in processed_data.items():
        factor_df = factors_by_group[group_name]
        for view_name, payload in group_views.items():
            y_true = np.asarray(payload["y_true"], dtype=float)
            obs_names = pd.Index(payload["obs_names"])
            var_names = pd.Index(payload["var_names"])

            if factor_names:
                factor_subset = factor_df.reindex(index=obs_names, columns=factor_names, fill_value=0.0)
                weight_subset = weights_by_view[view_name].reindex(
                    index=factor_names,
                    columns=var_names,
                    fill_value=0.0,
                )
                y_pred = factor_subset.to_numpy(dtype=float) @ weight_subset.to_numpy(dtype=float)
            else:
                y_pred = np.zeros_like(y_true, dtype=float)

            ss_res = float(np.nansum(np.square(y_true - y_pred)))
            ss_tot = float(np.nansum(np.square(y_true)))
            ss_res_total += ss_res
            ss_tot_total += ss_tot

            per_view_ss.setdefault(view_name, {"ss_res": 0.0, "ss_tot": 0.0})
            per_view_ss[view_name]["ss_res"] += ss_res
            per_view_ss[view_name]["ss_tot"] += ss_tot

    metrics = {
        "r2_total": float(1.0 - ss_res_total / ss_tot_total) if ss_tot_total > 0 else np.nan,
        "ss_res_total": float(ss_res_total),
        "ss_tot_total": float(ss_tot_total),
    }
    for view_name, values in per_view_ss.items():
        metrics[f"r2_{view_name}"] = (
            float(1.0 - values["ss_res"] / values["ss_tot"]) if values["ss_tot"] > 0 else np.nan
        )
        metrics[f"ss_res_{view_name}"] = float(values["ss_res"])
        metrics[f"ss_tot_{view_name}"] = float(values["ss_tot"])
    return metrics


def aggregate_sweep_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_factors, group in metrics.groupby("n_uninformed_factors", sort=True):
        row = {"n_uninformed_factors": int(n_factors), "n_runs": int(len(group))}
        for column in [
            "total_r2_sum",
            "total_r2_mean",
            "uninformed_r2_sum",
            "informed_r2_sum",
            "uninformed_r2_fraction",
            "informed_r2_fraction",
            "top_uninformed_factor_r2",
            "matched_pcgse_significant_fraction",
            "matched_pcgse_median_neglog10_padj",
            "hidden_factor_recovery_mean_abs_corr",
            "hidden_factor_recovery_median_abs_corr",
            "observed_factor_recovery_mean_abs_corr",
            "observed_factor_recovery_median_abs_corr",
        ]:
            series = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(series.mean()) if series.notna().any() else np.nan
            row[f"{column}_sd"] = float(series.std(ddof=1)) if series.notna().sum() > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("n_uninformed_factors").reset_index(drop=True)
    if not summary.empty:
        summary["delta_total_r2_sum_mean"] = summary["total_r2_sum_mean"].diff().fillna(summary["total_r2_sum_mean"])
        summary["delta_uninformed_r2_sum_mean"] = summary["uninformed_r2_sum_mean"].diff().fillna(
            summary["uninformed_r2_sum_mean"]
        )
    return summary


def smallest_within_one_sd(summary: pd.DataFrame, metric: str, maximize: bool = True) -> int | None:
    if summary.empty:
        return None
    mean_col = f"{metric}_mean"
    sd_col = f"{metric}_sd"
    valid = summary.loc[summary[mean_col].notna(), :].copy()
    if valid.empty:
        return None

    if maximize:
        best_idx = valid[mean_col].idxmax()
        threshold = float(valid.loc[best_idx, mean_col] - valid.loc[best_idx, sd_col])
        eligible = valid.loc[valid[mean_col] >= threshold, :]
    else:
        best_idx = valid[mean_col].idxmin()
        threshold = float(valid.loc[best_idx, mean_col] + valid.loc[best_idx, sd_col])
        eligible = valid.loc[valid[mean_col] <= threshold, :]

    if eligible.empty:
        return None
    return int(eligible["n_uninformed_factors"].min())


def save_sweep_tables(
    *,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path / "uninformed_factor_sweep_metrics.csv", index=False)
    summary.to_csv(output_path / "uninformed_factor_sweep_summary.csv", index=False)


def absolute_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denom = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denom == 0:
        return 0.0
    return float(abs(np.dot(left_centered, right_centered) / denom))


def match_factor_weights(true_weights: pd.DataFrame, estimated_weights: pd.DataFrame) -> dict[str, Any]:
    if true_weights.empty or estimated_weights.empty:
        return {
            "correlation_matrix": pd.DataFrame(index=true_weights.index, columns=estimated_weights.index, dtype=float),
            "matched_factor_correlations": pd.DataFrame(columns=["true_factor", "estimated_factor", "abs_pearson"]),
            "mean_abs_corr": np.nan,
            "median_abs_corr": np.nan,
            "optimal_order": [],
            "matched_true_rows": [],
        }

    corr = np.zeros((true_weights.shape[0], estimated_weights.shape[0]), dtype=float)
    for true_idx, true_row in enumerate(true_weights.to_numpy()):
        for est_idx, est_row in enumerate(estimated_weights.to_numpy()):
            corr[true_idx, est_idx] = absolute_pearson(true_row, est_row)

    row_ind, col_ind = linear_sum_assignment(-corr)
    sort_idx = np.argsort(row_ind)
    matched_rows = row_ind[sort_idx]
    optimal_order = col_ind[sort_idx]
    matched = pd.DataFrame(
        {
            "true_factor": true_weights.index.to_numpy()[matched_rows],
            "estimated_factor": estimated_weights.index.to_numpy()[optimal_order],
            "abs_pearson": corr[matched_rows, optimal_order],
        }
    )

    return {
        "correlation_matrix": pd.DataFrame(corr, index=true_weights.index, columns=estimated_weights.index),
        "matched_factor_correlations": matched,
        "mean_abs_corr": float(matched["abs_pearson"].mean()) if not matched.empty else np.nan,
        "median_abs_corr": float(matched["abs_pearson"].median()) if not matched.empty else np.nan,
        "optimal_order": optimal_order.tolist(),
        "matched_true_rows": matched_rows.tolist(),
    }
