import mofaflex as mfl
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from mofaflex._core.datasets import MofaFlexDataset
from mofaflex._core.preprocessing import MofaFlexPreprocessor
from sklearn.metrics import roc_auc_score, average_precision_score
from plotnine import *

from scipy.sparse import issparse

def get_r2(model, data, weights, factors, subsample=1000, second_level="pathway"):
    """
    Args:
        model -- MOFA-FLEX model
        data -- dictionary of groups - views
        weights -- numpy array of weights
        factors -- numpy array of factors
    """

    dataset = MofaFlexDataset(data, sample_names=model.sample_names, feature_names=model.feature_names)
    preprocessor = mfl.MOFAFLEX._make_preprocessor(model, dataset)

    def r2_wrapper(view, group_name, view_name):
        r2s = {}
        for pathway, idx in view.obs.groupby(second_level).indices.items():
            if subsample is not None and subsample > 0 and subsample < idx.shape[0]:
                sample_idx = np.random.choice(idx.shape[0], subsample, replace=False)
            else:
                sample_idx = slice(None)
            cdata = data.preprocessor(view.X[idx[sample_idx], :], slice(None), slice(None), group_name, view_name)[0]
            if issparse(cdata):
                cdata = cdata.toarray()

            dispersions = model._dispersions.mean.get(view_name)
            if dispersions is not None:
                dispersions = align_global_array_to_local(dispersions, group_name, view_name, align_to="features")  # noqa F821
            r2s[pathway] = model._model_opts.likelihoods[view_name].r2(
                view_name,
                y_true=cdata,
                factors=align_global_array_to_local(  # noqa F821
                    factors[group_name][idx[sample_idx]], group_name, view_name, align_to="samples", axis=0
                )[sample_idx, :],
                weights=align_global_array_to_local(  # noqa F821
                    weights[view_name], group_name, view_name, align_to="features", axis=1
                ),
                dispersions=dispersions,
                sample_means=align_global_array_to_local(  # noqa F821
                    data.preprocessor.sample_means[group_name][view_name],
                    group_name,
                    view_name,
                    align_to="samples",
                    axis=0,
                )[sample_idx],
            )
        return r2s

    df = dataset.apply(r2_wrapper)

    results_list = []

    for celltype in df.keys():
        for x in df[celltype]["RNA"].keys():
            results_list.append(pd.DataFrame(pd.Series(df[celltype]["RNA"][x][1], index=model.factor_names), columns=["R2"]))
            results_list[-1]["celltype"] = celltype
            results_list[-1][second_level] = x

    results = pd.concat(results_list).reset_index(names="factor")

    return results

def assign_markers_to_gene_sets(markers, annotations, threshold=0.2):
    """
    Assign each row of markers to a gene set using Spectra CPU-style labeling.

    This matches Spectra's CPU behavior conceptually:
    - each factor is labeled independently
    - the best-overlapping gene set is chosen per factor
    - labels are accepted only if overlap coefficient > threshold
    - multiple factors may receive the same gene-set label
    - gene sets are not forced to be used at most once

    markers: array-like of shape (n_factors, n_marker_genes)
    annotations: DataFrame of shape (n_genes, n_gene_sets) with boolean values
    threshold: minimum overlap coefficient to accept an assignment

    Returns a DataFrame with columns: factor, gene_set, overlap_coefficient
    """

    rows = []
    gene_sets = {
        gs_name: set(annotations.index[annotations[gs_name].to_numpy(dtype=bool)])
        for gs_name in annotations.columns
    }

    for i, factor_genes in enumerate(markers):
        factor_set = set(pd.Series(factor_genes).dropna().astype(str))
        best_gene_set = None
        best_score = -np.inf
        best_size = -1

        for gs_name, gs_genes in gene_sets.items():
            denom = min(len(factor_set), len(gs_genes))
            score = len(factor_set & gs_genes) / denom if denom > 0 else 0.0

            # Spectra resolves exact overlap ties by preferring the larger gene set.
            if score > best_score or (score == best_score and len(gs_genes) > best_size):
                best_gene_set = gs_name
                best_score = score
                best_size = len(gs_genes)

        rows.append(
            {
                "factor": f"Factor {i + 1}",
                "gene_set": best_gene_set if best_score > threshold else None,
                "overlap_coefficient": float(best_score),
            }
        )

    return pd.DataFrame(rows)

def compute_auroc_matrix(factors_df: pd.DataFrame, pathway_series: pd.Series, subsample_size=1000) -> pd.DataFrame:
    """
    Compute a factor × pathway AUROC matrix using one-vs-rest classification.

    For each pathway, samples are binarised (1 if assigned to that pathway, 0
    otherwise), and the AUROC is computed for every factor's scores as a
    discriminator — i.e. how well a factor's continuous values separate samples
    in that pathway from all others.

    Parameters
    ----------
    factors_df : pd.DataFrame
        Continuous factor scores; shape (n_samples, n_factors). Rows are
        samples, columns are factor identifiers.
    pathway_series : pd.Series
        Categorical pathway assignments; shape (n_samples,). Must be aligned
        with the index of `factors_df`.
    subsample_size : int or None
        If set, subsample this many indices prior to computing AUROCs —
        stratified by pathway membership so both positive and negative classes
        are represented. If None or >= n_samples, no subsampling is performed.

    Returns
    -------
    pd.DataFrame
        AUROC values; shape (n_factors, n_pathways), indexed by factor
        identifiers and pathway labels.
    """
    pathways = pathway_series.unique()
    auroc = pd.DataFrame(index=factors_df.columns, columns=pathways, dtype=float)

    n_samples = len(factors_df)
    rng = np.random.default_rng(seed=42)

    for pathway in pathways:
        labels = (pathway_series == pathway).astype(int).values

        if subsample_size is not None and subsample_size < n_samples:
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]

            # Allocate subsample proportionally, guaranteeing ≥1 of each class
            n_pos = max(1, round(subsample_size * len(pos_idx) / n_samples))
            n_neg = subsample_size - n_pos

            sampled_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
            sampled_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
            idx = np.concatenate([sampled_pos, sampled_neg])

            sub_labels = labels[idx]
            sub_factors = factors_df.iloc[idx]
        else:
            sub_labels = labels
            sub_factors = factors_df

        for factor in factors_df.columns:
            scores = sub_factors[factor].values
            auroc.loc[factor, pathway] = roc_auc_score(sub_labels, scores)

    return auroc

def plot_consensus_ranking(consensus_df: pd.DataFrame):
    """
    Visualise a consensus ranking produced by :func:`consensus_ranking`.

    Creates a heatmap of per-group ranks (colour) with factors ordered by
    consensus rank on the Y-axis and groups on the X-axis. A second panel
    shows the mean rank as a horizontal bar chart.

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Output of :func:`consensus_ranking`: factors as index, one column per
        group, plus ``mean_rank`` and ``consensus_rank`` columns.

    Returns
    -------
    ggplot
    """
    group_cols = [c for c in consensus_df.columns if c not in ("mean_rank", "consensus_rank")]

    factor_order = consensus_df.index.tolist()  # already sorted best-to-worst

    # --- panel 1: per-group rank heatmap ---
    heatmap_df = (
        consensus_df[group_cols]
        .reset_index(names="Factor")
        .melt(id_vars="Factor", var_name="Group", value_name="Rank")
    )
    heatmap_df["Factor"] = pd.Categorical(heatmap_df["Factor"], categories=factor_order[::-1])

    p_heat = (
        ggplot(heatmap_df, aes(x="Group", y="Factor", fill="Rank"))
        + geom_tile(color="white", size=0.3)
        + scale_fill_gradientn(
            colors=["#2166ac", "#f7f7f7", "#d6604d"],
            name="Rank\n(1 = best)",
        )
        + theme_bw()
        + theme(
            axis_text_x=element_text(rotation=45, hjust=1, size=9),
            axis_text_y=element_text(size=9),
            panel_grid=element_blank(),
            figure_size=(max(3, len(group_cols) * 0.6 + 1), max(4, len(factor_order) * 0.35 + 1)),
        )
        + labs(x="", y="Factor", title="Consensus Ranking of Factors by R²")
    )

    return p_heat

def consensus_ranking(series_dict: dict) -> pd.DataFrame:
    """
    Compute a consensus ranking of factors from a dict of R² Series.

    Each Series is ranked independently (rank 1 = highest value), then ranks
    are averaged across all Series to produce a consensus. Ties within a single
    Series are broken by average rank.

    Parameters
    ----------
    series_dict : dict[str, pd.DataFrame]
        Keys are group names (e.g. cell types), values are DataFrames with the
        same index (factor names) and a column ``"RNA"`` containing R² values.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by factor name with one column per group showing
        individual ranks, plus a ``mean_rank`` column and a ``consensus_rank``
        column (1 = top consensus factor), sorted by ``mean_rank``.
    """
    df = pd.DataFrame({k: v["RNA"] for k, v in series_dict.items()})
    ranked = df.rank(ascending=False, method="average")
    ranked["mean_rank"] = ranked.mean(axis=1)
    ranked["consensus_rank"] = ranked["mean_rank"].rank(method="min").astype(int)
    return ranked.sort_values("mean_rank")