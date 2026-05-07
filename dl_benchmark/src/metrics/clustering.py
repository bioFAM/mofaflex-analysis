import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


def _encode_labels(labels):
    labels = np.asarray(labels)
    mask = np.array([x is not None and str(x) != "nan" for x in labels], dtype=bool)
    filtered = labels[mask]
    uniques, encoded = np.unique(filtered.astype(str), return_inverse=True)
    return encoded, uniques, mask


def _cluster_leiden(z, y_true, seed=42, n_neighbors=15, resolution=None, resolution_grid=None):
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(np.asarray(z, dtype=np.float32))
    sc.pp.neighbors(adata, n_neighbors=int(n_neighbors), use_rep="X")

    if resolution is not None:
        sc.tl.leiden(adata, resolution=float(resolution), key_added="leiden")
        y_pred = adata.obs["leiden"].astype(str).to_numpy()
        return y_pred, float(resolution)

    if resolution_grid is None:
        # Keep Scanpy defaults unless user explicitly requests a sweep.
        sc.tl.leiden(adata, key_added="leiden")
        y_pred = adata.obs["leiden"].astype(str).to_numpy()
        return y_pred, None

    best_ari = -np.inf
    best_pred = None
    best_res = None
    for res in resolution_grid:
        sc.tl.leiden(adata, resolution=float(res), key_added="leiden_tmp")
        pred = adata.obs["leiden_tmp"].astype(str).to_numpy()
        ari = adjusted_rand_score(y_true, pred)
        if ari > best_ari:
            best_ari = ari
            best_pred = pred
            best_res = float(res)
    return best_pred, best_res


def evaluate_celltype_metrics(
    latent: np.ndarray,
    labels,
    n_clusters: int | None = None,
    seed: int = 42,
    clustering: str = "kmeans",
    leiden_resolution: float | None = None,
    leiden_resolution_grid: list[float] | None = None,
    leiden_neighbors: int = 15,
):
    y_true, unique_labels, keep = _encode_labels(labels)
    z = np.asarray(latent)[keep]

    if z.ndim != 2:
        raise ValueError(f"Expected 2D latent matrix, got shape {z.shape}.")
    if z.shape[0] != y_true.shape[0]:
        raise ValueError("Latent and labels length mismatch after filtering.")
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 cell types with non-missing labels.")

    if n_clusters is None:
        n_clusters = len(unique_labels)
    n_clusters = int(n_clusters)
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2.")

    if clustering == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        y_pred = kmeans.fit_predict(z)
        used_resolution = None
    elif clustering == "leiden":
        y_pred, used_resolution = _cluster_leiden(
            z=z,
            y_true=y_true,
            seed=seed,
            n_neighbors=leiden_neighbors,
            resolution=leiden_resolution,
            resolution_grid=leiden_resolution_grid,
        )
    else:
        raise ValueError(f"Unknown clustering mode: {clustering}")

    metrics = {
        "n_cells_evaluated": int(z.shape[0]),
        "n_cell_types": int(len(unique_labels)),
        "n_clusters": int(np.unique(y_pred).size),
        "clustering": clustering,
        "leiden_resolution": used_resolution,
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "silhouette_celltype": float(silhouette_score(z, y_true)),
        "silhouette_kmeans": float(silhouette_score(z, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
        "calinski_harabasz": float(calinski_harabasz_score(z, y_pred)),
        "davies_bouldin": float(davies_bouldin_score(z, y_pred)),
    }
    return metrics
