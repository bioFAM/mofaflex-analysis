import numpy as np


def _valid_mask(values):
    arr = np.asarray(values)
    return np.array([x is not None and str(x) != "nan" for x in arr], dtype=bool)


def evaluate_scib_metrics(
    latent: np.ndarray,
    labels,
    batch=None,
    seed: int = 42,
    n_neighbors: int = 15,
):
    import scib_metrics as sm
    from scib_metrics.nearest_neighbors import pynndescent

    z = np.asarray(latent)
    y = np.asarray(labels).astype(str)
    mask = _valid_mask(y)

    if batch is not None:
        b = np.asarray(batch).astype(str)
        mask &= _valid_mask(b)
    else:
        b = None

    z = z[mask]
    y = y[mask]
    if b is not None:
        b = b[mask]

    if z.ndim != 2 or z.shape[0] < 3:
        raise ValueError("scib-metrics needs a 2D latent matrix with at least 3 rows.")
    if np.unique(y).size < 2:
        raise ValueError("scib-metrics needs at least 2 unique labels.")

    metrics = {}
    metrics["scib_silhouette_label"] = float(sm.silhouette_label(X=z, labels=y))

    try:
        nmi_ari = sm.nmi_ari_cluster_labels_kmeans(
            X=z,
            labels=y,
            random_state=int(seed),
        )
    except TypeError:
        nmi_ari = sm.nmi_ari_cluster_labels_kmeans(X=z, labels=y)
    if isinstance(nmi_ari, dict):
        metrics["scib_nmi_kmeans"] = float(nmi_ari["nmi"])
        metrics["scib_ari_kmeans"] = float(nmi_ari["ari"])
    else:
        metrics["scib_nmi_kmeans"] = float(nmi_ari.nmi)
        metrics["scib_ari_kmeans"] = float(nmi_ari.ari)

    if b is None or np.unique(b).size < 2:
        return metrics

    metrics["scib_silhouette_batch"] = float(
        sm.silhouette_batch(X=z, labels=y, batch=b)
    )

    neighbors = pynndescent(
        X=z,
        n_neighbors=int(n_neighbors),
        random_state=int(seed),
        n_jobs=1,
    )
    metrics["scib_ilisi_knn"] = float(sm.ilisi_knn(X=neighbors, batches=b))
    metrics["scib_clisi_knn"] = float(sm.clisi_knn(X=neighbors, labels=y))
    metrics["scib_graph_connectivity"] = float(
        sm.graph_connectivity(X=neighbors, labels=y)
    )
    return metrics
