import argparse
import json
import os
import tempfile
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from src.metrics.io import load_run_cell_ids, parse_run_spec


def _ensure_numba_cache_dir():
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    cache_dir = os.path.join(tempfile.gettempdir(), "mofaflex_benchmark_numba_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = cache_dir


_ensure_numba_cache_dir()


def _load_base_adata(path: str, label_key: str, batch_key: str | None):
    adata = ad.read_h5ad(os.path.expanduser(path))
    if label_key not in adata.obs:
        raise KeyError(f"Missing label key '{label_key}' in adata.obs.")
    if batch_key is not None and batch_key not in adata.obs:
        raise KeyError(f"Missing batch key '{batch_key}' in adata.obs.")
    adata.obs_names = adata.obs_names.astype(str)
    return adata


def _configure_bio_metrics(args):
    from scib_metrics.benchmark import BioConservation

    return BioConservation(
        isolated_labels=not args.disable_isolated_labels,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=not args.bio_no_kmeans,
        silhouette_label=not args.disable_silhouette_label,
        clisi_knn=not args.disable_clisi,
    )


def _configure_batch_metrics(args) -> Optional[object]:
    from scib_metrics.benchmark import BatchCorrection

    if args.bio_only:
        return None
    return BatchCorrection()


def _load_latent_with_ids(run_dir: str):
    latent_path = os.path.join(run_dir, "latent.npy")
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Missing latent file: {latent_path}")
    latent = np.load(latent_path)
    if latent.ndim != 2:
        raise ValueError(f"Expected 2D latent matrix in {latent_path}, got {latent.shape}.")
    cell_ids = load_run_cell_ids(run_dir)
    if cell_ids is not None:
        cell_ids = pd.Index(np.asarray(cell_ids, dtype=str))
        if len(cell_ids) != latent.shape[0]:
            raise ValueError(
                f"Mismatch for {run_dir}: latent rows={latent.shape[0]}, cell_ids={len(cell_ids)}."
            )
    return latent, cell_ids


def main(args):
    try:
        from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation
    except Exception as exc:
        raise RuntimeError(
            "scib_metrics Benchmarker API not available. "
            "Install/upgrade scib-metrics (e.g. >=0.5.8)."
        ) from exc

    effective_batch_key = args.batch_key if args.batch_key else None
    adata = _load_base_adata(args.data, args.label_key, effective_batch_key)
    run_specs = [parse_run_spec(spec) for spec in args.run]

    loaded = []
    common_ids = pd.Index(adata.obs_names)
    for run_name, run_dir in run_specs:
        latent, ids = _load_latent_with_ids(run_dir)
        if ids is None:
            if latent.shape[0] != adata.n_obs:
                raise ValueError(
                    f"{run_name}: latent rows={latent.shape[0]} but data rows={adata.n_obs}. "
                    "Add cell_ids.txt/cell_ids.npy for robust alignment."
                )
            ids = pd.Index(adata.obs_names)
        common_ids = common_ids.intersection(ids)
        loaded.append((run_name, run_dir, latent, ids))

    if len(common_ids) == 0:
        raise ValueError("No overlapping cell IDs across data and runs.")

    adata_eval = adata[common_ids].copy()
    adata_eval.obs[args.label_key] = adata_eval.obs[args.label_key].astype(str)
    if effective_batch_key is None:
        effective_batch_key = "__scib_single_batch"
        adata_eval.obs[effective_batch_key] = "batch1"
    else:
        adata_eval.obs[effective_batch_key] = adata_eval.obs[effective_batch_key].astype(str)

    embedding_keys = []
    for run_name, _, latent, ids in loaded:
        loc = ids.get_indexer(common_ids)
        if np.any(loc < 0):
            raise ValueError(f"{run_name}: failed to align all common cell IDs.")
        adata_eval.obsm[run_name] = latent[loc]
        embedding_keys.append(run_name)

    bio_metrics = _configure_bio_metrics(args)
    batch_metrics = _configure_batch_metrics(args)

    bm = Benchmarker(
        adata_eval,
        batch_key=effective_batch_key,
        label_key=args.label_key,
        bio_conservation_metrics=bio_metrics,
        batch_correction_metrics=batch_metrics,
        embedding_obsm_keys=embedding_keys,
        n_jobs=int(args.n_jobs),
    )
    bm.benchmark()
    results = bm.get_results()

    out_csv = os.path.expanduser(args.out_csv)
    out_json = os.path.expanduser(args.out_json)
    if os.path.dirname(out_csv):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.dirname(out_json):
        os.makedirs(os.path.dirname(out_json), exist_ok=True)

    results.to_csv(out_csv)
    with open(out_json, "w") as f:
        json.dump(results.reset_index().to_dict(orient="records"), f, indent=2)

    print(results)
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to h5ad with labels/batches in .obs.")
    p.add_argument("--run", action="append", required=True, help="Run spec 'name=output_dir'.")
    p.add_argument("--label-key", default="cell_types")
    p.add_argument(
        "--batch-key",
        default="batch",
        help="Batch column in adata.obs. Pass empty string for single-batch / bio-only evaluation.",
    )
    p.add_argument(
        "--bio-only",
        action="store_true",
        help="Run only BioConservation metrics and skip batch-correction aggregates.",
    )
    p.add_argument(
        "--bio-no-kmeans",
        action="store_true",
        help="Disable KMeans NMI/ARI inside the BioConservation metric group.",
    )
    p.add_argument("--disable-isolated-labels", action="store_true")
    p.add_argument("--disable-silhouette-label", action="store_true")
    p.add_argument("--disable-clisi", action="store_true")
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--out-csv", default="outputs/benchmark/scib_benchmarker.csv")
    p.add_argument("--out-json", default="outputs/benchmark/scib_benchmarker.json")
    args = p.parse_args()
    if args.batch_key == "":
        args.batch_key = None
    main(args)
