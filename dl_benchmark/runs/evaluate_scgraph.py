import argparse
import json
import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from src.metrics.io import load_run_cell_ids, parse_run_spec


def _load_base_adata(path: str, label_key: str, batch_key: str):
    adata = ad.read_h5ad(os.path.expanduser(path))
    if label_key not in adata.obs:
        raise KeyError(f"Missing label key '{label_key}' in adata.obs.")
    if batch_key not in adata.obs:
        raise KeyError(f"Missing batch key '{batch_key}' in adata.obs.")
    adata.obs_names = adata.obs_names.astype(str)
    return adata


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


def _save_results(results: pd.DataFrame, out_csv: str, out_json: str):
    out_csv = os.path.expanduser(out_csv)
    out_json = os.path.expanduser(out_json)
    if os.path.dirname(out_csv):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.dirname(out_json):
        os.makedirs(os.path.dirname(out_json), exist_ok=True)

    results.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2)


def _prepare_rna_matrix_for_scgraph(adata: ad.AnnData):
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if hasattr(adata.X, "data"):
        mask = np.isfinite(adata.X.data)
        if not np.all(mask):
            adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        adata.X = np.nan_to_num(np.asarray(adata.X), nan=0.0, posinf=0.0, neginf=0.0)
    return adata


def _latent_to_umap(latent: np.ndarray, random_state: int = 0) -> np.ndarray:
    adata_latent = ad.AnnData(X=np.asarray(latent, dtype=np.float32))
    sc.pp.neighbors(adata_latent, use_rep="X")
    sc.tl.umap(adata_latent, random_state=int(random_state))
    return np.asarray(adata_latent.obsm["X_umap"], dtype=np.float32)


def main(args):
    try:
        from scgraph import scGraph
    except Exception as exc:
        raise RuntimeError(
            "scgraph-eval is not installed. Install it in scvi_env first."
        ) from exc

    adata = _load_base_adata(args.data, args.label_key, args.batch_key)
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
    adata_eval.obs[args.batch_key] = adata_eval.obs[args.batch_key].astype(str)
    adata_eval = _prepare_rna_matrix_for_scgraph(adata_eval)

    embedding_keys = []
    for run_name, _, latent, ids in loaded:
        loc = ids.get_indexer(common_ids)
        if np.any(loc < 0):
            raise ValueError(f"{run_name}: failed to align all common cell IDs.")
        aligned_latent = latent[loc]
        if args.only_umap:
            obsm_key = f"{run_name}_umap"
            adata_eval.obsm[obsm_key] = _latent_to_umap(
                aligned_latent,
                random_state=args.umap_random_state,
            )
        else:
            obsm_key = run_name
            adata_eval.obsm[obsm_key] = aligned_latent
        embedding_keys.append(obsm_key)

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        adata_eval.write_h5ad(tmp_path)
        scgraph = scGraph(
            adata_path=tmp_path,
            batch_key=args.batch_key,
            label_key=args.label_key,
            only_umap=args.only_umap,
            trim_rate=float(args.trim_rate),
            thres_batch=int(args.thres_batch),
            thres_celltype=int(args.thres_celltype),
        )
        results = scgraph.main(_obsm_list=embedding_keys).reset_index()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    results = results.rename(columns={"index": "Embedding"})
    if args.only_umap:
        results["Embedding"] = results["Embedding"].str.removesuffix("_umap")
    _save_results(results, out_csv=args.out_csv, out_json=args.out_json)
    print(results)
    print(f"Saved CSV: {os.path.expanduser(args.out_csv)}")
    print(f"Saved JSON: {os.path.expanduser(args.out_json)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to h5ad with labels/batches in .obs.")
    p.add_argument("--run", action="append", required=True, help="Run spec 'name=output_dir'.")
    p.add_argument("--label-key", default="cell_types")
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--trim-rate", type=float, default=0.05)
    p.add_argument("--thres-batch", type=int, default=100)
    p.add_argument("--thres-celltype", type=int, default=10)
    p.add_argument(
        "--only-umap",
        action="store_true",
        default=True,
        help="Evaluate 2D UMAPs derived from each embedding instead of raw latent spaces.",
    )
    p.add_argument("--umap-random-state", type=int, default=0)
    p.add_argument("--out-csv", default="outputs/benchmark/scgraph.csv")
    p.add_argument("--out-json", default="outputs/benchmark/scgraph.json")
    args = p.parse_args()
    main(args)
