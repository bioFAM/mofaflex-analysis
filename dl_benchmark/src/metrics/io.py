from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data.load_multiome import load_10x_multiome_views


def load_obs_vector(
    data_path: str,
    input_format: str,
    obs_key: str,
    obs_mod: str | None = "rna",
    meta_path: str | None = None,
    n_top_rna: int = 4000,
    n_top_atac: int = 10000,
) -> pd.Series:
    data_path = os.path.expanduser(data_path)
    if input_format == "h5ad":
        import anndata as ad

        adata = ad.read_h5ad(data_path)
        if obs_key not in adata.obs:
            raise KeyError(f"Missing obs key '{obs_key}' in adata.obs.")
        values = adata.obs[obs_key].copy()
        values.index = adata.obs_names.astype(str)
        return values

    if input_format == "h5mu":
        import mudata as md

        mdata = md.read_h5mu(data_path)
        if obs_mod:
            if obs_mod not in mdata.mod:
                raise KeyError(f"Missing modality '{obs_mod}' in mdata.mod.")
            adata = mdata.mod[obs_mod]
            if obs_key not in adata.obs:
                raise KeyError(f"Missing obs key '{obs_key}' in mdata.mod['{obs_mod}'].obs.")
            values = adata.obs[obs_key].copy()
            values.index = adata.obs_names.astype(str)
            return values
        if obs_key not in mdata.obs:
            raise KeyError(f"Missing obs key '{obs_key}' in mdata.obs.")
        values = mdata.obs[obs_key].copy()
        values.index = mdata.obs_names.astype(str)
        return values

    if input_format == "10x":
        if meta_path is None:
            meta_path = os.path.join(data_path, "meta_data.csv")
        meta_path = os.path.expanduser(meta_path)

        views = load_10x_multiome_views(
            data_path,
            n_top_rna=n_top_rna,
            n_top_atac=n_top_atac,
        )
        barcodes = views["rna"].obs_names

        meta = pd.read_csv(meta_path, index_col=0)
        meta.index = meta.index.astype(str)
        if obs_key not in meta.columns:
            raise KeyError(f"Missing obs key '{obs_key}' in {meta_path}.")
        missing = barcodes.difference(meta.index)
        if len(missing) > 0:
            raise KeyError(f"{len(missing)} barcodes missing in metadata file {meta_path}.")
        values = meta.loc[barcodes, obs_key].copy()
        values.index = values.index.astype(str)
        return values

    raise ValueError(f"Unsupported input format: {input_format}")


def load_labels(
    data_path: str,
    input_format: str,
    label_key: str,
    obs_mod: str | None = "rna",
    meta_path: str | None = None,
    n_top_rna: int = 4000,
    n_top_atac: int = 10000,
) -> pd.Series:
    return load_obs_vector(
        data_path=data_path,
        input_format=input_format,
        obs_key=label_key,
        obs_mod=obs_mod,
        meta_path=meta_path,
        n_top_rna=n_top_rna,
        n_top_atac=n_top_atac,
    )


def parse_run_spec(spec: str):
    if "=" in spec:
        name, out_dir = spec.split("=", 1)
        name = name.strip()
        out_dir = out_dir.strip()
        if not name:
            raise ValueError(f"Invalid run spec '{spec}': missing run name before '='.")
        if not out_dir:
            raise ValueError(f"Invalid run spec '{spec}': missing output directory after '='.")
        return name, os.path.expanduser(out_dir)

    out_dir = os.path.expanduser(spec.strip())
    if not out_dir:
        raise ValueError(f"Invalid run spec '{spec}'.")
    return os.path.basename(os.path.normpath(out_dir)), out_dir


def load_run_cell_ids(out_dir: str):
    txt_path = os.path.join(out_dir, "cell_ids.txt")
    npy_path = os.path.join(out_dir, "cell_ids.npy")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return np.array([line.strip() for line in f if line.strip()], dtype=str)
    if os.path.exists(npy_path):
        return np.load(npy_path).astype(str)
    return None


def align_latent_and_labels(latent: np.ndarray, labels: pd.Series, out_dir: str):
    run_cell_ids = load_run_cell_ids(out_dir)
    if run_cell_ids is None:
        if latent.shape[0] != len(labels):
            raise ValueError(
                f"Row mismatch: latent has {latent.shape[0]} rows but labels has {len(labels)}. "
                "Add cell_ids.txt or cell_ids.npy in run directory to align by cell IDs."
            )
        return latent, labels.to_numpy(), int(latent.shape[0])

    if latent.shape[0] != len(run_cell_ids):
        raise ValueError(
            f"Row mismatch: latent has {latent.shape[0]} rows but cell_ids has {len(run_cell_ids)}."
        )
    run_cell_ids_idx = pd.Index(run_cell_ids.astype(str))
    mask = run_cell_ids_idx.isin(labels.index)
    if not np.any(mask):
        raise ValueError("No overlapping cell IDs between run output and labels data.")
    matched_ids = run_cell_ids_idx[mask]
    latent_eval = latent[mask]
    labels_eval = labels.loc[matched_ids].to_numpy()
    return latent_eval, labels_eval, int(latent_eval.shape[0])
