import argparse
import os
import tempfile


def _ensure_numba_cache_dir():
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    cache_dir = os.path.join(tempfile.gettempdir(), "mofaflex_benchmark_numba_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = cache_dir


_ensure_numba_cache_dir()

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from src.metrics.io import align_latent_and_labels, load_labels, parse_run_spec


def main(args):
    labels = load_labels(
        data_path=args.data,
        input_format=args.input_format,
        label_key=args.label_key,
        obs_mod=args.obs_mod,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    for spec in args.run:
        run_name, run_dir = parse_run_spec(spec)
        latent_path = os.path.join(run_dir, "latent.npy")
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Missing latent file: {latent_path}")
        latent = np.load(latent_path)
        if latent.ndim != 2:
            raise ValueError(f"Expected 2D latent matrix, got shape {latent.shape}.")

        latent_eval, labels_eval, _ = align_latent_and_labels(latent, labels, run_dir)

        adata_latent = ad.AnnData(np.asarray(latent_eval, dtype=np.float32))
        adata_latent.obs[args.label_key] = pd.Categorical(labels_eval.astype(str))

        sc.pp.neighbors(adata_latent, n_neighbors=args.leiden_neighbors, use_rep="X")
        if args.leiden_resolution is None:
            sc.tl.leiden(adata_latent, key_added="leiden")
        else:
            sc.tl.leiden(adata_latent, resolution=float(args.leiden_resolution), key_added="leiden")
        sc.tl.umap(adata_latent, random_state=args.seed)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        sc.pl.umap(adata_latent, color="leiden", ax=axes[0], show=False, title=f"{run_name}: Leiden")
        sc.pl.umap(adata_latent, color=args.label_key, ax=axes[1], show=False, title=f"{run_name}: {args.label_key}")

        out_png = os.path.join(args.out_dir, f"{run_name}_umap.png")
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)
        print(out_png)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help="Run spec as 'name=output_dir' or output_dir.")
    p.add_argument("--data", required=True, help="Path to labels source (.h5ad or .h5mu).")
    p.add_argument("--input-format", choices=["h5ad", "h5mu"], default="h5ad")
    p.add_argument("--label-key", default="cell_types")
    p.add_argument("--obs-mod", default="rna", help="For h5mu: modality to read labels from. Empty for mdata.obs.")
    p.add_argument("--leiden-resolution", type=float, default=None)
    p.add_argument("--leiden-neighbors", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out-dir", default="outputs/plots")
    args = p.parse_args()

    if args.obs_mod == "":
        args.obs_mod = None

    main(args)
