import argparse
import os

from src.models.baselines import run_nmf_baseline, run_pca_baseline


def main(args):
    data = os.path.expanduser(args.data)
    out_root = os.path.expanduser(args.out_root)
    name_prefix = args.name_prefix

    run_pca_baseline(
        h5ad_path=data,
        out_dir=os.path.join(out_root, f"{name_prefix}_pca"),
        protein_obsm_key=args.protein_obsm_key,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    run_nmf_baseline(
        h5ad_path=data,
        out_dir=os.path.join(out_root, f"{name_prefix}_nmf"),
        protein_obsm_key=args.protein_obsm_key,
        latent_dim=args.latent_dim,
        seed=args.seed,
        max_iter=args.nmf_max_iter,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default="/home/aqoku/projects/data/mfl_bench/sln_208_totalvi.h5ad",
        help="Input h5ad for sln_208.",
    )
    p.add_argument("--out-root", default="outputs", help="Root output directory.")
    p.add_argument("--name-prefix", default="sln_208", help="Output name prefix.")
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nmf-max-iter", type=int, default=1000)
    args = p.parse_args()
    main(args)
