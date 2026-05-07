import argparse
import os

import anndata as ad


def main(args):
    in_path = os.path.expanduser(args.data)
    out_path = os.path.expanduser(args.out)
    adata = ad.read_h5ad(in_path)

    if args.batch_key not in adata.obs:
        raise KeyError(f"Missing batch key '{args.batch_key}' in adata.obs.")

    batch_vals = adata.obs[args.batch_key].astype(str)
    keep = batch_vals == str(args.batch_value)
    if int(keep.sum()) == 0:
        raise ValueError(
            f"No cells found for {args.batch_key}={args.batch_value}. "
            f"Available: {sorted(batch_vals.unique())}"
        )

    adata_sub = adata[keep].copy()
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    adata_sub.write_h5ad(out_path)
    print(f"Wrote {out_path} with shape {adata_sub.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default="/home/aqoku/projects/data/mfl_bench/sln_208_totalvi.h5ad",
    )
    p.add_argument(
        "--out",
        default="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1.h5ad",
    )
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--batch-value", default="SLN208-D1")
    args = p.parse_args()
    main(args)
