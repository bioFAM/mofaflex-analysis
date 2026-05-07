import argparse
import os

import mudata as md


def main(args):
    in_path = os.path.expanduser(args.data)
    out_path = os.path.expanduser(args.out)

    mdata = md.read_h5mu(in_path)
    if "rna" not in mdata.mod:
        raise KeyError("Expected 'rna' modality in input h5mu.")
    if args.batch_key not in mdata.mod["rna"].obs:
        raise KeyError(f"Missing batch key '{args.batch_key}' in mdata.mod['rna'].obs.")

    keep = mdata.mod["rna"].obs[args.batch_key].astype(str) == str(args.batch_value)
    if int(keep.sum()) == 0:
        avail = sorted(mdata.mod["rna"].obs[args.batch_key].astype(str).unique())
        raise ValueError(
            f"No cells found for {args.batch_key}={args.batch_value}. Available: {avail}"
        )

    m_sub = mdata[keep.to_numpy(), :].copy()
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m_sub.write_h5mu(out_path)
    print(f"Wrote {out_path} with n_obs={m_sub.n_obs}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/aqoku/projects/data/mfl_bench/sln_208_mofaflex.h5mu")
    p.add_argument(
        "--out",
        default="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1_mofaflex.h5mu",
    )
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--batch-value", default="SLN208-D1")
    args = p.parse_args()
    main(args)
