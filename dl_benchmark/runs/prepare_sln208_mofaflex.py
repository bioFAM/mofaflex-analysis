import argparse
import os

import anndata as ad
import fsspec
import mudata as md
import numpy as np
import pandas as pd


RELEVANT_CELL_TYPES = [
    "Activated CD4 T",
    "B1 B",
    "CD122+ CD8 T",
    "CD4 T",
    "CD8 T",
    "Erythrocytes",
    "GD T",
    "ICOS-high Tregs",
    "Ifit3-high B",
    "Ifit3-high CD4 T",
    "Ifit3-high CD8 T",
    "Ly6-high mono",
    "Ly6-low mono",
    "MZ B",
    "MZ/Marco-high macrophages",
    "Mature B",
    "Migratory DCs",
    "NK",
    "NKT",
    "Neutrophils",
    "Plasma B",
    "Red-pulp macrophages",
    "Transitional B",
    "Tregs",
    "cDC1s",
    "cDC2s",
    "pDCs",
]

CELL_TYPE_MAP = {
    "Mature B": "B",
    "Transitional B": "B",
    "Ifit3-high B": "B",
    "MZ B": "B",
    "B1 B": "B",
    "Plasma B": "B",
    "CD4 T": "CD4",
    "Ifit3-high CD4 T": "CD4",
    "Activated CD4 T": "CD4",
    "CD8 T": "CD8",
    "CD122+ CD8 T": "CD8",
    "Ifit3-high CD8 T": "CD8",
    "Tregs": "Tregs",
    "ICOS-high Tregs": "Tregs",
    "NKT": "NK",
    "NK": "NK",
    "Neutrophils": "Neutrophils",
    "Ly6-high mono": "Ly6",
    "Ly6-low mono": "Ly6",
    "cDC2s": "DC",
    "cDC1s": "DC",
    "Migratory DCs": "DC",
    "pDCs": "DC",
    "Erythrocytes": "Erythrocytes",
    "MZ/Marco-high macrophages": "Macrophages",
    "Red-pulp macrophages": "Macrophages",
}

SLN208_URL = "https://github.com/YosefLab/scVI-data/raw/master/sln_208.h5ad"


def _zipped_global_scale(x):
    x = np.log1p(x)
    x = x - np.nanmin(x, axis=0)
    global_std = np.nanstd(x)
    if global_std > 0:
        x = x / global_std
    return x.astype(np.float32, copy=False)


def _to_dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    elif hasattr(x, "todense"):
        x = np.asarray(x.todense())
    return np.asarray(x, dtype=np.float32)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with fsspec.open(
        args.url,
        client_kwargs={"trust_env": True},
    ) as f:
        adata_rna = ad.read_h5ad(f)

    adata_rna.var_names_make_unique()
    if args.out_raw_file:
        out_raw_h5ad = os.path.join(args.out_dir, args.out_raw_file)
        adata_rna.write_h5ad(out_raw_h5ad)
        print(out_raw_h5ad)

    adata_rna.obs_names = adata_rna.obs_names.str.upper()
    adata_rna.var_names = adata_rna.var_names.str.upper()

    if "highly_variable" in adata_rna.var:
        adata_rna._inplace_subset_var(adata_rna.var["highly_variable"])

    adata_rna._inplace_subset_obs(adata_rna.obs["cell_types"].isin(RELEVANT_CELL_TYPES))
    adata_rna.obs["cell_types"] = adata_rna.obs["cell_types"].cat.remove_unused_categories()
    adata_rna.obs["cell_types (high)"] = (
        adata_rna.obs["cell_types"].map(CELL_TYPE_MAP).astype("category")
    )

    if args.out_totalvi_file:
        adata_totalvi = adata_rna.copy()
        if "counts" not in adata_totalvi.layers:
            adata_totalvi.layers["counts"] = adata_totalvi.X.copy()
        out_totalvi_h5ad = os.path.join(args.out_dir, args.out_totalvi_file)
        adata_totalvi.write_h5ad(out_totalvi_h5ad)
        print(out_totalvi_h5ad)

    adata_mofa = adata_rna.copy()
    adata_mofa.X = _to_dense_float32(adata_mofa.X)

    df_prot = pd.DataFrame(
        adata_mofa.obsm["protein_expression"],
        index=adata_mofa.obs_names,
        columns=pd.Index(adata_mofa.uns["protein_names"]).str.upper(),
        dtype=np.float32,
    )
    df_prot = df_prot.dropna(axis=1, how="all")
    adata_prot = ad.AnnData(df_prot)

    mdata = md.MuData(
        {
            "rna": adata_mofa,
            "prot": adata_prot,
        }
    )

    for cadata in mdata.mod.values():
        cadata.X = _zipped_global_scale(np.asarray(cadata.X, dtype=np.float32))

    out_h5mu = os.path.join(args.out_dir, args.out_file)
    mdata.write_h5mu(out_h5mu)
    print(out_h5mu)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=SLN208_URL)
    p.add_argument("--out-dir", default="/home/aqoku/projects/data/mfl_bench")
    p.add_argument(
        "--out-raw-file",
        default="sln_208.h5ad",
        help="Optional .h5ad filename for the raw downloaded AnnData. Use empty string to disable.",
    )
    p.add_argument(
        "--out-totalvi-file",
        default="sln_208_totalvi.h5ad",
        help="Filtered .h5ad for totalVI/multigrate training. Use empty string to disable.",
    )
    p.add_argument("--out-file", default="sln_208_mofaflex.h5mu")
    args = p.parse_args()
    if args.out_raw_file is not None and args.out_raw_file.strip() == "":
        args.out_raw_file = None
    if args.out_totalvi_file is not None and args.out_totalvi_file.strip() == "":
        args.out_totalvi_file = None
    main(args)
