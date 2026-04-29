import anndata as ad
import scanpy as sc
import pickle as pkl
import pandas as pd
import numpy as np

from scarches.models import EXPIMAP

celltypes = ["A549", "BXPC3", "HAP1", "HT29", "K562", "MCF7"]

for celltype in celltypes:
    adata = ad.read_h5ad(f"data/anndata.h5ad", backed="r")
    adata = adata[(adata.obs.cell_type==celltype) & (adata.obs.pathway!="INS")].to_memory()
    adata.obs_names_make_unique()

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, batch_key="Batch_info")

    sc.pp.normalize_total(adata)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    adata.X = adata.X.astype(np.float32)

    adata.obs["ct"] = "ct"

    mask = pd.read_csv("hallmark_mask.csv", index_col=0)
    mask = mask.loc[adata.var_names]
    mask = mask.loc[:, mask.sum(axis=0) >= 5]
    adata.varm["I"] = mask
    adata.uns["terms"] = mask.columns.to_list()

    adata.obs["cond"] = "cond"

    # remove genes which are not in any gene set
    adata._inplace_subset_var(adata.varm["I"].sum(1) > 0)
    adata.varm["I"] = adata.varm["I"].values

    model = EXPIMAP(
        adata=adata,
        condition_key="cond",
        mask_key="I",
        recon_loss="mse",
        soft_mask=True,
        )

    model.train(
        n_epochs=10000,
        seed=12345,
    )

    latents = model.get_latent(mean=False, only_active=True)
    directions = model.latent_directions(adata=adata)
    dict = {
        "latents": latents,
        "directions": directions,
        "weights": model.model.decoder.L0.expr_L.weight.cpu().detach().numpy().T,
        "obs_names": adata.obs_names,
        "var_names": adata.var_names,
        "mask": adata.varm["I"],
        }

    pkl.dump(dict, open(f"models/expimap_{celltype}.pkl", "wb"))