import anndata as ad
import torch
import scanpy as sc
import pickle as pkl
import pandas as pd

from Spectra import Spectra_gpu as Spectra

adata = ad.read_h5ad(f"data/anndata.h5ad", backed="r")
adata = adata[adata.obs.pathway!="INS"].to_memory()
adata.obs_names_make_unique()

sc.pp.filter_genes(adata, min_cells=10)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, batch_key="Batch_info")

sc.pp.normalize_total(adata)
adata = adata[:, adata.var.highly_variable]
sc.pp.log1p(adata)

mask = pd.read_csv("hallmark_mask.csv", index_col=0)
mask = mask.loc[adata.var_names]
mask = mask.loc[:, mask.sum(axis=0) >= 5]

gene_set_collection = {}
for gene_set in mask.columns:
    gene_set_collection[gene_set] = list(adata[:, mask[gene_set]].var_names)
    print(gene_set, len(gene_set_collection[gene_set]))

torch.manual_seed(12345)

model = Spectra.est_spectra(
    adata=adata,
    gene_set_dictionary={"global": gene_set_collection, "ct": {}},
    L={"global": len(gene_set_collection), "ct": 0},
    use_highly_variable=False,
    cell_type_key="cell_type",
    num_epochs=1000
)

factors = model.return_factors()
cell_scores = model.return_cell_scores()
eta_diag = model.return_eta_diag()
eta = model.return_eta()
rho = model.return_rho()
kappa = model.return_kappa()
gene_scalings = model.return_gene_scalings()

dict = {
    "factors": factors,
    "cell_scores": cell_scores,
    "eta_diag": eta_diag,
    "eta": eta,
    "rho": rho,
    "kappa": kappa,
    "gene_scalings": gene_scalings,
    "gene_set_collection": gene_set_collection,
    "SPECTRA_markers": adata.uns["SPECTRA_markers"],
    "obs_names": adata.obs_names,
    "var_names": adata.var_names,
    }
pkl.dump(dict, open(f"models/spectra_complete.pkl", "wb"))