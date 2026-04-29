import anndata as ad
import scanpy as sc
import mofaflex as mfl

adata = ad.read_h5ad(f"data/anndata.h5ad", backed="r")
adata = adata[adata.obs.pathway!="INS"].to_memory()
adata.obs_names_make_unique()

sc.pp.filter_genes(adata, min_cells=100)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, batch_key="Batch_info")

sc.pp.normalize_total(adata)
adata = adata[:, adata.var.highly_variable]
sc.pp.log1p(adata)

# obtain and filter prior knowledge genesets
hallmark_collection = mfl.tl.msigdb_get_features(category="h.all", dbver="7.5.1").filter(adata.var_names, min_count=5, min_fraction=0.0)
gene_set_collection = hallmark_collection.merge_similar(metric="jaccard")

# add prior knowledge to data
adata.varm["annotations"] = gene_set_collection.to_mask(adata.var_names.tolist()).T

data = {}
for celltype in adata.obs.cell_type.unique():
    data[celltype] = {"RNA": adata[adata.obs.cell_type==celltype]}

# run mofaflex
model = mfl.MOFAFLEX(
    data,
    mfl.ModelOptions(
        n_factors=3,
        weight_prior="Horseshoe",
        factor_prior="Normal",
        likelihoods="Normal",
        nonnegative_weights=True,
        nonnegative_factors=True,
        annotation_confidence=0.999,
        init_factors=0.0,
        init_scale=0.1,
    ),
    mfl.DataOptions(
        plot_data_overview=False,
        scale_per_group=True,
        annotations_varm_key="annotations",
    ),

    mfl.TrainingOptions(
        device="cuda:1",
        batch_size=10000,
        seed=1234,
        max_epochs=10000,
        early_stopper_patience=50,
        lr=5e-3,
        pin_memory=True,
        num_workers=4,
        save_path=f"models/mfl_complete.h5",
    ),
)