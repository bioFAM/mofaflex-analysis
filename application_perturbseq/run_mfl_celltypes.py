import anndata as ad
import scanpy as sc
import mofaflex as mfl

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

    # obtain and filter prior knowledge genesets
    hallmark_collection = mfl.tl.msigdb_get_features(category="h.all", dbver="7.5.1").filter(adata.var_names, min_count=5, min_fraction=0.0)
    gene_set_collection = hallmark_collection.merge_similar(metric="jaccard")
    
    # add prior knowledge to data
    adata.varm["annotations"] = gene_set_collection.to_mask(adata.var_names.tolist()).T

    # run mofaflex
    model = mfl.MOFAFLEX(
        {celltype: {"RNA": adata}},
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
            scale_per_group=False,
            annotations_varm_key="annotations",
        ),

        mfl.TrainingOptions(
            device="cuda:0",
            batch_size=10000,
            seed=12345,
            max_epochs=1000,
            early_stopper_patience=50,
            lr=5e-3,
            pin_memory=True,
            num_workers=16,
            save_path=f"models/mfl_{celltype}.h5",
        ),
    )