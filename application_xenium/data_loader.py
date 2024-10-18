import pandas as pd
import scanpy as sc


def load_xenium_breast_cancer():
    visium = sc.read_visium("/data/walter/prismo-analysis/application_xenium/data/visium")
    visium_celltypes = pd.read_excel("data/celltypes.xlsx", sheet_name="Visium")
    visium.obs = pd.merge(visium.obs, visium_celltypes, left_index=True, right_on="Barcode", how="left")
    visium.obs.index = visium.obs["Barcode"]
    visium.obs.drop(columns=["in_tissue", "Barcode"], inplace=True)
    visium.obs.columns = ["array_row", "array_col", "cluster", "celltype"]
    visium.var["symbol"] = visium.var.index.copy()
    visium.var.index = visium.var["gene_ids"]
    visium.var.drop(columns=["gene_ids", "feature_types", "genome"], inplace=True)

    xenium = sc.read_10x_h5("/data/walter/prismo-analysis/application_xenium/data/xenium/cell_feature_matrix.h5")
    xenium_celltypes = pd.read_excel("data/celltypes.xlsx", sheet_name="Xenium R1 Fig1-5 (supervised)")
    xenium_celltypes["Barcode"] = xenium_celltypes["Barcode"].astype(str)
    xenium.obs["celltype"] = pd.merge(xenium.obs, xenium_celltypes, left_index=True, right_on="Barcode")["Cluster"]
    xenium_metadata = pd.read_csv("data/xenium/cells.csv")
    xenium_metadata["cell_id"] = xenium_metadata["cell_id"].astype(str)
    xenium.obs.index = xenium.obs.index.astype(str)
    xenium.obs = pd.merge(xenium.obs, xenium_metadata, left_index=True, right_on="cell_id", how="left")
    xenium.var["symbol"] = xenium.var.index.copy()
    xenium.var.index = xenium.var["gene_ids"]
    xenium.var.drop(columns=["gene_ids", "feature_types", "genome"], inplace=True)
    xenium.obsm["spatial"] = xenium.obs[["x_centroid", "y_centroid"]].values

    chromium = sc.read_10x_h5(
        "/data/walter/prismo-analysis/application_xenium/data/chromium/filtered_feature_bc_matrix.h5"
    )
    chromium_celltypes = pd.read_excel("data/celltypes.xlsx", sheet_name="scFFPE-Seq")
    chromium_celltypes.index = chromium_celltypes["Barcode"]
    chromium.obs["celltype"] = pd.merge(chromium.obs, chromium_celltypes, left_index=True, right_index=True)[
        "Annotation"
    ]
    chromium.var["symbol"] = chromium.var.index.copy()
    chromium.var.index = chromium.var["gene_ids"]
    chromium.var.drop(columns=["gene_ids", "feature_types", "genome"], inplace=True)

    data = {"group_visium": {"rna": visium}, "group_xenium": {"rna": xenium}, "group_chromium": {"rna": chromium}}

    return data
