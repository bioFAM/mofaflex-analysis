import pandas as pd
import scanpy as sc
import os


def load_xenium_breast_cancer():
    module_dir = os.path.dirname(__file__)

    if os.path.isfile(os.path.join(module_dir, "data/xenium.h5ad")):
        xenium = sc.read_h5ad(os.path.join(module_dir, "data/xenium.h5ad"))
    else:
        xenium = sc.read_10x_h5(
            os.path.join(module_dir, "data/xenium/cell_feature_matrix.h5")
        )
        xenium_celltypes = pd.read_excel(
            os.path.join(module_dir, "data/celltypes.xlsx"),
            sheet_name="Xenium R1 Fig1-5 (supervised)",
        )
        xenium_celltypes.set_index("Barcode", inplace=True)
        xenium_celltypes.index.name = None
        xenium_celltypes.index = xenium_celltypes.index.astype(str)
        xenium_celltypes.columns = ["celltype"]
        xenium_metadata = pd.read_csv(os.path.join(module_dir, "data/xenium/cells.csv"))
        xenium_metadata["cell_id"] = xenium_metadata["cell_id"].astype(str)
        xenium_metadata.set_index("cell_id", inplace=True)
        xenium.obs = pd.merge(
            xenium_celltypes,
            xenium_metadata,
            left_index=True,
            right_index=True,
            how="left",
        )
        xenium.var["symbol"] = xenium.var.index.copy()
        xenium.var.index = xenium.var["gene_ids"]
        xenium.var.drop(columns=["gene_ids", "feature_types", "genome"], inplace=True)
        xenium.obsm["spatial"] = xenium.obs[["x_centroid", "y_centroid"]].values
        xenium.layers["counts"] = xenium.X.copy()
        sc.pp.normalize_total(xenium)
        sc.pp.log1p(xenium)
        xenium.write_h5ad(os.path.join(module_dir, "data/xenium.h5ad"))

    if os.path.isfile(os.path.join(module_dir, "data/chromium.h5ad")):
        chromium = sc.read_h5ad(os.path.join(module_dir, "data/chromium.h5ad"))
    else:
        chromium = sc.read_10x_h5(
            os.path.join(module_dir, "data/chromium/filtered_feature_bc_matrix.h5")
        )
        chromium_celltypes = pd.read_excel(
            os.path.join(module_dir, "data/celltypes.xlsx"), sheet_name="scFFPE-Seq"
        )
        chromium_celltypes.index = chromium_celltypes["Barcode"]
        chromium.obs["celltype"] = pd.merge(
            chromium.obs, chromium_celltypes, left_index=True, right_index=True
        )["Annotation"]
        chromium.var["symbol"] = chromium.var.index.copy()
        chromium.var.index = chromium.var["gene_ids"]
        chromium.var.drop(columns=["gene_ids", "feature_types", "genome"], inplace=True)
        chromium.layers["counts"] = chromium.X.copy()
        sc.pp.normalize_total(chromium)
        sc.pp.log1p(chromium)
        chromium.write_h5ad(os.path.join(module_dir, "data/chromium.h5ad"))

    data = {
        "group_xenium": {"rna": xenium},
        "group_chromium": {"rna": chromium},
    }

    return data
