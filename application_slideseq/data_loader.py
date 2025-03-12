import anndata as ad
import scanpy as sc


def load_data():
    data = ad.read_h5ad("data/slideseq.h5ad")
    data.layers["counts"] = data.X.copy()
    data.obsm["spatial"] = data.obs[["x", "y"]].values
    data.var_names = data.var_names.str.upper()
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, subset=True)

    data_dict = {}
    for group in data.obs["SampleID"].unique():
        data_dict[group] = {}
        data_dict[group]["rna"] = data[data.obs["SampleID"] == group]

    return data_dict