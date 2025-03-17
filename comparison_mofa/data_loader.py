import rdata
import pandas as pd
import anndata as ad
import numpy as np
import mudata as md

def load_cll():
    data = rdata.read_rda("data/CLL_data.RData")
    metadata = pd.read_table("data/sample_metadata.txt")

    mdata = {}
    for view_name, view_data in data["CLL_data"].items():
        view_data = view_data[:, np.isnan(view_data).sum(axis=0) < view_data.shape[0]].T
        mdata[view_name] = ad.AnnData(
            X=view_data.values,
            obs=pd.DataFrame(index=view_data.coords["dim_1"].to_index()),
            var=pd.DataFrame(index=view_data.coords["dim_0"].to_index()),
        )
    mdata = md.MuData(mdata)
    return mdata


# import pandas as pd
# from anndata import AnnData


# def load_cll():
#     """Load and return CLL dataset."""
#     modalities = {}

#     obs = pd.read_csv(filepath_or_buffer="./data/cll_metadata.csv", index_col="Sample")

#     for ome in ["drugs", "methylation", "mrna", "mutations"]:
#         modality = pd.read_csv(filepath_or_buffer=f"./data/cll_{ome}.csv", sep=",", index_col=0, encoding="utf_8").T
#         modalities[ome] = AnnData(X=modality)

#     # Replace with gene ID with gene name
#     gene_ids = pd.read_csv("./data/cll_gene_ids.csv", index_col=0)
#     cols = list(modalities["mrna"].var_names)

#     # Replace each value in cols with the corrsponding value in the gene_ids dataframe
#     cols = [gene_ids.loc[gene_ids["GENEID"] == gene, "SYMBOL"].item() for gene in cols]
#     modalities["mrna"].var_names = cols

#     # avoid duplicated names with the Mutations view
#     modalities["mutations"].var_names = [f"m_{x}" for x in modalities["mutations"].var_names]

#     # Replace drug names
#     # Create mapping from drug_id to name
#     drug_names = pd.read_csv("./data/drugs.txt", sep=",", index_col=0)
#     mapping = drug_names["name"].to_dict()

#     # Replace all substrings in drugs.columns as keys with the corresponding values in the mapping
#     cols = []
#     for k in modalities["drugs"].var_names:
#         for v in mapping.keys():
#             if v in k:
#                 cols.append(k.replace(v, mapping[v]))
#                 break

#     modalities["drugs"].var_names = cols

#     for view in modalities.keys():
#         modalities[view] = modalities[view][modalities[view].obs_names.argsort()]
#         modalities[view] = modalities[view][:, modalities[view].var_names.argsort()]

#     return modalities