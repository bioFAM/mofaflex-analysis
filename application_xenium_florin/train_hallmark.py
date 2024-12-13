import scanpy as sc
import argparse
from data_loader import load_xenium_breast_cancer
import numpy as np

from prismo import (
    PRISMO,
    DataOptions,
    ModelOptions,
    TrainingOptions,
    SmoothOptions,
    feature_sets
)

def to_upper(feature_set_collection):
    return feature_sets.FeatureSets(
        [
            feature_sets.FeatureSet([f.upper() for f in fs], fs.name)
            for fs in feature_set_collection
        ],
        name=feature_set_collection.name,
    )

def main():
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # load and preprocess data
    data = load_xenium_breast_cancer()
    sc.pp.highly_variable_genes(data["group_chromium"]["rna"], subset=True, n_top_genes=4000)
    data["group_chromium"]["rna"].var_names = (data["group_chromium"]["rna"].var["symbol"].astype(str))
    data["group_xenium"]["rna"].var_names = (data["group_xenium"]["rna"].var["symbol"].astype(str))

    # load annotations
    gene_set_collection = to_upper(feature_sets.FeatureSets.from_gmt(
        "../msigdb/h.all.v7.5.1.symbols.gmt", name="hallmark"
    ))

    # filter annotations by overlap with available genes
    gene_set_collection = gene_set_collection.filter(
        np.union1d(data["group_chromium"]["rna"].var_names, data["group_xenium"]["rna"].var_names),
        min_fraction=0.0,
        min_count=0,
        max_count=None,
    )

    # create gene set masks for each group
    data["group_chromium"]["rna"].varm["gene_set_mask"] = gene_set_collection.to_mask(
        data["group_chromium"]["rna"].var_names.tolist()
    ).T

    data["group_xenium"]["rna"].varm["gene_set_mask"] = gene_set_collection.to_mask(
        data["group_xenium"]["rna"].var_names.tolist()
    ).T

    # train model
    PRISMO(
        data,
        DataOptions(
            covariates_obsm_key={"group_xenium": "spatial"},
            plot_data_overview=False
        ),
        ModelOptions(
            n_factors=3,
            weight_prior="Horseshoe",
            factor_prior={"group_xenium": "GP", "group_chromium": "Normal"},
            likelihoods="Normal",
            annotations_varm_key={"rna": "gene_set_mask"},
            prior_penalty=0.001,
            nonnegative_weights=True,
            nonnegative_factors=True,
        ),
        TrainingOptions(
            device=args.device,
            batch_size=10000,
            max_epochs=500,
            lr=3e-3,
            early_stopper_patience=20,
            print_every=1,
            save_path=args.save_path,
            seed=92832,
        ),
        SmoothOptions(
            n_inducing=500,
            kernel="RBF",
        )
    )


if __name__ == "__main__":
    main()