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
)
from prismo import feature_sets

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

    # train model
    PRISMO(
        data,
        DataOptions(
            covariates_obsm_key={"group_xenium": "spatial"},
            plot_data_overview=False
        ),
        ModelOptions(
            n_factors=20,
            weight_prior="Horseshoe",
            factor_prior={"group_xenium": "GP", "group_chromium": "Normal"},
            likelihoods="Normal",
            nonnegative_weights=True,
            nonnegative_factors=True,
        ),
        TrainingOptions(
            device=args.device,
            batch_size=10000,
            max_epochs=300,
            lr=3e-3,
            early_stopper_patience=20,
            print_every=1,
            save_path=args.save_path,
            seed=59283,
        ),
        SmoothOptions(
            n_inducing=500,
            kernel="RBF",
        )
    )


if __name__ == "__main__":
    main()