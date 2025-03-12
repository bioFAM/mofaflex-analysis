import scanpy as sc
import anndata as ad
import numpy as np
from data_loader import load_data

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
    # load and preprocess data
    data = load_data()

    # load annotations
    gene_set_collection = to_upper(feature_sets.FeatureSets.from_gmt(
        "../msigdb/mh.all.v2024.1.Mm.symbols.gmt", name="hallmark"
    ))

    # filter annotations by overlap with available genes
    gene_set_collection = gene_set_collection.filter(
        data[list(data.keys())[0]]["rna"].var_names,
        min_fraction=0.05,
        min_count=10,
        max_count=None,
    )

    # create gene set masks for each group
    for group in data.keys():
        data[group]["rna"].varm["gene_set_mask"] = gene_set_collection.to_mask(
            data[group]["rna"].var_names.tolist()
        ).T

    # train model
    PRISMO(
        data,
        DataOptions(
            group_by=None,
            scale_per_group=True,
            covariates_obs_key=None,
            covariates_obsm_key="spatial",
            use_obs='union',
            use_var='union',
            plot_data_overview=False
            ),
        ModelOptions(
            n_factors=5,
            weight_prior="Horseshoe",
            factor_prior="GP",
            likelihoods="Normal",
            nonnegative_weights=True,
            nonnegative_factors=True,
            annotations=None,
            annotations_varm_key={"rna" : "gene_set_mask"},
            prior_penalty=0.003,
            ),
        TrainingOptions(
            device="cuda:0",
            batch_size=1000,
            max_epochs=500,
            lr=0.005,
            early_stopper_patience=10,
            print_every=1,
            save_path="model.h5",
            seed=1234,
            ),
        SmoothOptions(
            n_inducing=500,
            kernel='RBF',
            warp_groups=[],
            mefisto_kernel=False,
        ),
    )

if __name__ == "__main__":
    main()