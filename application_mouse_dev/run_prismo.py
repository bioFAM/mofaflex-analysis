import scanpy as sc
import anndata as ad
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
    # load and preprocess data
    data = ad.read_h5ad("data/E16.5_E1S1.MOSTA.h5ad")
    sc.pp.highly_variable_genes(data, subset=True)
    data.var_names = data.var_names.str.upper()

    # load annotations
    gene_set_collection = to_upper(feature_sets.FeatureSets.from_gmt(
        "../msigdb/mh.all.v2024.1.Mm.symbols.gmt", name="hallmark"
    ))

    # filter annotations by overlap with available genes
    gene_set_collection = gene_set_collection.filter(
        data.var_names,
        min_fraction=0.05,
        min_count=10,
        max_count=None,
    )

    # create gene set masks for each group
    data.varm["gene_set_mask"] = gene_set_collection.to_mask(
        data.var_names.tolist()
    ).T

    data = {"group_1" : {"rna" : data}}

    # train model
    PRISMO(
        data,
        DataOptions(
            group_by=None,
            scale_per_group=True,
            covariates_obs_key=None,
            covariates_obsm_key={"group_1" : "spatial"},
            use_obs='union',
            use_var='union',
            plot_data_overview=False
            ),
        ModelOptions(
            n_factors=5,
            weight_prior={"rna" : "Horseshoe"},
            factor_prior={"group_1" : "GP"},
            likelihoods={"rna" : "Normal"},
            nonnegative_weights=True,
            nonnegative_factors=True,
            annotations=None,
            annotations_varm_key={"rna" : "gene_set_mask"},
            prior_penalty=0.003,
            ),
        TrainingOptions(
            device="cuda:0",
            batch_size=10000,
            max_epochs=100,
            lr=0.005,
            early_stopper_patience=10,
            print_every=1,
            save_path="model.h5",
            seed=12345,
            ),
        SmoothOptions(
            n_inducing=500,
            kernel='RBF',
            warp_groups=[],
        ),
    )


if __name__ == "__main__":
    main()