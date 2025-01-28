import scanpy as sc
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
    # set up parameters that will be used in different models
    n_factors = [3, 2, 4, 5, 4, 3, 3, 5, 4, 3]
    seed = np.random.randint(0, 1e5, size=len(n_factors))
    prior_penalty = [0.002, 0.003, 0.004, 0.003, 0.003, 0.004, 0.005, 0.003, 0.004, 0.002]
    batch_size = [10000, 9000, 11000, 8000, 9000, 10000, 11000, 8000, 9000, 10000]
    lr = [0.003, 0.002, 0.004, 0.005, 0.003, 0.002, 0.004, 0.005, 0.003, 0.002]
    n_inducing = [500, 400, 400, 450, 500, 500, 400, 450, 500, 500]
    n_top_genes = [4000, 3800, 4500, 5000, 4500, 4000, 4500, 5000, 4500, 4000]
    min_fraction = [0.3, 0.25, 0.35, 0.2, 0.3, 0.25, 0.35, 0.2, 0.3, 0.25]
    min_count = [10, 8, 5, 4, 9, 7, 6, 3, 8, 6]

    for i in range(len(n_factors)):
        # load and preprocess data
        data = load_xenium_breast_cancer()
        sc.pp.highly_variable_genes(data["group_chromium"]["rna"], subset=True, n_top_genes=n_top_genes[i])
        data["group_chromium"]["rna"].var_names = (data["group_chromium"]["rna"].var["symbol"].astype(str))
        data["group_xenium"]["rna"].var_names = (data["group_xenium"]["rna"].var["symbol"].astype(str))

        # load annotations
        gene_set_collection = to_upper(feature_sets.FeatureSets.from_gmt(
            "../msigdb/h.all.v7.5.1.symbols.gmt", name="hallmark"
        ))

        # filter annotations by overlap with available genes
        gene_set_collection = gene_set_collection.filter(
            np.union1d(data["group_chromium"]["rna"].var_names, data["group_xenium"]["rna"].var_names),
            min_fraction=min_fraction[i],
            min_count=min_count[i],
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
                group_by=None,
                scale_per_group=True,
                covariates_obs_key=None,
                covariates_obsm_key={'group_xenium': 'spatial'},
                use_obs='union',
                use_var='union',
                plot_data_overview=False
                ),
            ModelOptions(
                n_factors=n_factors[i],
                weight_prior={'rna': 'Horseshoe'},
                factor_prior={'group_chromium': 'Normal', 'group_xenium': 'GP'},
                likelihoods={'rna': 'Normal'},
                nonnegative_weights=True,
                nonnegative_factors=True,
                annotations=None,
                annotations_varm_key={'rna': 'gene_set_mask'},
                prior_penalty=prior_penalty[i],
                ),
            TrainingOptions(
                device="cuda:0",
                batch_size=batch_size[i],
                max_epochs=200,
                lr=lr[i],
                early_stopper_patience=10,
                print_every=100,
                save_path=f"models/model_{i}.h5",
                seed=seed[i]
                ),
            SmoothOptions(
                n_inducing=n_inducing[i],
                kernel='RBF',
                warp_groups=[],
            ),
        )


if __name__ == "__main__":
    main()