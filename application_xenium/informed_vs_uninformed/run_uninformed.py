import os
import sys

parent_dir = os.path.abspath(os.path.join('..'))
sys.path.append(parent_dir)

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
    for seed in range(10):
        data = load_xenium_breast_cancer()
        sc.pp.highly_variable_genes(data["group_chromium"]["rna"], subset=True)
        data["group_chromium"]["rna"].var_names = (data["group_chromium"]["rna"].var["symbol"].astype(str))
        data["group_xenium"]["rna"].var_names = (data["group_xenium"]["rna"].var["symbol"].astype(str))

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
                n_factors=15,
                weight_prior={'rna': 'Horseshoe'},
                factor_prior={'group_chromium': 'Normal', 'group_xenium': 'GP'},
                likelihoods={'rna': 'Normal'},
                nonnegative_weights=True,
                nonnegative_factors=True,
                annotations=None,
                annotations_varm_key=None,
                prior_penalty=0.003,
                ),
            TrainingOptions(
                device="cuda:1",
                batch_size=10000,
                max_epochs=200,
                lr=0.003,
                early_stopper_patience=20,
                print_every=10,
                save_path=f"models/uninformed_{seed}.h5",
                seed=seed
                ),
            SmoothOptions(
                n_inducing=500,
                kernel='RBF',
                warp_groups=[],
            ),
        )

if __name__ == "__main__":
    main()