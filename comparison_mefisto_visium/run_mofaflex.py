from data_loader import load_mefisto_visium
import mofaflex as mfl
import pandas as pd


def main():
    adata = load_mefisto_visium()
    adata.X = adata.X.toarray()
    data = {"group_1" : {"rna" : adata}}

    for seed in range(10):
        mfl.MOFAFLEX(
            data,
            mfl.DataOptions(
                covariates_obsm_key="spatial",
                plot_data_overview=False
            ),
            mfl.ModelOptions(
                n_factors=4,
                weight_prior="Horseshoe",
                factor_prior="GP",
                likelihoods="Normal"
            ),
            mfl.TrainingOptions(
                device="cuda:1",
                early_stopper_patience=500,
                lr=5e-2,
                save_path=f"models/mofaflex_{seed}",
                seed=seed
            ),
            mfl.SmoothOptions(
                n_inducing=1000,
                kernel="RBF"
            )
        )

if __name__ == "__main__":
    main()