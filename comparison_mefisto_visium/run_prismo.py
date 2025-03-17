# run with "prismo" environment

def main():
    from data_loader import load_mefisto_visium
    from prismo import PRISMO, DataOptions, ModelOptions, TrainingOptions, SmoothOptions
    import pandas as pd

    adata = load_mefisto_visium()
    adata.X = adata.X.toarray()
    data = {"group_1" : {"rna" : adata}}

    for seed in range(10):
        prismo_model = PRISMO(
            data,
            DataOptions(
                covariates_obsm_key="spatial",
                plot_data_overview=False
            ),
            ModelOptions(
                n_factors=4,
                weight_prior="Horseshoe",
                factor_prior="GP",
                likelihoods="Normal"
            ),
            TrainingOptions(
                device="cuda:0",
                early_stopper_patience=500,
                lr=5e-2,
                save_path=f"models/prismo_hs_{seed}",
                seed=seed
            ),
            SmoothOptions(
                n_inducing=1000,
                kernel="RBF"
            )
        )

if __name__ == "__main__":
    main()