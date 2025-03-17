def main():
    from data_loader import load_nsf_slideseq
    import pandas as pd
    from prismo import PRISMO, DataOptions, ModelOptions, TrainingOptions, SmoothOptions

    adata = load_nsf_slideseq()
    data = {"group_1" : {"rna" : adata}}

    PRISMO(
        data,
        DataOptions(
            covariates_obsm_key="spatial",
            plot_data_overview=False
        ),
        ModelOptions(
            n_factors=6,
            weight_prior="Horseshoe",
            factor_prior="GP",
            likelihoods="GammaPoisson",
            nonnegative_weights=True,
            nonnegative_factors=True
        ),
        TrainingOptions(
            device="cuda:0",
            early_stopper_patience=100,
            batch_size=1000,
            max_epochs=2000,
            lr=0.01,
            save_path=f"models/prismo.h5",
            seed=1234
        ),
        SmoothOptions(
            n_inducing=2000,
            kernel="Matern"
        )
    )

if __name__ == "__main__":
    main()