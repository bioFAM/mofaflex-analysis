def main():
    from data_loader import load_nsf_slideseq
    from prismo import PRISMO, DataOptions, ModelOptions, TrainingOptions, SmoothOptions

    adata = load_nsf_slideseq()

    for seed in range(10):
        print(seed)
        PRISMO(
            adata,
            DataOptions(
                covariates_obsm_key="spatial",
                plot_data_overview=False
            ),
            ModelOptions(
                n_factors=10,
                weight_prior="Horseshoe",
                factor_prior="GP",
                likelihoods="GammaPoisson",
                nonnegative_weights=True,
                nonnegative_factors=True
            ),
            TrainingOptions(
                device="cuda:1",
                early_stopper_patience=50,
                batch_size=1000,
                max_epochs=1000,
                lr=5e-3,
                print_every=10,
                save_path=f"models/prismo_10_{seed}",
                seed=seed
            ),
            SmoothOptions(
                n_inducing=500,
                kernel="Matern"
            )
        )

if __name__ == "__main__":
    main()