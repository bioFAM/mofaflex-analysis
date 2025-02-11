# run with "prismo" kernel

def main():
    from prismo import (
        PRISMO,
        DataOptions,
        ModelOptions,
        TrainingOptions,
    )
    from data_loader import load_cll

    data = load_cll()

    for seed in range(10):
        prismo_model = PRISMO(
            data,
            DataOptions(
                plot_data_overview=False,
            ),
            ModelOptions(
                n_factors=10,
                weight_prior="Horseshoe",
                factor_prior="Normal",
                likelihoods="Normal",
            ),
            TrainingOptions(
                device="cuda:0",
                max_epochs=10000,
                lr=0.05,
                early_stopper_patience=1000,
                print_every=500,
                save_path=f"models_hs/prismo_{seed}",
                seed=seed,
            )
        )

if __name__ == "__main__":
    main()