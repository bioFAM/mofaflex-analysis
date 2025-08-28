import mofaflex as mfl
from data_loader import load_cll

def main():
    data = load_cll()

    for seed in range(10):
        mfl.MOFAFLEX(
            data,
            mfl.DataOptions(
                plot_data_overview=False,
            ),
            mfl.ModelOptions(
                n_factors=10,
                weight_prior="Horseshoe",
                factor_prior="Normal",
                likelihoods="Normal",
            ),
            mfl.TrainingOptions(
                device="cuda:0",
                max_epochs=10000,
                lr=0.05,
                early_stopper_patience=1000,
                save_path=f"models/mofaflex_{seed}",
                seed=seed,
            )
        )

if __name__ == "__main__":
    main()