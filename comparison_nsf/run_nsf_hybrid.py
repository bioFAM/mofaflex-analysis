import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    import sys
    import dill as pkl
    from contextlib import suppress
    import tensorflow_probability as tfp

    tfk = tfp.math.psd_kernels

    sys.path.append("nsf-paper")

    from models import sfh
    from utils import training, misc
    from utils.preprocess import load_data
    from utils.postprocess import interpret_nsfh

    n_inducing = 2000
    n_factors = 20
    data_path = "data/sshippo_J2000.h5ad"

    D, fmeans = load_data(
        data_path,
        model="NSF",
        lik="poi",
        sz="scanpy",
        train_frac=1.
        )

    n_samples, n_features = D["raw"]["tr"]["Y"].shape
    with suppress(TypeError):
        ker = getattr(tfk, "MaternThreeHalves")

    inducing_points = misc.kmeans_inducing_pts(D["raw"]["tr"]["X"], n_inducing)

    fit = sfh.SpatialFactorizationHybrid(n_samples, n_features, n_factors, inducing_points, lik="poi",
                                         nonneg=True, psd_kernel=ker,
                                         isotropic=True, nugget=1e-5,
                                         length_scale=0.1, disp="default",
                                         feature_means=fmeans)

    tro = training.ModelTrainer(fit, lr=0.01, pickle_path="model_hybrid.pkl", max_to_keep=3)
    tro.train_model(D["raw"]["tf"][0], D["raw"]["tf"][1], Dval=None, S=3, verbose=True,
        num_epochs=10000, ckpt_freq=50, kernel_hp_update_freq=10,
        status_freq=10, span=100, tol=5e-5, pickle_freq=None,
        lr_reduce=0.5, maxtry=10)
    

    insf = interpret_nsfh(fit, D["raw"]["tr"]["X"], lda_mode=False)
    pkl.dump(insf, open("insfh.pkl", "wb"))
    insf_lda = interpret_nsfh(fit, D["raw"]["tr"]["X"], lda_mode=True)
    pkl.dump(insf_lda, open("insfh_lda.pkl", "wb"))

if __name__ == "__main__":
    main()