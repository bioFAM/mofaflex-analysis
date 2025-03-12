import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    import sys
    import dill as pkl
    from contextlib import suppress
    import tensorflow_probability as tfp

    tfk = tfp.math.psd_kernels

    sys.path.append("nsf-paper")

    from models import sf
    from utils import training, misc
    from utils.preprocess import load_data
    from utils.postprocess import interpret_nsf

    n_inducing = 2000
    n_factors = 10
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

    fit = sf.SpatialFactorization(J=n_features, L=n_factors, Z=inducing_points, lik="poi", psd_kernel=ker,
        nugget=1e-5, length_scale=0.1, disp="default",
        nonneg=True, isotropic=True,
        feature_means=fmeans)

    tro = training.ModelTrainer(fit, lr=0.01, pickle_path="model_sf_scanpy.pkl", max_to_keep=3)
    tro.train_model(D["raw"]["tf"][0], D["raw"]["tf"][1], Dval=None, S=3, verbose=True,
        num_epochs=10000, ckpt_freq=50, kernel_hp_update_freq=10,
        status_freq=10, span=100, tol=5e-5, pickle_freq=None,
        lr_reduce=0.5, maxtry=10)

    insf = interpret_nsf(fit, D["raw"]["tr"]["X"], lda_mode=False)
    insf_lda = interpret_nsf(fit, D["raw"]["tr"]["X"], lda_mode=True)

    lvs = {}
    lvs["factors"] = insf["factors"]
    lvs["factors_lda"] = insf_lda["factors"]
    lvs["loadings"] = insf["loadings"]
    lvs["loadings_lda"] = insf_lda["loadings"]
    lvs["inducing_points"] = inducing_points
    lvs["spatial"] = D["raw"]["tr"]["X"]

    pkl.dump(lvs, open("lvs_sf_scanpy.pkl", "wb"))
    
if __name__ == "__main__":
    main()