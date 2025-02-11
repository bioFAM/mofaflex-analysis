# requires the nsf-paper conda environment to be activated

def main():
    import pickle as pkl

    import numpy as np
    import spatial_factorization as sf
    from data_loader import load_nsf_slideseq
    from tensorflow.data import Dataset

    adata = load_nsf_slideseq()

    # prepare data for SpatialFactorization model
    data = {
        "X": adata.obsm["spatial"].copy().astype("float32"),
        "Y": adata.layers["counts"].toarray().astype("float32"),
        "sz" : np.ones((adata.n_obs, 1), dtype="float32"),
        "idx" : np.arange(adata.n_obs)
        }
    data_tf = Dataset.from_tensor_slices(data)
    data_tf = data_tf.batch(adata.n_obs)
    inducing_locations = sf.misc.kmeans_inducing_pts(data["X"], 3000)

    # setup and train SpatialFactorization model
    nsf_model = sf.SpatialFactorization(
        J=adata.n_vars,
        L=5,
        Z=inducing_locations,
        lik="poi",
        nonneg=True,
    )
    nsf_model.init_loadings(data["Y"])
    trainer = sf.ModelTrainer(nsf_model)
    trainer.train_model(data_tf, adata.n_obs, None)

    # obtain inferred latent variables
    z_nsf = np.exp(nsf_model.sample_latent_GP_funcs(data["X"], S=100).numpy().mean(axis=0).T)
    w_nsf = nsf_model.get_loadings()

    pkl.dump({"z" : z_nsf, "w" : w_nsf}, open("lvs/nsf.pkl", "wb"))

if __name__ == "__main__":
    main()