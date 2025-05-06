import logging
import warnings
import anndata as ad
import numpy as np
import pandas as pd
import scarches as sca
import Spectra
import scanpy as sc
from sklearn.decomposition import NMF
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    precision_recall_fscore_support,
    root_mean_squared_error,
)
from Spectra import Spectra_gpu
from mofaflex import FeatureSets as fs
from mofaflex import MOFAFLEX, DataOptions, ModelOptions, TrainingOptions

logger = logging.getLogger(__name__)


def get_data(fpr, fnr, database=None, version=None, seed=None, rng=None):
    adata = ad.read_h5ad("data/kang_tutorial.h5ad").copy()
    adata.var_names = adata.var_names.str.upper()
    # adata._inplace_subset_var(adata.to_df().std() > 0.2)

    if "v1" in version:
        genes = int(version.split("-")[1])
        feature_names = (
            adata.to_df().var().sort_values(ascending=False).iloc[:genes].index
        )
        adata = adata[:, [x for x in adata.var_names if x in feature_names]].copy()
    elif "v2" in version:
        # Compute HVG
        genes = int(version.split("-")[1])
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=genes,
            batch_key="stim",
            flavor="seurat_v3",
        )
        adata = adata[:, adata.var["highly_variable"]].copy()
    elif "v3" in version:
        # Compute HVG
        genes = int(version.split("-")[1])
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=genes,
            flavor="seurat_v3",
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    if database == "RH":
        hallmark_collection = fs.from_gmt("../msigdb/h.all.v7.5.1.symbols.gmt").filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
        reactome_collection = fs.from_gmt(
            "../msigdb/c2.cp.reactome.v7.5.1.symbols.gmt"
        ).filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
        gene_set_collection = hallmark_collection | reactome_collection
        gene_set_collection = gene_set_collection.merge_similar(
            metric="jaccard",
            similarity_threshold=0.8,
            iteratively=True,
        )
    elif database == "R":
        gene_set_collection = fs.from_gmt(
            "../msigdb/c2.cp.reactome.v7.5.1.symbols.gmt"
        ).filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
    elif database == "H":
        gene_set_collection = fs.from_gmt("../msigdb/h.all.v7.5.1.symbols.gmt").filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )

    true_mask = gene_set_collection.to_mask(adata.var_names.tolist())
    terms = true_mask.index.tolist()

    # Modify the prior knowledge introducing noise
    true_mask_copy = true_mask.copy()
    true_mask = true_mask.values
    noisy_mask = get_rand_noisy_mask(rng, true_mask, fpr=fpr, fnr=fnr, seed=seed)

    return adata, true_mask, noisy_mask, terms, true_mask_copy


def get_data_synthetic_missing_input(
    noise_level,
    database=None,
    version=None,
    seed=None,
    rng=None,
    y_noise=0.01,
    n_samples=10000,
):
    adata = ad.read_h5ad("data/kang_tutorial.h5ad").copy()
    adata.var_names = adata.var_names.str.upper()

    if "v1" in version:
        genes = int(version.split("-")[1])
        feature_names = (
            adata.to_df().var().sort_values(ascending=False).iloc[:genes].index
        )
        adata = adata[:, [x for x in adata.var_names if x in feature_names]].copy()
    elif "v2" in version:
        # Compute HVG
        genes = int(version.split("-")[1])
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=genes,
            batch_key="stim",
            flavor="seurat_v3",
        )
        adata = adata[:, adata.var["highly_variable"]].copy()
    elif "v3" in version:
        # Compute HVG
        genes = int(version.split("-")[1])
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=genes,
            flavor="seurat_v3",
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    if database == "RH":
        hallmark_collection = fs.from_gmt("../msigdb/h.all.v7.5.1.symbols.gmt").filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
        reactome_collection = fs.from_gmt(
            "../msigdb/c2.cp.reactome.v7.5.1.symbols.gmt"
        ).filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
        gene_set_collection = hallmark_collection | reactome_collection
        gene_set_collection = gene_set_collection.merge_similar(
            metric="jaccard",
            similarity_threshold=0.8,
            iteratively=True,
        )
    elif database == "R":
        gene_set_collection = fs.from_gmt(
            "../msigdb/c2.cp.reactome.v7.5.1.symbols.gmt"
        ).filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
    elif database == "H":
        gene_set_collection = fs.from_gmt("../msigdb/h.all.v7.5.1.symbols.gmt").filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )

    true_mask = gene_set_collection.to_mask(adata.var_names.tolist())
    terms = true_mask.index.tolist()

    # Modify the prior knowledge introducing noise
    true_mask_copy = true_mask.copy()
    true_mask = true_mask.values
    noisy_mask = get_rand_noisy_mask(rng, true_mask, fpr=0, fnr=0, seed=seed)

    # Create synthetic z
    z = np.random.normal(size=(n_samples, true_mask.shape[0]))
    z = z - z.mean(axis=0)
    z = z / z.std(axis=0)

    # Create synthetic w
    w = noisy_mask

    # Create synthetic data
    y = z @ w
    # y = y - y.mean(axis=0)
    # y = y / y.std(axis=0)
    y = y.astype(np.float32)

    # Add some final noise onto y
    y += np.random.normal(0, y_noise, size=y.shape)

    adata_synthetic_true = ad.AnnData(y.copy())
    adata_synthetic_true.obs = pd.DataFrame(index=[str(x) for x in range(y.shape[0])])
    adata_synthetic_true.var = pd.DataFrame(index=[str(x) for x in range(y.shape[1])])
    adata_synthetic_true.obs_names = [str(x) for x in range(y.shape[0])]
    adata_synthetic_true.var_names = [str(x) for x in range(y.shape[1])]
    adata_synthetic_true.varm["I"] = noisy_mask.T
    adata_synthetic_true.uns["terms"] = terms

    # Create missing input
    missing_input = np.random.choice(
        [True, False], size=y.shape, p=[noise_level, 1 - noise_level]
    )
    # Replace with nan
    y[missing_input] = np.nan

    # Create synthetic adata
    adata_synthetic = ad.AnnData(y)
    adata_synthetic.obs = pd.DataFrame(index=[str(x) for x in range(y.shape[0])])
    adata_synthetic.var = pd.DataFrame(index=[str(x) for x in range(y.shape[1])])
    adata_synthetic.obs_names = [str(x) for x in range(y.shape[0])]
    adata_synthetic.var_names = [str(x) for x in range(y.shape[1])]
    adata_synthetic.varm["I"] = noisy_mask.T
    adata_synthetic.uns["terms"] = terms

    return adata_synthetic, adata_synthetic_true, true_mask, noisy_mask, terms, true_mask_copy

def get_data_synthetic(
    fpr,
    fnr,
    database=None,
    version=None,
    seed=None,
    rng=None,
    y_noise=0.01,
    n_samples=10000,
):
    adata = ad.read_h5ad("data/kang_tutorial.h5ad").copy()
    adata.var_names = adata.var_names.str.upper()

    if "v1" in version:
        genes = int(version.split("-")[1])
        feature_names = (
            adata.to_df().var().sort_values(ascending=False).iloc[:genes].index
        )
        adata = adata[:, [x for x in adata.var_names if x in feature_names]].copy()
    elif "v2" in version:
        # Compute HVG
        genes = int(version.split("-")[1])
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=genes,
            batch_key="stim",
            flavor="seurat_v3",
        )
        adata = adata[:, adata.var["highly_variable"]].copy()
    elif "v3" in version:
        # Compute HVG
        genes = int(version.split("-")[1])
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=genes,
            flavor="seurat_v3",
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    if database == "RH":
        hallmark_collection = fs.from_gmt("../msigdb/h.all.v7.5.1.symbols.gmt").filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
        reactome_collection = fs.from_gmt(
            "../msigdb/c2.cp.reactome.v7.5.1.symbols.gmt"
        ).filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
        gene_set_collection = hallmark_collection | reactome_collection
        gene_set_collection = gene_set_collection.merge_similar(
            metric="jaccard",
            similarity_threshold=0.8,
            iteratively=True,
        )
    elif database == "R":
        gene_set_collection = fs.from_gmt(
            "../msigdb/c2.cp.reactome.v7.5.1.symbols.gmt"
        ).filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )
    elif database == "H":
        gene_set_collection = fs.from_gmt("../msigdb/h.all.v7.5.1.symbols.gmt").filter(
            adata.var_names,
            min_fraction=0.4,
            min_count=40,
            max_count=200,
        )

    true_mask = gene_set_collection.to_mask(adata.var_names.tolist())
    terms = true_mask.index.tolist()

    # Modify the prior knowledge introducing noise
    true_mask_copy = true_mask.copy()
    true_mask = true_mask.values
    noisy_mask = get_rand_noisy_mask(rng, true_mask, fpr=fpr, fnr=fnr, seed=seed)

    # Create synthetic z
    z = np.random.normal(size=(n_samples, true_mask.shape[0]))
    z = z - z.mean(axis=0)
    z = z / z.std(axis=0)

    # Create synthetic w
    w = noisy_mask

    # Create synthetic data
    y = z @ w
    # y = y - y.mean(axis=0)
    # y = y / y.std(axis=0)
    y = y.astype(np.float32)

    # Add some final noise onto y
    y += np.random.normal(0, y_noise, size=y.shape)

    # Create synthetic adata
    adata_synthetic = ad.AnnData(y)
    adata_synthetic.obs = pd.DataFrame(index=[str(x) for x in range(y.shape[0])])
    adata_synthetic.var = pd.DataFrame(index=[str(x) for x in range(y.shape[1])])
    adata_synthetic.obs_names = [str(x) for x in range(y.shape[0])]
    adata_synthetic.var_names = [str(x) for x in range(y.shape[1])]
    adata_synthetic.varm["I"] = noisy_mask.T
    adata_synthetic.uns["terms"] = terms

    # import ipdb; ipdb.set_trace()

    return adata_synthetic, adata, true_mask, noisy_mask, terms, true_mask_copy


def preprocess(adata):
    x = adata.X
    x = x - x.min(axis=0)
    log_x = np.log1p(x)
    log_x = log_x / log_x.std()
    log_x_centered = log_x - log_x.mean(axis=0)
    # log_x_stdised = log_x_centered / log_x_centered.std()

    return {
        "expimap": log_x_centered.astype(np.float32),
        "expimap_nb": x.astype(np.float32),
        "expimap_hardmask": log_x_centered.astype(np.float32),
        "expimap_hardmask_nb": x.astype(np.float32),
        "spectra": log_x.astype(np.float32),
        "mofaflex": log_x_centered.astype(np.float32),
        "mofaflex_nmf": log_x.astype(np.float32),
    }

def preprocess_missing(adata):
    "Normalize data, but skip missing values"
    x = adata.X
    # import ipdb; ipdb.set_trace()
    x = x - np.nanmin(x, axis=0)
    log_x = np.log1p(x)
    log_x = log_x / np.nanstd(log_x, axis=0)
    log_x_centered = log_x - np.nanmean(log_x, axis=0)

    return {
        "expimap": log_x_centered.astype(np.float32),
        "expimap_nb": x.astype(np.float32),
        "expimap_hardmask": log_x_centered.astype(np.float32),
        "expimap_hardmask_nb": x.astype(np.float32),
        "spectra": log_x.astype(np.float32),
        "mofaflex": log_x_centered.astype(np.float32),
        "mofaflex_nmf": log_x.astype(np.float32),
    }


def get_rand_noisy_mask(rng, true_mask, fpr=0.2, fnr=0.2, seed=None):
    if seed:
        rng = np.random.default_rng(seed)
    noisy_mask = np.array(true_mask, copy=True)

    for k in range(true_mask.shape[0]):
        active_idx = noisy_mask[k, :].nonzero()[0]
        inactive_idx = (~noisy_mask[k, :]).nonzero()[0]
        fp = int(fpr * len(active_idx))
        fn = int(fnr * len(active_idx))
        fp_idx = rng.choice(inactive_idx, fp, replace=False)
        fn_idx = rng.choice(active_idx, fn, replace=False)
        noisy_mask[k, fp_idx] = True
        noisy_mask[k, fn_idx] = False

    return noisy_mask


def train_spectra(data, mask, terms=None, **kwargs):
    adata = ad.AnnData(data)
    annot = pd.DataFrame(mask, columns=adata.var_names)
    if terms is not None:
        annot.index = pd.Index(terms, name="terms")

    annot = {
        f"all_{idx!s}": annot.columns[annot.loc[idx, :]].tolist() for idx in annot.index
    }

    lam = kwargs.pop("lam", 0.01)
    delta = kwargs.pop("delta", 0.001)
    kappa = kwargs.pop("kappa", None)
    rho = kwargs.pop("rho", 0.001)
    num_epochs = kwargs.pop("num_epochs", 10000)

    return Spectra.est_spectra(
        adata=adata,
        gene_set_dictionary=annot,  # because we do not use the cell types
        # L=n_factors,
        # we will supply a regular dict
        # instead of the nested dict above
        use_highly_variable=False,
        cell_type_key=None,  # "cell_type_annotations"
        use_weights=True,
        lam=lam,
        delta=delta,
        kappa=kappa,
        rho=rho,
        use_cell_types=False,  # set to False to not use the cell type annotations
        n_top_vals=25,
        filter_sets=False,
        label_factors=False,
        clean_gs=False,
        min_gs_num=3,
        overlap_threshold=0.2,
        num_epochs=num_epochs,  # for demonstration purposes we will only run 2 epochs, we recommend 10,000 epochs
        **kwargs,
    )


def train_expimap(data, mask, seed=0, terms=None, **kwargs):
    adata = ad.AnnData(data)
    adata.obs["cond"] = "cond"
    adata.varm["I"] = mask.T
    if terms is not None:
        adata.uns["terms"] = terms
    else:
        adata.uns["terms"] = [f"factor_{k}" for k in range(mask.shape[0])]

    recon_loss = kwargs.pop("recon_loss", "mse")
    soft_mask = kwargs.pop("soft_mask", True)
    alpha = kwargs.pop("alpha", None)
    alpha_l1 = kwargs.pop("alpha_l1", None)
    n_epochs = kwargs.pop("n_epochs", 500)
    batch_size = kwargs.pop("batch_size", adata.shape[0])

    hidden_size_1 = kwargs.pop("hidden_size_1", 512)
    hidden_size_2 = kwargs.pop("hidden_size_2", 128)

    # create an instance of the model
    model = sca.models.EXPIMAP(
        adata=adata,
        condition_key="cond",
        hidden_layer_sizes=[hidden_size_1, hidden_size_2],
        recon_loss=recon_loss,
        soft_mask=soft_mask,
    )

    early_stopping_kwargs = {
        "early_stopping_metric": "val_unweighted_loss",  # val_unweighted_loss
        "threshold": 0,
        "patience": 50,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    model.train(
        n_epochs=n_epochs,
        # alpha_epoch_anneal=100,
        alpha=alpha,
        # # alpha_kl=0.5,
        alpha_l1=alpha_l1,
        # weight_decay=0.0,
        use_early_stopping=True,
        early_stopping_kwargs=early_stopping_kwargs,
        batch_size=batch_size,
        monitor_only_val=False,
        seed=seed,
        **kwargs,
    )

    return model


def train_mofaflex(
    data,
    mask,
    obs,
    var,
    obs_names=None,
    var_names=None,
    seed=None,
    terms=None,
    **kwargs,
):
    adata = ad.AnnData(data, obs=obs, var=var)
    if obs_names is not None:
        adata.obs_names = obs_names
    if var_names is not None:
        adata.var_names = var_names
    if terms is None:
        terms = [f"factor_{k}" for k in range(mask.shape[0])]
    adata.varm["I"] = pd.DataFrame(mask, index=terms, columns=adata.var_names).T

    # import ipdb; ipdb.set_trace()

    device = kwargs.pop("device", "cpu")
    prior_penalty = kwargs.pop("prior_penalty", 0.005)
    n_factors = kwargs.pop("n_factors", 3)
    likelihood = kwargs.pop("likelihood", "Normal")
    nmf = kwargs.pop("nmf", False)
    init_factors = kwargs.pop("init_factors", "random")
    init_scale = kwargs.pop("init_scale", 0.1)

    batch_size = kwargs.pop("batch_size", 0)
    max_epochs = kwargs.pop("max_epochs", 10000)
    n_particles = kwargs.pop("n_particles", 1)
    lr = kwargs.pop("lr", 0.003)
    save_path = kwargs.pop("save_path", None)
    remove_constant_features = kwargs.pop("remove_constant_features", False)

    early_stopper_patience = kwargs.pop("early_stopper_patience", 100)

    data_opts = DataOptions(
        group_by=None,
        scale_per_group=False,
        covariates_obs_key=None,
        covariates_obsm_key=None,
        use_obs=None,
        use_var=None,
        plot_data_overview=False,
        remove_constant_features=remove_constant_features,
        annotations_varm_key="I",
    )

    model_opts = ModelOptions(
        n_factors=n_factors,
        weight_prior="Horseshoe",
        factor_prior="Normal",
        likelihoods={"view_1": likelihood},
        nonnegative_weights=nmf,
        nonnegative_factors=nmf,
        prior_penalty=prior_penalty,
        init_factors=init_factors,
        init_scale=init_scale,
    )

    training_opts = TrainingOptions(
        device=device,
        batch_size=batch_size,
        max_epochs=max_epochs,
        n_particles=n_particles,
        lr=lr,
        early_stopper_patience=early_stopper_patience,
        save_path=save_path,
        seed=seed,
    )

    # import ipdb; ipdb.set_trace()

    return MOFAFLEX({"group_1": {"view_1": adata}}, data_opts, model_opts, training_opts)


def get_factor_loadings(model, with_dense=False):
    if type(model).__name__ == "NMF":
        return model.components_
    if type(model).__name__ == "MOFAFLEX":
        w_hat = model.get_weights("numpy")["view_1"]
        if not with_dense and model.n_dense_factors > 0:
            return w_hat[model.n_dense_factors :, :]
        return w_hat
    if type(model).__name__ == "MuVI":
        w_hat = model.get_factor_loadings(as_df=False)["view_1"]
        if model.n_dense_factors > 0:
            return w_hat[: -model.n_dense_factors, :]
        return w_hat
    if type(model).__name__ == "SPECTRA_Model":
        return model.return_factors()[:-1, :]
    if type(model).__name__ == "EXPIMAP":
        return model.model.decoder.L0.expr_L.weight.cpu().detach().numpy().T
    if type(model).__name__ == "dict":
        return model["w"].copy()

    raise ValueError(f"Unknown model type: {type(model)}")


def get_factor_scores(model, data, with_dense=False):
    if type(model).__name__ == "NMF":
        return model.transform(data)
    if type(model).__name__ == "MOFAFLEX":
        z_hat = model.get_factors("numpy")["group_1"]
        if not with_dense and model.n_dense_factors > 0:
            return z_hat[:, model.n_dense_factors :]
        return z_hat
    if type(model).__name__ == "MuVI":
        z_hat = model.get_factor_scores(as_df=False)
        if model.n_dense_factors > 0:
            return z_hat[:, : -model.n_dense_factors]
        return z_hat
    if type(model).__name__ == "SPECTRA_Model":
        return model.return_cell_scores()[:, :-1]
    if type(model).__name__ == "EXPIMAP":
        return model.get_latent()
    if type(model).__name__ == "dict":
        return model["z"].copy()

    raise ValueError(f"Unknown model type: {type(model)}")


def get_reconstructed(model, data):
    if type(model).__name__ == "NMF":
        return get_factor_scores(model, data) @ get_factor_loadings(model)
    if type(model).__name__ == "MOFAFLEX":
        return get_factor_scores(model, data, with_dense=True) @ get_factor_loadings(
            model, with_dense=True
        )
    if type(model).__name__ == "MuVI":
        return model.get_reconstructed(as_df=False)["view_1"]
    if type(model).__name__ == "SPECTRA_Model":
        return model.return_cell_scores() @ (
            model.return_factors() * model.return_gene_scalings()
        )
    if type(model).__name__ == "EXPIMAP":
        return model.get_y()
    if type(model).__name__ == "dict":
        return get_factor_scores(model, data) @ get_factor_loadings(model)

    raise ValueError(f"Unknown model type: {type(model)}")


def _r2(y_true, y_pred):
    ss_res = np.nansum(np.square(y_true - y_pred))
    ss_tot = np.nansum(np.square(y_true))
    return 1.0 - (ss_res / ss_tot)


def get_variance_explained(model, data, per_factor=False):
    z = get_factor_scores(model, data)
    w = get_factor_loadings(model)
    if not per_factor:
        return _r2(data, get_reconstructed(model, data))
    n_factors = z.shape[1]
    r2 = []
    for k in range(n_factors):
        y_pred_fac_k = np.outer(z[:, k], w[k, :])
        r2.append(_r2(data, y_pred_fac_k))
    return r2


def get_rmse(model, data, per_factor=False):
    if not per_factor:
        return root_mean_squared_error(data, get_reconstructed(model, data))
    z = get_factor_scores(model, data)
    w = get_factor_loadings(model)
    rmse = []
    for k in range(z.shape[1]):
        y_pred_fac_k = np.outer(z[:, k], w[k, :])
        rmse.append(root_mean_squared_error(data, y_pred_fac_k))
    return rmse


def get_top_factors(model, data, r2_thresh=0.95):
    r2 = get_variance_explained(model, data, per_factor=True)
    r2_argsorted = np.argsort(r2)[::-1]
    r2_sorted = np.sort(r2)[::-1]

    if r2_thresh < 1.0:
        r2_thresh = (np.cumsum(r2_sorted) / np.sum(r2_sorted) < r2_thresh).sum() + 1

    return r2_argsorted[:r2_thresh], r2[:r2_thresh]


def sort_and_subset(w_hat, true_mask, top=None):
    # descending order
    argsort_indices = np.argsort(-np.abs(w_hat), axis=1)

    sorted_w_hat = np.array(list(map(lambda x, y: y[x], argsort_indices, w_hat)))
    sorted_true_mask = np.array(
        list(map(lambda x, y: y[x], argsort_indices, true_mask))
    )

    if top is not None:
        argsort_indices = argsort_indices[:, :top]
        sorted_w_hat = sorted_w_hat[:, :top]
        sorted_true_mask = sorted_true_mask[:, :top]

    return argsort_indices, sorted_w_hat, sorted_true_mask


def get_binary_scores(
    true_mask, model, threshold=0.0, per_factor=False, top=None, verbose=False
):
    w_hat = get_factor_loadings(model)
    feature_idx, w_hat, true_mask = sort_and_subset(w_hat, true_mask, top)

    if threshold is not None:
        prec, rec, f1, supp = precision_recall_fscore_support(
            (true_mask).flatten(),
            (np.abs(w_hat) > threshold).flatten(),
            average="binary",
            zero_division=1,
        )
    else:
        prec = None
        rec = None
        f1 = None
        supp = None
        sorted_w_hat = np.sort(np.abs(w_hat).flatten())
        for threshold_idx in np.linspace(
            0, len(sorted_w_hat), num=100, endpoint=False, dtype=int
        ):
            threshold_ = sorted_w_hat[threshold_idx]
            prec_, rec_, f1_, supp_ = precision_recall_fscore_support(
                (true_mask).flatten(),
                (np.abs(w_hat) > threshold_).flatten(),
                average="binary",
                zero_division=1,
            )

            if f1 is None or f1_ > f1:
                threshold = threshold_
                prec, rec, f1, supp = prec_, rec_, f1_, supp_
        if verbose:
            print(f"best threshold: {threshold}")

    if not per_factor:
        return prec, rec, f1, threshold

    per_factor_prec = []
    per_factor_rec = []
    per_factor_f1 = []
    for k in range(w_hat.shape[0]):
        mask = true_mask[k, :]
        loadings_hat = np.abs(w_hat[k, :])
        order = np.argsort(loadings_hat)[::-1]
        prec, rec, f1, _ = precision_recall_fscore_support(
            mask[order],
            loadings_hat[order] > threshold,
            average="binary",
            zero_division=1,
        )
        per_factor_prec.append(prec)
        per_factor_rec.append(rec)
        per_factor_f1.append(f1)

    return per_factor_prec, per_factor_rec, per_factor_f1, threshold


def get_reconstruction_fraction(true_mask, noisy_mask, model, top=None):
    fi_1, w_hat, true_mask = sort_and_subset(get_factor_loadings(model), true_mask, top)
    fi_2, w_hat, noisy_mask = sort_and_subset(
        get_factor_loadings(model), noisy_mask, top
    )
    assert (fi_1 == fi_2).all()
    feature_idx = fi_1
    # return true_mask & ~noisy_mask
    return feature_idx, w_hat, true_mask, noisy_mask
    # return (np.abs(w_hat) > 1e-3).mean()


def get_average_precision(true_mask, model, per_factor=False, top=None):
    w_hat = get_factor_loadings(model)
    _, w_hat, true_mask = sort_and_subset(w_hat, true_mask, top)
    if not per_factor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return average_precision_score(
                (true_mask).flatten(),
                np.abs(w_hat).flatten(),
            )
    per_factor_aupr = []
    for k in range(w_hat.shape[0]):
        mask = true_mask[k, :]
        loadings_hat = np.abs(w_hat[k, :])
        order = np.argsort(loadings_hat)[::-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            per_factor_aupr.append(
                average_precision_score(mask[order], loadings_hat[order])
            )
    return per_factor_aupr


def plot_precision_recall(true_mask, model, top=None):
    w_hat = get_factor_loadings(model)
    _, w_hat, true_mask = sort_and_subset(w_hat, true_mask, top)
    return PrecisionRecallDisplay.from_predictions(
        (true_mask).flatten(), np.abs(w_hat).flatten(), plot_chance_level=True
    )
