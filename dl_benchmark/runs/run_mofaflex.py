import argparse
import os
import sys

import numpy as np

from src.data.load_multiome import load_10x_multiome_views, preprocess_views_for_mofaflex
from src.data.loaders import load_h5ad_rna_protein_views, load_h5mu
from src.models.mofaflex import run_mofaflex
from src.utils.config import get_cfg, load_yaml_config


def _parse_bool(text: str) -> bool:
    val = text.strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {text}")


def _maybe_add_mofaflex_src(mofaflex_src: str | None):
    if not mofaflex_src:
        return
    src_path = os.path.join(os.path.expanduser(mofaflex_src), "src")
    if not os.path.isdir(src_path):
        raise FileNotFoundError(f"Expected MOFAFLEX source dir at: {src_path}")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _cast_views_to_float32(data):
    # MOFAFLEX preprocessing applies in-place scaling; integer X causes dtype-casting errors.
    if isinstance(data, dict):
        for group in data.values():
            for adata in group.values():
                adata.X = adata.X.astype(np.float32)
    else:
        for adata in data.mod.values():
            adata.X = adata.X.astype(np.float32)


def _to_upper_feature_sets(mfl, feature_set_collection):
    return mfl.FeatureSets(
        [mfl.FeatureSet([f.upper() for f in fs], fs.name) for fs in feature_set_collection],
        name=feature_set_collection.name,
    )


def _add_sln208_gene_set_mask(
    mdata,
    mca_gmt_url: str,
    msigdb_category: str,
    msigdb_dbver: str,
    min_fraction: float,
    min_count: int,
    max_count: int,
    similarity_threshold: float,
    gene_set_source: str = "hallmark+mca",
):
    import fsspec
    import mofaflex as mfl
    import tempfile

    if "rna" not in mdata.mod:
        raise KeyError("Expected modality 'rna' in MuData for gene-set annotations.")

    rna = mdata.mod["rna"]
    var_names = rna.var_names

    with tempfile.NamedTemporaryFile("w", suffix=".gmt", delete=True) as tmp:
        with fsspec.open(mca_gmt_url, mode="rt", client_kwargs={"trust_env": True}) as f:
            tmp.write(f.read())
            tmp.flush()
        mca_collection = _to_upper_feature_sets(
            mfl, mfl.FeatureSets.from_gmt(tmp.name, name="mca")
        ).filter(
            var_names,
            min_fraction=min_fraction,
            min_count=min_count,
            max_count=max_count,
        )

    if gene_set_source == "mca":
        gene_set_collection = mca_collection
    else:
        hallmark_collection = _to_upper_feature_sets(
            mfl,
            mfl.tl.msigdb_get_features(category=msigdb_category, dbver=msigdb_dbver),
        ).filter(
            var_names,
            min_fraction=min_fraction,
            min_count=min_count,
            max_count=max_count,
        )
        gene_set_collection = (hallmark_collection | mca_collection).merge_similar(
            metric="jaccard",
            similarity_threshold=similarity_threshold,
            iteratively=True,
        )

    rna.varm["gene_set_mask"] = gene_set_collection.to_mask(var_names.tolist()).T
    return mdata


def _apply_config(args):
    if not args.config:
        return args
    cfg = load_yaml_config(args.config)

    cli_tokens = sys.argv[1:]

    def cli_provided(*flags: str) -> bool:
        for token in cli_tokens:
            for flag in flags:
                if token == flag or token.startswith(f"{flag}="):
                    return True
        return False

    args.data = args.data or get_cfg(cfg, ("dataset", "path"))
    args.out = args.out or get_cfg(cfg, ("output", "dir"))
    if not cli_provided("--input-format"):
        args.input_format = get_cfg(cfg, ("dataset", "input_format"), args.input_format)
    if not cli_provided("--n-top-rna"):
        args.n_top_rna = get_cfg(cfg, ("dataset", "n_top_rna"), args.n_top_rna)
    if not cli_provided("--n-top-atac"):
        args.n_top_atac = get_cfg(cfg, ("dataset", "n_top_atac"), args.n_top_atac)
    if not cli_provided("--preprocess"):
        args.preprocess = get_cfg(cfg, ("dataset", "preprocess"), args.preprocess)
    if not cli_provided("--protein-obsm-key"):
        args.protein_obsm_key = get_cfg(cfg, ("dataset", "protein_obsm_key"), args.protein_obsm_key)
    if not cli_provided("--enable-gene-set-mask"):
        args.enable_gene_set_mask = get_cfg(
            cfg, ("dataset", "enable_gene_set_mask"), args.enable_gene_set_mask
        )
    if not cli_provided("--mca-gmt-url"):
        args.mca_gmt_url = get_cfg(cfg, ("dataset", "mca_gmt_url"), args.mca_gmt_url)
    if not cli_provided("--msigdb-category"):
        args.msigdb_category = get_cfg(cfg, ("dataset", "msigdb_category"), args.msigdb_category)
    if not cli_provided("--msigdb-dbver"):
        args.msigdb_dbver = get_cfg(cfg, ("dataset", "msigdb_dbver"), args.msigdb_dbver)
    if not cli_provided("--gene-set-min-fraction"):
        args.gene_set_min_fraction = get_cfg(
            cfg, ("dataset", "gene_set_min_fraction"), args.gene_set_min_fraction
        )
    if not cli_provided("--gene-set-min-count"):
        args.gene_set_min_count = get_cfg(
            cfg, ("dataset", "gene_set_min_count"), args.gene_set_min_count
        )
    if not cli_provided("--gene-set-max-count"):
        args.gene_set_max_count = get_cfg(
            cfg, ("dataset", "gene_set_max_count"), args.gene_set_max_count
        )
    if not cli_provided("--gene-set-similarity-threshold"):
        args.gene_set_similarity_threshold = get_cfg(
            cfg, ("dataset", "gene_set_similarity_threshold"), args.gene_set_similarity_threshold
        )
    if not cli_provided("--gene-set-source"):
        args.gene_set_source = get_cfg(cfg, ("dataset", "gene_set_source"), args.gene_set_source)
    if not cli_provided("--n-factors"):
        args.n_factors = get_cfg(cfg, ("model", "n_factors"), args.n_factors)
    if not cli_provided("--weight-prior"):
        args.weight_prior = get_cfg(cfg, ("model", "weight_prior"), args.weight_prior)
    if not cli_provided("--rna-likelihood"):
        args.rna_likelihood = get_cfg(cfg, ("model", "rna_likelihood"), args.rna_likelihood)
    if not cli_provided("--view2-key"):
        args.view2_key = get_cfg(cfg, ("model", "view2_key"), args.view2_key)
    if not cli_provided("--view2-likelihood"):
        args.view2_likelihood = get_cfg(cfg, ("model", "view2_likelihood"), args.view2_likelihood)
    if not cli_provided("--nonnegative-weights"):
        args.nonnegative_weights = get_cfg(cfg, ("model", "nonnegative_weights"), args.nonnegative_weights)
    if not cli_provided("--nonnegative-factors"):
        args.nonnegative_factors = get_cfg(cfg, ("model", "nonnegative_factors"), args.nonnegative_factors)
    if not cli_provided("--init-factors"):
        args.init_factors = get_cfg(cfg, ("model", "init_factors"), args.init_factors)
    if not cli_provided("--init-scale"):
        args.init_scale = get_cfg(cfg, ("model", "init_scale"), args.init_scale)
    if not cli_provided("--annotations-varm-key"):
        args.annotations_varm_key = get_cfg(cfg, ("dataset", "annotations_varm_key"), args.annotations_varm_key)
    if not cli_provided("--lr"):
        args.lr = get_cfg(cfg, ("model", "lr"), args.lr)
    if not cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    if not cli_provided("--max-epochs"):
        args.max_epochs = get_cfg(cfg, ("model", "max_epochs"), args.max_epochs)
    if not cli_provided("--device"):
        args.device = get_cfg(cfg, ("model", "device"), args.device)
    if not cli_provided("--mofaflex-src"):
        args.mofaflex_src = get_cfg(cfg, ("model", "mofaflex_src"), args.mofaflex_src)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out:
        raise ValueError("Either provide --data and --out or set them in --config.")

    _maybe_add_mofaflex_src(args.mofaflex_src)
    in_path = os.path.expanduser(args.data)
    if args.input_format == "h5mu":
        data = load_h5mu(in_path)
    elif args.input_format == "h5ad":
        views = load_h5ad_rna_protein_views(
            in_path,
            protein_obsm_key=args.protein_obsm_key,
        )
        data = {"group_1": views}
    else:
        views = load_10x_multiome_views(
            in_path,
            n_top_rna=args.n_top_rna,
            n_top_atac=args.n_top_atac,
        )
        views = preprocess_views_for_mofaflex(views, mode=args.preprocess)
        data = {"group_1": views}
    if args.input_format in {"h5mu", "h5ad"}:
        _cast_views_to_float32(data)

    if args.enable_gene_set_mask:
        if not hasattr(data, "mod"):
            raise ValueError(
                "Gene-set annotations require --input-format h5mu (MuData with rna modality)."
            )
        data = _add_sln208_gene_set_mask(
            data,
            mca_gmt_url=args.mca_gmt_url,
            msigdb_category=args.msigdb_category,
            msigdb_dbver=args.msigdb_dbver,
            min_fraction=args.gene_set_min_fraction,
            min_count=args.gene_set_min_count,
            max_count=args.gene_set_max_count,
            similarity_threshold=args.gene_set_similarity_threshold,
            gene_set_source=args.gene_set_source,
        )
        if not args.annotations_varm_key:
            args.annotations_varm_key = "gene_set_mask"

    likelihoods = {"rna": args.rna_likelihood, args.view2_key: args.view2_likelihood}
    annotations = {"rna": args.annotations_varm_key} if args.annotations_varm_key else None

    run_mofaflex(
        data=data,
        out_dir=args.out,
        n_factors=args.n_factors,
        weight_prior=args.weight_prior,
        likelihoods=likelihoods,
        nonnegative_weights=args.nonnegative_weights,
        nonnegative_factors=args.nonnegative_factors,
        init_factors=args.init_factors,
        init_scale=args.init_scale,
        group_by=None,
        scale_per_group=False,
        annotations_varm_key=annotations,
        lr=args.lr,
        seed=args.seed,
        max_epochs=args.max_epochs,
        device=args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--data", default=None, help="Path to input data (.h5mu, .h5ad, or 10x multiome directory).")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument("--input-format", choices=["h5mu", "h5ad", "10x"], default="10x")
    parser.add_argument("--protein-obsm-key", default="protein_expression")
    parser.add_argument("--enable-gene-set-mask", type=_parse_bool, default=False)
    parser.add_argument(
        "--mca-gmt-url",
        default="https://github.com/bioFAM/mofaflex-analysis/raw/refs/heads/main/msigdb/mmc5_gene.gmt",
    )
    parser.add_argument("--msigdb-category", default="mh.all")
    parser.add_argument("--msigdb-dbver", default="2023.2.Mm")
    parser.add_argument("--gene-set-min-fraction", type=float, default=0.1)
    parser.add_argument("--gene-set-min-count", type=int, default=15)
    parser.add_argument("--gene-set-max-count", type=int, default=300)
    parser.add_argument("--gene-set-similarity-threshold", type=float, default=0.8)
    parser.add_argument("--gene-set-source", choices=["hallmark+mca", "mca"], default="hallmark+mca")
    parser.add_argument("--n-top-rna", type=int, default=4000)
    parser.add_argument("--n-top-atac", type=int, default=10000)
    parser.add_argument(
        "--preprocess",
        choices=["paper", "none"],
        default="paper",
        help="MOFAFLEX preprocessing mode for 10x inputs.",
    )
    parser.add_argument(
        "--mofaflex-src",
        default=None,
        help="Optional path to local MOFAFLEX repo root. Its src/ is added to PYTHONPATH.",
    )
    parser.add_argument("--n-factors", type=int, default=3)
    parser.add_argument("--weight-prior", default="Horseshoe")
    parser.add_argument("--rna-likelihood", default="Normal")
    parser.add_argument("--view2-key", default="atac", help="Name of second modality (e.g. atac, prot).")
    parser.add_argument("--view2-likelihood", default="Normal")
    parser.add_argument("--nonnegative-weights", type=_parse_bool, default=True)
    parser.add_argument("--nonnegative-factors", type=_parse_bool, default=True)
    parser.add_argument(
        "--init-factors",
        default="0.0",
        help="Float value or one of: random, orthogonal, pca, nmf.",
    )
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--annotations-varm-key", default="")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=2511021902)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        args.init_factors = float(args.init_factors)
    except ValueError:
        args.init_factors = args.init_factors

    main(args)
