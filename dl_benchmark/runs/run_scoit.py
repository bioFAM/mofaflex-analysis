import argparse
import os
import sys

from src.models.scoit import run_scoit
from src.utils.config import get_cfg, load_yaml_config


def _cli_provided(flag: str) -> bool:
    argv = sys.argv[1:]
    return flag in argv or any(arg.startswith(flag + "=") for arg in argv)


def _apply_config(args):
    if not args.config:
        return args
    cfg = load_yaml_config(args.config)
    if not _cli_provided("--data"):
        args.data = args.data or get_cfg(cfg, ("dataset", "path"))
    if not _cli_provided("--out"):
        args.out = args.out or get_cfg(cfg, ("output", "dir"))
    if not _cli_provided("--protein-obsm-key"):
        args.protein_obsm_key = get_cfg(cfg, ("dataset", "protein_obsm_key"), args.protein_obsm_key)
    if not _cli_provided("--k1"):
        args.k1 = get_cfg(cfg, ("model", "k1"), args.k1)
    if not _cli_provided("--k2"):
        args.k2 = get_cfg(cfg, ("model", "k2"), args.k2)
    if not _cli_provided("--k3"):
        args.k3 = get_cfg(cfg, ("model", "k3"), args.k3)
    if not (_cli_provided("--normalization") or _cli_provided("--no-normalization")):
        args.normalization = get_cfg(cfg, ("model", "normalization"), args.normalization)
    if not _cli_provided("--pre-impute"):
        args.pre_impute = get_cfg(cfg, ("model", "pre_impute"), args.pre_impute)
    if not _cli_provided("--opt"):
        args.opt = get_cfg(cfg, ("model", "opt"), args.opt)
    if not _cli_provided("--dist"):
        args.dist = get_cfg(cfg, ("model", "dist"), args.dist)
    if not _cli_provided("--lr"):
        args.lr = get_cfg(cfg, ("model", "lr"), args.lr)
    if not _cli_provided("--n-epochs"):
        args.n_epochs = get_cfg(cfg, ("model", "n_epochs"), args.n_epochs)
    if not _cli_provided("--lambda-c-regularizer"):
        args.lambda_c_regularizer = get_cfg(cfg, ("model", "lambda_c_regularizer"), args.lambda_c_regularizer)
    if not _cli_provided("--lambda-g-regularizer"):
        args.lambda_g_regularizer = get_cfg(cfg, ("model", "lambda_g_regularizer"), args.lambda_g_regularizer)
    if not _cli_provided("--lambda-o-regularizer"):
        args.lambda_o_regularizer = get_cfg(cfg, ("model", "lambda_o_regularizer"), args.lambda_o_regularizer)
    if not _cli_provided("--no-earlystopping"):
        args.earlystopping = get_cfg(cfg, ("model", "earlystopping"), args.earlystopping)
    if not _cli_provided("--device"):
        args.device = get_cfg(cfg, ("runtime", "device"), args.device)
    if not _cli_provided("--verbose"):
        args.verbose = get_cfg(cfg, ("runtime", "verbose"), args.verbose)
    if not _cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out:
        raise ValueError("Either provide --data and --out or set them in --config.")
    run_scoit(
        h5ad_path=os.path.expanduser(args.data),
        out_dir=os.path.expanduser(args.out),
        protein_obsm_key=args.protein_obsm_key,
        k1=int(args.k1),
        k2=int(args.k2),
        k3=int(args.k3),
        normalization=bool(args.normalization),
        pre_impute=bool(args.pre_impute),
        opt=args.opt,
        dist=args.dist,
        lr=float(args.lr),
        n_epochs=int(args.n_epochs),
        lambda_c_regularizer=float(args.lambda_c_regularizer),
        lambda_g_regularizer=float(args.lambda_g_regularizer),
        lambda_o_regularizer=args.lambda_o_regularizer,
        earlystopping=bool(args.earlystopping),
        device=args.device,
        verbose=bool(args.verbose),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--k1", type=int, default=30)
    p.add_argument("--k2", type=int, default=30)
    p.add_argument("--k3", type=int, default=30)
    p.add_argument("--normalization", dest="normalization", action="store_true", default=True)
    p.add_argument("--no-normalization", dest="normalization", action="store_false")
    p.add_argument("--pre-impute", action="store_true", default=False)
    p.add_argument("--opt", default="Adam")
    p.add_argument("--dist", default="gaussian")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--lambda-c-regularizer", type=float, default=0.0)
    p.add_argument("--lambda-g-regularizer", type=float, default=0.0)
    p.add_argument("--lambda-o-regularizer", nargs="*", type=float, default=None)
    p.add_argument("--no-earlystopping", dest="earlystopping", action="store_false")
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    main(args)
