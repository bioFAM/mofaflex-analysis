import argparse
import os
import sys

from src.models.mojitoo import run_mojitoo
from src.utils.config import get_cfg, load_yaml_config


CLI_TOKENS = sys.argv[1:]


def _cli_provided(*flags: str) -> bool:
    for token in CLI_TOKENS:
        for flag in flags:
            if token == flag or token.startswith(flag + "="):
                return True
    return False


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
    if not _cli_provided("--rna-pcs"):
        args.rna_pcs = get_cfg(cfg, ("model", "rna_pcs"), args.rna_pcs)
    if not _cli_provided("--prot-pcs"):
        args.prot_pcs = get_cfg(cfg, ("model", "prot_pcs"), args.prot_pcs)
    if not _cli_provided("--corr-pval"):
        args.corr_pval = get_cfg(cfg, ("model", "corr_pval"), args.corr_pval)
    if not _cli_provided("--fdr-method"):
        args.fdr_method = get_cfg(cfg, ("model", "fdr_method"), args.fdr_method)
    if not _cli_provided("--is-reduction-center"):
        args.is_reduction_center = get_cfg(
            cfg, ("model", "is_reduction_center"), args.is_reduction_center
        )
    if not _cli_provided("--is-reduction-scale"):
        args.is_reduction_scale = get_cfg(
            cfg, ("model", "is_reduction_scale"), args.is_reduction_scale
        )
    if not _cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out:
        raise ValueError("Either provide --data and --out or set them in --config.")
    run_mojitoo(
        h5ad_path=os.path.expanduser(args.data),
        out_dir=os.path.expanduser(args.out),
        protein_obsm_key=args.protein_obsm_key,
        rna_pcs=int(args.rna_pcs),
        prot_pcs=int(args.prot_pcs),
        corr_pval=float(args.corr_pval),
        fdr_method=args.fdr_method,
        is_reduction_center=bool(args.is_reduction_center),
        is_reduction_scale=bool(args.is_reduction_scale),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--rna-pcs", type=int, default=20)
    p.add_argument("--prot-pcs", type=int, default=20)
    p.add_argument("--corr-pval", type=float, default=0.05)
    p.add_argument("--fdr-method", default="fdr_bh")
    p.add_argument("--is-reduction-center", action="store_true")
    p.add_argument("--is-reduction-scale", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
