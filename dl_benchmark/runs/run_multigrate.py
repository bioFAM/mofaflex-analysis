import argparse
import os
import sys

from src.data.load_multiome import load_10x_multiome_views
from src.data.loaders import load_h5ad_rna_protein_views
from src.models.multigrate import run_multigrate
from src.utils.config import get_cfg, load_yaml_config


def main(args):
    in_path = os.path.expanduser(args.data)
    if args.input_format == "h5ad":
        views = load_h5ad_rna_protein_views(
            in_path,
            protein_obsm_key=args.protein_obsm_key,
        )
    else:
        views = load_10x_multiome_views(
            in_path,
            n_top_rna=args.n_top_rna,
            n_top_atac=args.n_top_atac,
        )
    run_multigrate(
        views=views,
        out_dir=args.out,
        latent_dim=args.latent_dim,
        max_epochs=args.max_epochs,
        seed=args.seed,
        second_view_key=args.view2_key,
        losses=(args.rna_loss, args.atac_loss),
        kl_coef=args.kl_coef,
        integ_coef=args.integ_coef,
        batch_key=args.batch_key,
        integrate_on=args.integrate_on,
        mmd=args.mmd,
    )


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

    if not cli_provided("--data"):
        args.data = args.data or get_cfg(cfg, ("dataset", "path"))
    if not cli_provided("--out"):
        args.out = args.out or get_cfg(cfg, ("output", "dir"))
    if not cli_provided("--input-format"):
        args.input_format = get_cfg(cfg, ("dataset", "input_format"), args.input_format)
    if not cli_provided("--protein-obsm-key"):
        args.protein_obsm_key = get_cfg(cfg, ("dataset", "protein_obsm_key"), args.protein_obsm_key)
    if not cli_provided("--latent-dim"):
        args.latent_dim = get_cfg(cfg, ("model", "latent_dim"), args.latent_dim)
    if not cli_provided("--max-epochs"):
        args.max_epochs = get_cfg(cfg, ("model", "max_epochs"), args.max_epochs)
    if not cli_provided("--view2-key"):
        args.view2_key = get_cfg(cfg, ("model", "view2_key"), args.view2_key)
    if not cli_provided("--n-top-rna"):
        args.n_top_rna = get_cfg(cfg, ("dataset", "n_top_rna"), args.n_top_rna)
    if not cli_provided("--n-top-atac"):
        args.n_top_atac = get_cfg(cfg, ("dataset", "n_top_atac"), args.n_top_atac)
    if not cli_provided("--rna-loss"):
        args.rna_loss = get_cfg(cfg, ("model", "rna_loss"), args.rna_loss)
    if not cli_provided("--atac-loss"):
        args.atac_loss = get_cfg(cfg, ("model", "atac_loss"), args.atac_loss)
    if not cli_provided("--kl-coef"):
        args.kl_coef = get_cfg(cfg, ("model", "kl_coef"), args.kl_coef)
    if not cli_provided("--integ-coef"):
        args.integ_coef = get_cfg(cfg, ("model", "integ_coef"), args.integ_coef)
    if not cli_provided("--batch-key"):
        args.batch_key = get_cfg(cfg, ("model", "batch_key"), args.batch_key)
    if not cli_provided("--integrate-on"):
        args.integrate_on = get_cfg(cfg, ("model", "integrate_on"), args.integrate_on)
    if not cli_provided("--mmd"):
        args.mmd = get_cfg(cfg, ("model", "mmd"), args.mmd)
    if not cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    return args


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Optional YAML config path.")
    p.add_argument("--data", default=None, help="Path to 10x multiome directory or .h5ad.")
    p.add_argument("--out", default=None, help="Output directory.")
    p.add_argument("--input-format", choices=["10x", "h5ad"], default="10x")
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--n-top-rna", type=int, default=4000)
    p.add_argument("--n-top-atac", type=int, default=10000)
    p.add_argument("--view2-key", default="atac")
    p.add_argument("--rna-loss", default="nb")
    p.add_argument("--atac-loss", default="mse")
    p.add_argument("--kl-coef", type=float, default=1e-1)
    p.add_argument("--integ-coef", type=float, default=3000.0)
    p.add_argument("--batch-key", default=None)
    p.add_argument("--integrate-on", default=None)
    p.add_argument("--mmd", default="marginal")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args = _apply_config(args)
    if not args.data or not args.out:
        raise ValueError("Either provide --data and --out or set them in --config.")
    main(args)
