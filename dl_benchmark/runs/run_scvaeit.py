import argparse
import os
import sys

from src.models.scvaeit import run_scvaeit
from src.utils.config import get_cfg, load_yaml_config


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
    if not cli_provided("--protein-obsm-key"):
        args.protein_obsm_key = get_cfg(cfg, ("dataset", "protein_obsm_key"), args.protein_obsm_key)
    if not cli_provided("--dim-block-enc"):
        args.dim_block_enc = get_cfg(cfg, ("model", "dim_block_enc"), args.dim_block_enc)
    if not cli_provided("--dim-block-dec"):
        args.dim_block_dec = get_cfg(cfg, ("model", "dim_block_dec"), args.dim_block_dec)
    if not cli_provided("--dim-block-embed"):
        args.dim_block_embed = get_cfg(cfg, ("model", "dim_block_embed"), args.dim_block_embed)
    if not cli_provided("--dimensions"):
        args.dimensions = get_cfg(cfg, ("model", "dimensions"), args.dimensions)
    if not cli_provided("--dim-latent"):
        args.dim_latent = get_cfg(cfg, ("model", "dim_latent"), args.dim_latent)
    if not cli_provided("--dist-block"):
        args.dist_block = get_cfg(cfg, ("model", "dist_block"), args.dist_block)
    if not cli_provided("--beta-unobs"):
        args.beta_unobs = get_cfg(cfg, ("model", "beta_unobs"), args.beta_unobs)
    if not cli_provided("--beta-modal"):
        args.beta_modal = get_cfg(cfg, ("model", "beta_modal"), args.beta_modal)
    if not cli_provided("--p-feat"):
        args.p_feat = get_cfg(cfg, ("model", "p_feat"), args.p_feat)
    if not cli_provided("--learning-rate"):
        args.learning_rate = get_cfg(cfg, ("model", "learning_rate"), args.learning_rate)
    if not cli_provided("--num-epoch"):
        args.num_epoch = get_cfg(cfg, ("model", "num_epoch"), args.num_epoch)
    if not cli_provided("--batch-size"):
        args.batch_size = get_cfg(cfg, ("model", "batch_size"), args.batch_size)
    if not cli_provided("--batch-size-inference"):
        args.batch_size_inference = get_cfg(
            cfg, ("model", "batch_size_inference"), args.batch_size_inference
        )
    if not cli_provided("--save-every-epoch"):
        args.save_every_epoch = get_cfg(cfg, ("model", "save_every_epoch"), args.save_every_epoch)
    if not cli_provided("--early-stopping-patience"):
        args.early_stopping_patience = get_cfg(
            cfg, ("model", "early_stopping_patience"), args.early_stopping_patience
        )
    if not cli_provided("--early-stopping-tolerance"):
        args.early_stopping_tolerance = get_cfg(
            cfg, ("model", "early_stopping_tolerance"), args.early_stopping_tolerance
        )
    if not (
        cli_provided("--no-early-stopping-relative")
        or cli_provided("--early-stopping-relative")
    ):
        args.early_stopping_relative = get_cfg(
            cfg, ("model", "early_stopping_relative"), args.early_stopping_relative
        )
    if not cli_provided("--training-mode-for-eval"):
        args.training_mode_for_eval = get_cfg(
            cfg, ("runtime", "training_mode_for_eval"), args.training_mode_for_eval
        )
    if not cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out:
        raise ValueError("Either provide --data and --out or set them in --config.")
    run_scvaeit(
        h5ad_path=os.path.expanduser(args.data),
        out_dir=os.path.expanduser(args.out),
        protein_obsm_key=args.protein_obsm_key,
        dim_block_enc=args.dim_block_enc,
        dim_block_dec=args.dim_block_dec,
        dim_block_embed=args.dim_block_embed,
        dimensions=args.dimensions,
        dim_latent=int(args.dim_latent),
        dist_block=args.dist_block,
        beta_unobs=float(args.beta_unobs),
        beta_modal=args.beta_modal,
        p_feat=float(args.p_feat),
        learning_rate=float(args.learning_rate),
        num_epoch=int(args.num_epoch),
        batch_size=int(args.batch_size),
        batch_size_inference=int(args.batch_size_inference),
        save_every_epoch=int(args.save_every_epoch),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_tolerance=float(args.early_stopping_tolerance),
        early_stopping_relative=bool(args.early_stopping_relative),
        training_mode_for_eval=bool(args.training_mode_for_eval),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--dim-block-enc", nargs="*", type=int, default=[128, 64])
    p.add_argument("--dim-block-dec", nargs="*", type=int, default=[128, 64])
    p.add_argument("--dim-block-embed", nargs="*", type=int, default=[128, 64])
    p.add_argument("--dimensions", nargs="*", type=int, default=[32])
    p.add_argument("--dim-latent", type=int, default=20)
    p.add_argument("--dist-block", nargs="*", default=["NB", "NB"])
    p.add_argument("--beta-unobs", type=float, default=0.9)
    p.add_argument("--beta-modal", nargs="*", type=float, default=[0.05, 0.95])
    p.add_argument("--p-feat", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-epoch", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--batch-size-inference", type=int, default=512)
    p.add_argument("--save-every-epoch", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-tolerance", type=float, default=1e-4)
    p.add_argument("--no-early-stopping-relative", dest="early_stopping_relative", action="store_false")
    p.add_argument("--training-mode-for-eval", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
