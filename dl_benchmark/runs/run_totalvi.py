import argparse
import os
import sys

from src.data.loaders import load_h5ad
from src.models.totalvi import run_totalvi
from src.utils.config import get_cfg, load_yaml_config


def _none_if_empty(value: str | None):
    if value is None:
        return None
    if value.strip() == "":
        return None
    return value


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
    if not cli_provided("--counts-layer"):
        args.counts_layer = get_cfg(cfg, ("dataset", "counts_layer"), args.counts_layer)
    if not cli_provided("--batch-key"):
        args.batch_key = get_cfg(cfg, ("dataset", "batch_key"), args.batch_key)
    if not cli_provided("--latent-dim"):
        args.latent_dim = get_cfg(cfg, ("model", "latent_dim"), args.latent_dim)
    if not cli_provided("--max-epochs"):
        args.max_epochs = get_cfg(cfg, ("model", "max_epochs"), args.max_epochs)
    if not cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out:
        raise ValueError("Either provide --data and --out or set them in --config.")

    adata = load_h5ad(
        path=os.path.expanduser(args.data),
        counts_layer=args.counts_layer,
    )
    run_totalvi(
        adata=adata,
        out_dir=args.out,
        protein_obsm_key=args.protein_obsm_key,
        counts_layer=args.counts_layer,
        batch_key=_none_if_empty(args.batch_key),
        latent_dim=args.latent_dim,
        max_epochs=args.max_epochs,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Optional YAML config path.")
    p.add_argument("--data", default=None, help="Path to input .h5ad file.")
    p.add_argument("--out", default=None, help="Output directory.")
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--counts-layer", default="counts")
    p.add_argument("--batch-key", default="batch_indices")
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
