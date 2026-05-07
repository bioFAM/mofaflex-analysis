import argparse
import os
import sys

from src.models.scmm import run_scmm
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
    if not _cli_provided("--script-dir"):
        args.script_dir = args.script_dir or get_cfg(cfg, ("model", "script_dir"))
    if not _cli_provided("--protein-obsm-key"):
        args.protein_obsm_key = get_cfg(cfg, ("dataset", "protein_obsm_key"), args.protein_obsm_key)
    if not _cli_provided("--counts-layer"):
        args.counts_layer = get_cfg(cfg, ("dataset", "counts_layer"), args.counts_layer)
    if not _cli_provided("--latent-dim"):
        args.latent_dim = get_cfg(cfg, ("model", "latent_dim"), args.latent_dim)
    if not _cli_provided("--epochs"):
        args.epochs = get_cfg(cfg, ("model", "epochs"), args.epochs)
    if not _cli_provided("--batch-size"):
        args.batch_size = get_cfg(cfg, ("model", "batch_size"), args.batch_size)
    if not _cli_provided("--lr"):
        args.lr = get_cfg(cfg, ("model", "lr"), args.lr)
    if not _cli_provided("--num-hidden-layers"):
        args.num_hidden_layers = get_cfg(cfg, ("model", "num_hidden_layers"), args.num_hidden_layers)
    if not _cli_provided("--r-hidden-dim"):
        args.r_hidden_dim = get_cfg(cfg, ("model", "r_hidden_dim"), args.r_hidden_dim)
    if not _cli_provided("--p-hidden-dim"):
        args.p_hidden_dim = get_cfg(cfg, ("model", "p_hidden_dim"), args.p_hidden_dim)
    if not _cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    if not (_cli_provided("--no-cuda") or _cli_provided("--cuda")):
        args.no_cuda = bool(get_cfg(cfg, ("runtime", "no_cuda"), args.no_cuda))
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out or not args.script_dir:
        raise ValueError("Require --data, --out, and --script-dir (or config values).")
    run_scmm(
        h5ad_path=os.path.expanduser(args.data),
        out_dir=os.path.expanduser(args.out),
        script_dir=os.path.expanduser(args.script_dir),
        protein_obsm_key=args.protein_obsm_key,
        counts_layer=args.counts_layer,
        latent_dim=int(args.latent_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        num_hidden_layers=int(args.num_hidden_layers),
        r_hidden_dim=int(args.r_hidden_dim),
        p_hidden_dim=int(args.p_hidden_dim),
        seed=int(args.seed),
        no_cuda=bool(args.no_cuda),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--script-dir", default=None)
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--counts-layer", default="counts")
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-hidden-layers", type=int, default=1)
    p.add_argument("--r-hidden-dim", type=int, default=100)
    p.add_argument("--p-hidden-dim", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--cuda", dest="no_cuda", action="store_false")
    args = p.parse_args()
    main(args)
