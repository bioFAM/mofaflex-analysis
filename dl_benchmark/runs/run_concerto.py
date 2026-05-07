import argparse
import os
import sys

from src.models.concerto import run_concerto
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
    if not _cli_provided("--concerto-root"):
        args.concerto_root = args.concerto_root or get_cfg(cfg, ("model", "concerto_root"))
    if not _cli_provided("--protein-obsm-key"):
        args.protein_obsm_key = get_cfg(cfg, ("dataset", "protein_obsm_key"), args.protein_obsm_key)
    if not _cli_provided("--batch-key"):
        args.batch_key = get_cfg(cfg, ("dataset", "batch_key"), args.batch_key)
    if not _cli_provided("--train-batch-size"):
        args.train_batch_size = get_cfg(cfg, ("model", "train_batch_size"), args.train_batch_size)
    if not _cli_provided("--test-batch-size"):
        args.test_batch_size = get_cfg(cfg, ("model", "test_batch_size"), args.test_batch_size)
    if not _cli_provided("--epoch-pretrain"):
        args.epoch_pretrain = get_cfg(cfg, ("model", "epoch_pretrain"), args.epoch_pretrain)
    if not _cli_provided("--lr"):
        args.lr = get_cfg(cfg, ("model", "lr"), args.lr)
    if not _cli_provided("--drop-rate"):
        args.drop_rate = get_cfg(cfg, ("model", "drop_rate"), args.drop_rate)
    if not _cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out or not args.concerto_root:
        raise ValueError("Require --data, --out, and --concerto-root (or config values).")
    run_concerto(
        h5ad_path=os.path.expanduser(args.data),
        out_dir=os.path.expanduser(args.out),
        concerto_root=os.path.expanduser(args.concerto_root),
        protein_obsm_key=args.protein_obsm_key,
        batch_key=args.batch_key,
        train_batch_size=int(args.train_batch_size),
        test_batch_size=int(args.test_batch_size),
        epoch_pretrain=int(args.epoch_pretrain),
        lr=float(args.lr),
        drop_rate=float(args.drop_rate),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--concerto-root", default=None)
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--test-batch-size", type=int, default=128)
    p.add_argument("--epoch-pretrain", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--drop-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
