import argparse
import os

from src.models.llm_embeddings import run_scfoundation_embedding
from src.utils.config import get_cfg, load_yaml_config


def _apply_config(args):
    if not args.config:
        return args
    cfg = load_yaml_config(args.config)
    args.data = args.data or get_cfg(cfg, ("dataset", "path"))
    args.out = args.out or get_cfg(cfg, ("output", "dir"))
    args.model_dir = args.model_dir or get_cfg(cfg, ("model", "model_dir"))
    args.task_name = get_cfg(cfg, ("model", "task_name"), args.task_name)
    args.pool_type = get_cfg(cfg, ("model", "pool_type"), args.pool_type)
    args.tgthighres = get_cfg(cfg, ("model", "tgthighres"), args.tgthighres)
    args.pre_normalized = get_cfg(cfg, ("model", "pre_normalized"), args.pre_normalized)
    args.version = get_cfg(cfg, ("model", "version"), args.version)
    args.batch_size = get_cfg(cfg, ("runtime", "batch_size"), args.batch_size)
    args.device = get_cfg(cfg, ("runtime", "device"), args.device)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out or not args.model_dir:
        raise ValueError("Require --data, --out, --model-dir (or config values).")
    run_scfoundation_embedding(
        h5ad_path=os.path.expanduser(args.data),
        out_dir=os.path.expanduser(args.out),
        model_dir=os.path.expanduser(args.model_dir),
        batch_size=int(args.batch_size),
        device=args.device,
        task_name=args.task_name,
        pool_type=args.pool_type,
        tgthighres=args.tgthighres,
        pre_normalized=args.pre_normalized,
        version=args.version,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--task-name", default=None)
    p.add_argument("--pool-type", default="all")
    p.add_argument("--tgthighres", default="t4")
    p.add_argument("--pre-normalized", default="F")
    p.add_argument("--version", default="ce")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    main(args)
