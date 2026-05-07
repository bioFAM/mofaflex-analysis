import argparse
import os

from src.models.llm_embeddings import run_scgpt_embedding
from src.utils.config import get_cfg, load_yaml_config


def _apply_config(args):
    if not args.config:
        return args
    cfg = load_yaml_config(args.config)
    args.data = args.data or get_cfg(cfg, ("dataset", "path"))
    args.out = args.out or get_cfg(cfg, ("output", "dir"))
    args.model_dir = args.model_dir or get_cfg(cfg, ("model", "model_dir"))
    args.gene_col = get_cfg(cfg, ("model", "gene_col"), args.gene_col)
    args.batch_size = get_cfg(cfg, ("runtime", "batch_size"), args.batch_size)
    args.device = get_cfg(cfg, ("runtime", "device"), args.device)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out or not args.model_dir:
        raise ValueError("Require --data, --out, --model-dir (or config values).")
    run_scgpt_embedding(
        h5ad_path=os.path.expanduser(args.data),
        out_dir=os.path.expanduser(args.out),
        model_dir=os.path.expanduser(args.model_dir),
        gene_col=args.gene_col,
        batch_size=int(args.batch_size),
        device=args.device,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--gene-col", default="index")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    main(args)
