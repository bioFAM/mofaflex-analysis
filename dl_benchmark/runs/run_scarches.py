import argparse
import os
import sys

from src.models.scarches import run_scarches_totalvi_surgery
from src.utils.config import get_cfg, load_yaml_config


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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
    if not cli_provided("--reference-batch-values"):
        args.reference_batch_values = _as_list(
            get_cfg(cfg, ("dataset", "reference_batch_values"), args.reference_batch_values)
        )
    if not cli_provided("--query-batch-values"):
        args.query_batch_values = _as_list(
            get_cfg(cfg, ("dataset", "query_batch_values"), args.query_batch_values)
        )
    if not cli_provided("--latent-dim"):
        args.latent_dim = get_cfg(cfg, ("model", "latent_dim"), args.latent_dim)
    if not cli_provided("--reference-max-epochs"):
        args.reference_max_epochs = get_cfg(
            cfg, ("model", "reference_max_epochs"), args.reference_max_epochs
        )
    if not cli_provided("--surgery-max-epochs"):
        args.surgery_max_epochs = get_cfg(
            cfg, ("model", "surgery_max_epochs"), args.surgery_max_epochs
        )
    if not cli_provided("--freeze-expression"):
        args.freeze_expression = get_cfg(
            cfg, ("model", "freeze_expression"), args.freeze_expression
        )
    if not cli_provided("--plan-kwargs"):
        args.plan_kwargs = get_cfg(cfg, ("model", "plan_kwargs"), args.plan_kwargs)
    if not cli_provided("--save-reference-anndata"):
        args.save_reference_anndata = get_cfg(
            cfg, ("model", "save_reference_anndata"), args.save_reference_anndata
        )
    if not cli_provided("--seed"):
        args.seed = get_cfg(cfg, ("model", "seed"), args.seed)
    return args


def main(args):
    args = _apply_config(args)
    if not args.data or not args.out:
        raise ValueError("Either provide --data and --out or set them in --config.")
    if not args.reference_batch_values or not args.query_batch_values:
        raise ValueError("Both reference and query batch values are required.")

    run_scarches_totalvi_surgery(
        data_path=os.path.expanduser(args.data),
        out_dir=args.out,
        protein_obsm_key=args.protein_obsm_key,
        counts_layer=args.counts_layer,
        batch_key=args.batch_key,
        reference_batch_values=args.reference_batch_values,
        query_batch_values=args.query_batch_values,
        latent_dim=args.latent_dim,
        reference_max_epochs=args.reference_max_epochs,
        surgery_max_epochs=args.surgery_max_epochs,
        freeze_expression=bool(args.freeze_expression),
        plan_kwargs=args.plan_kwargs,
        save_reference_anndata=bool(args.save_reference_anndata),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Optional YAML config path.")
    p.add_argument("--data", default=None, help="Path to input .h5ad file.")
    p.add_argument("--out", default=None, help="Output directory.")
    p.add_argument("--protein-obsm-key", default="protein_expression")
    p.add_argument("--counts-layer", default="counts")
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--reference-batch-values", nargs="+", default=None)
    p.add_argument("--query-batch-values", nargs="+", default=None)
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--reference-max-epochs", type=int, default=200)
    p.add_argument("--surgery-max-epochs", type=int, default=200)
    p.add_argument("--freeze-expression", action="store_true", default=True)
    p.add_argument("--plan-kwargs", default=None)
    p.add_argument("--save-reference-anndata", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)
