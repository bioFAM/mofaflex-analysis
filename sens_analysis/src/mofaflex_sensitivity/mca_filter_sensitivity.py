from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mudata as md
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mofaflex_sensitivity.uninformed_sensitivity import fit_mofaflex_model
    from mofaflex_sensitivity.plot_style import CAT_PALETTE, clean_ax, make_color_map, savefig, set_house_style
else:
    from .plot_style import CAT_PALETTE, clean_ax, make_color_map, savefig, set_house_style
    from .uninformed_sensitivity import fit_mofaflex_model


DEFAULT_DATA_PATH = Path("/home/aqoku/projects/data/mfl_bench/sln_208_mofaflex.h5mu")
DEFAULT_HALLMARK_OUT_DIR = Path("artifacts/sln208_hallmark_mca_pruning_sensitivity")
DEFAULT_REACTOME_OUT_DIR = Path("artifacts/sln208_reactome_mca_pruning_sensitivity")
DEFAULT_MCA_GMT_URL = (
    "https://github.com/bioFAM/mofaflex-analysis/raw/refs/heads/main/msigdb/mmc5_gene.gmt"
)
DEFAULT_MSIGDB_CATEGORY = "mh.all"
DEFAULT_MSIGDB_DBVER = "2023.2.Mm"
DEFAULT_EXPECTED_PRIMARY_COUNT = 50
DEFAULT_EXPECTED_MCA_COUNT = 11
MCA_PREFIX = "MCA::"
DEFAULT_LIKELIHOODS = {"rna": "Normal", "prot": "Normal"}
DEFAULT_MODALITIES = ("rna", "prot")
DEFAULT_WEIGHT_PRIOR = "Horseshoe"
DEFAULT_NONNEGATIVE_WEIGHTS = True
DEFAULT_NONNEGATIVE_FACTORS = True
DEFAULT_INIT_FACTORS = 0.0
DEFAULT_INIT_SCALE = 0.1
DEFAULT_LR = 0.003
DEFAULT_N_PARTICLES = 1


@dataclass(frozen=True)
class PruningStep:
    step_index: int
    n_removed_smallest: int
    n_removed_largest: int
    n_gene_sets: int
    pruning_direction: str

    @property
    def step_label(self) -> str:
        return f"step_{self.step_index:02d}_n{self.n_gene_sets}"


@dataclass
class RunArtifact:
    step: PruningStep
    n_uninformed_factors: int
    gene_set_stats: pd.DataFrame
    ranking: pd.DataFrame


# Named presets for the MCA pruning sensitivity analysis.
#
# Each preset bundles the MSigDB collection, gene-set filter thresholds, and
# pruning schedule defaults for one analysis run.  CLI flags override any value
# here on a per-flag basis.
#
# Key parameters:
#   msigdb_category        — MSigDB collection identifier
#   gene_set_min_fraction  — min overlap fraction with RNA var_names (0 = disabled)
#   gene_set_min/max_count — gene-count filter applied after overlap check
#   gene_set_similarity_threshold — Jaccard threshold for merging near-duplicate sets (1.0 = off)
#   expected_primary_count — sanity-check: expected number of primary (non-MCA) sets
#   expected_mca_count     — sanity-check: expected number of MCA sets retained
#   prune_step_size        — number of gene sets removed per pruning step
#   min_informed_factors   — stop pruning when informed factor count falls below this
#   min_total_factors      — (optional) additional stopping criterion on total factors
#   n_uninformed_factors   — uninformed factors given to MOFA-FLEX at each step
#   prune_direction        — "smallest" (default) or "largest" sets pruned first;
#                            overridden at runtime by --prune-direction
COLLECTION_PRESETS = {
    # Hallmark gene sets (50 sets) + MCA cell-type markers (11 sets).
    # Small collection → fine-grained pruning steps of 5 sets.
    "hallmark": {
        "msigdb_category": "mh.all",
        "msigdb_dbver": "2023.2.Mm",
        "gene_set_min_fraction": 0.0,
        "gene_set_min_count": 1,
        "gene_set_max_count": 100000,
        "gene_set_similarity_threshold": 1.0,
        "expected_primary_count": 50,
        "expected_mca_count": 11,
        "prune_step_size": 5,
        "min_informed_factors": 20,
        "min_total_factors": None,
        "n_uninformed_factors": 3,
        "out_dir": DEFAULT_HALLMARK_OUT_DIR,
    },
    # Reactome pathways (271 sets after filtering) + MCA cell-type markers (10 sets).
    # Larger collection → coarser pruning steps of 20 sets.
    # Additional min_total_factors=50 prevents over-pruning the factor model.
    "reactome": {
        "msigdb_category": "m2.cp.reactome",
        "msigdb_dbver": "2023.2.Mm",
        "gene_set_min_fraction": 0.1,
        "gene_set_min_count": 15,
        "gene_set_max_count": 300,
        "gene_set_similarity_threshold": 0.8,
        "expected_primary_count": 271,
        "expected_mca_count": 10,
        "prune_step_size": 20,
        "min_informed_factors": 20,
        "min_total_factors": 50,
        "n_uninformed_factors": 3,
        "out_dir": DEFAULT_REACTOME_OUT_DIR,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a MOFA-FLEX prior-pruning sensitivity analysis on the prepared SLN208 "
            "mouse CITE-seq dataset using the full dataset, a chosen MSigDB collection, "
            "and MCA."
        )
    )
    parser.add_argument(
        "--collection",
        choices=sorted(COLLECTION_PRESETS),
        default="hallmark",
        help="Named preset for the primary MSigDB collection plus default pruning/filter settings.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. If omitted, a collection-specific default is used.",
    )
    parser.add_argument("--mca-gmt-url", default=DEFAULT_MCA_GMT_URL)
    parser.add_argument("--msigdb-category", default=DEFAULT_MSIGDB_CATEGORY)
    parser.add_argument("--msigdb-dbver", default=DEFAULT_MSIGDB_DBVER)
    parser.add_argument("--gene-set-min-fraction", type=float, default=0.0)
    parser.add_argument("--gene-set-min-count", type=int, default=1)
    parser.add_argument("--gene-set-max-count", type=int, default=100000)
    parser.add_argument("--gene-set-similarity-threshold", type=float, default=1.0)
    parser.add_argument("--prune-step-size", type=int, default=5)
    parser.add_argument(
        "--prune-direction",
        choices=["smallest", "largest"],
        default="smallest",
        help="Whether each pruning step removes the smallest or largest gene sets first.",
    )
    parser.add_argument("--min-informed-factors", type=int, default=20)
    parser.add_argument("--min-total-factors", type=int, default=None)
    parser.add_argument("--n-uninformed-factors", type=int, default=3)
    parser.add_argument(
        "--expected-primary-count",
        "--expected-hallmark-count",
        dest="expected_primary_count",
        type=int,
        default=DEFAULT_EXPECTED_PRIMARY_COUNT,
    )
    parser.add_argument("--expected-mca-count", type=int, default=DEFAULT_EXPECTED_MCA_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=sorted(DEFAULT_LIKELIHOODS),
        default=list(DEFAULT_MODALITIES),
        help="Modalities to include when fitting MOFA-FLEX.",
    )
    parser.add_argument("--expected-mca-pathways", nargs="*", default=[])
    parser.add_argument("--selected-mca-rank-pathways", nargs="*", default=[])
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[3, 5, 10])
    return parser


def infer_primary_source_name(msigdb_category: str) -> str:
    category = msigdb_category.lower()
    if category == "mh.all":
        return "hallmark"
    if "reactome" in category:
        return "reactome"
    return re.sub(r"[^a-z0-9]+", "_", category).strip("_") or "msigdb"


def primary_prefix(primary_source_name: str) -> str:
    return f"{primary_source_name.upper()}::"


def prettify_pathway_name(pathway_name: str) -> str:
    cleaned = pathway_name.strip()
    if not cleaned:
        return cleaned

    if cleaned == "all_models":
        return "All models"

    collection = ""
    payload = cleaned
    if "::" in cleaned:
        collection, payload = cleaned.split("::", 1)
    else:
        match = re.match(r"^([A-Z0-9]+)_(.+)$", cleaned)
        if match is not None:
            collection, payload = match.group(1), match.group(2)

    collection = collection.strip().upper()
    if collection == "MCA":
        payload = re.sub(r"^HAN[_\s]+", "", payload)

    payload = re.sub(r"[_\s]+", " ", payload).strip()
    pretty_payload = payload.title()

    if not collection:
        return pretty_payload

    collection_letter = next((char for char in collection if char.isalpha()), "")
    if not collection_letter:
        return pretty_payload
    return f"{pretty_payload} [{collection_letter}]"


def _parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = text.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {text!r}.")


def _cli_provided(flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in sys.argv[1:])


def resolve_collection_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = COLLECTION_PRESETS[args.collection]
    field_to_flag = {
        "out_dir": "--out-dir",
        "msigdb_category": "--msigdb-category",
        "msigdb_dbver": "--msigdb-dbver",
        "gene_set_min_fraction": "--gene-set-min-fraction",
        "gene_set_min_count": "--gene-set-min-count",
        "gene_set_max_count": "--gene-set-max-count",
        "gene_set_similarity_threshold": "--gene-set-similarity-threshold",
        "expected_primary_count": "--expected-primary-count",
        "expected_mca_count": "--expected-mca-count",
        "prune_step_size": "--prune-step-size",
        "min_informed_factors": "--min-informed-factors",
        "min_total_factors": "--min-total-factors",
        "n_uninformed_factors": "--n-uninformed-factors",
    }

    for field_name, flag in field_to_flag.items():
        if _cli_provided(flag):
            continue
        setattr(args, field_name, preset[field_name])
    return args


def normalize_modalities(modalities: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if modalities is None:
        return tuple(DEFAULT_MODALITIES)
    normalized = tuple(dict.fromkeys(str(modality) for modality in modalities))
    if not normalized:
        raise ValueError("At least one modality must be selected.")
    invalid = sorted(set(normalized) - set(DEFAULT_LIKELIHOODS))
    if invalid:
        raise ValueError(f"Unknown modalities requested: {invalid}.")
    return normalized


def resolve_likelihoods_for_mdata(mdata: md.MuData) -> dict[str, str]:
    return {
        modality: DEFAULT_LIKELIHOODS[modality]
        for modality in normalize_modalities(tuple(mdata.mod.keys()))
        if modality in mdata.mod
    }


def load_sln208_mdata(data_path: Path, *, modalities: list[str] | tuple[str, ...] | None = None) -> md.MuData:
    mdata = md.read_h5mu(data_path)
    selected_modalities = normalize_modalities(tuple(mdata.mod.keys()) if modalities is None else modalities)
    missing = [modality for modality in selected_modalities if modality not in mdata.mod]
    if missing:
        raise ValueError(f"Requested modalities not present in input MuData: {missing}.")
    if tuple(mdata.mod.keys()) != selected_modalities:
        mdata = md.MuData({modality: mdata.mod[modality].copy() for modality in selected_modalities})
    for adata in mdata.mod.values():
        adata.X = adata.X.astype(np.float32)
    return mdata


def _to_upper_feature_sets(mfl, feature_set_collection):
    return mfl.FeatureSets(
        [mfl.FeatureSet([feature.upper() for feature in feature_set], feature_set.name) for feature_set in feature_set_collection],
        name=feature_set_collection.name,
    )


def _prefix_feature_sets(mfl, feature_set_collection, *, prefix: str, collection_name: str):
    return mfl.FeatureSets(
        [mfl.FeatureSet(list(feature_set), f"{prefix}{feature_set.name}") for feature_set in feature_set_collection],
        name=collection_name,
    )


def _merge_identical_feature_sets(mfl, feature_set_collection, *, similarity_threshold: float):
    return feature_set_collection.merge_similar(
        metric="jaccard",
        similarity_threshold=similarity_threshold,
        iteratively=True,
    )


def _feature_set_stats(feature_set_collection, *, source: str) -> pd.DataFrame:
    records = []
    for feature_set in feature_set_collection:
        records.append(
            {
                "pathway_name": feature_set.name,
                "source": source,
                "set_size": len(list(feature_set)),
                "features": list(feature_set),
            }
        )
    return pd.DataFrame.from_records(records)


def load_sorted_gene_set_stats(
    *,
    var_names: pd.Index,
    mca_gmt_url: str,
    msigdb_category: str,
    msigdb_dbver: str,
    gene_set_min_fraction: float,
    gene_set_min_count: int,
    gene_set_max_count: int,
    gene_set_similarity_threshold: float,
    expected_primary_count: int,
    expected_mca_count: int,
) -> pd.DataFrame:
    import mofaflex as mfl

    source_name = infer_primary_source_name(msigdb_category)
    source_prefix = primary_prefix(source_name)

    with tempfile.NamedTemporaryFile("w", suffix=".gmt", delete=True) as tmp:
        with fsspec.open(mca_gmt_url, mode="rt", client_kwargs={"trust_env": True}) as handle:
            tmp.write(handle.read())
            tmp.flush()
        mca_collection = _to_upper_feature_sets(
            mfl, mfl.FeatureSets.from_gmt(tmp.name, name="mca")
        ).filter(
            var_names,
            min_fraction=gene_set_min_fraction,
            min_count=gene_set_min_count,
            max_count=gene_set_max_count,
        )

    primary_collection = _to_upper_feature_sets(
        mfl, mfl.tl.msigdb_get_features(category=msigdb_category, dbver=msigdb_dbver)
    ).filter(
        var_names,
        min_fraction=gene_set_min_fraction,
        min_count=gene_set_min_count,
        max_count=gene_set_max_count,
    )

    primary_collection = _merge_identical_feature_sets(
        mfl,
        primary_collection,
        similarity_threshold=gene_set_similarity_threshold,
    )
    mca_collection = _merge_identical_feature_sets(
        mfl,
        mca_collection,
        similarity_threshold=gene_set_similarity_threshold,
    )

    if len(primary_collection) != expected_primary_count or len(mca_collection) != expected_mca_count:
        raise ValueError(
            "Unexpected retained gene-set counts after filtering: "
            f"{source_name}={len(primary_collection)} (expected {expected_primary_count}), "
            f"mca={len(mca_collection)} (expected {expected_mca_count})."
        )

    primary_collection = _prefix_feature_sets(
        mfl,
        primary_collection,
        prefix=source_prefix,
        collection_name=source_name,
    )
    mca_collection = _prefix_feature_sets(
        mfl,
        mca_collection,
        prefix=MCA_PREFIX,
        collection_name="mca",
    )

    stats = pd.concat(
        [
            _feature_set_stats(primary_collection, source=source_name),
            _feature_set_stats(mca_collection, source="mca"),
        ],
        ignore_index=True,
    )
    return stats.sort_values(["set_size", "pathway_name"], ascending=[True, True]).reset_index(drop=True)


def load_collection_gene_set_stats(
    *,
    var_names: pd.Index,
    collection: str,
    mca_gmt_url: str = DEFAULT_MCA_GMT_URL,
) -> pd.DataFrame:
    if collection not in COLLECTION_PRESETS:
        raise ValueError(f"Unknown collection preset {collection!r}.")

    preset = COLLECTION_PRESETS[collection]
    return load_sorted_gene_set_stats(
        var_names=var_names,
        mca_gmt_url=mca_gmt_url,
        msigdb_category=preset["msigdb_category"],
        msigdb_dbver=preset["msigdb_dbver"],
        gene_set_min_fraction=preset["gene_set_min_fraction"],
        gene_set_min_count=preset["gene_set_min_count"],
        gene_set_max_count=preset["gene_set_max_count"],
        gene_set_similarity_threshold=preset["gene_set_similarity_threshold"],
        expected_primary_count=preset["expected_primary_count"],
        expected_mca_count=preset["expected_mca_count"],
    )


def build_feature_set_collection(gene_set_stats: pd.DataFrame):
    import mofaflex as mfl

    return mfl.FeatureSets(
        [
            mfl.FeatureSet(list(row.features), row.pathway_name)
            for row in gene_set_stats.itertuples(index=False)
        ],
        name="primary_mca_sorted",
    )


def attach_gene_set_mask(
    mdata: md.MuData,
    gene_set_stats: pd.DataFrame,
    *,
    mask_key: str = "gene_set_mask",
) -> md.MuData:
    updated = mdata.copy()
    gene_set_collection = build_feature_set_collection(gene_set_stats)
    updated.mod["rna"].varm[mask_key] = gene_set_collection.to_mask(updated.mod["rna"].var_names.tolist()).T
    return updated


def fit_model(
    *,
    mdata: md.MuData,
    n_uninformed_factors: int,
    seed: int,
    save_path: str | Path | bool = False,
):
    import mofaflex as mfl

    data_options = mfl.DataOptions(
        group_by=None,
        scale_per_group=False,
        annotations_varm_key={"rna": "gene_set_mask"},
        plot_data_overview=False,
    )
    model_options = mfl.ModelOptions(
        n_factors=n_uninformed_factors,
        weight_prior=DEFAULT_WEIGHT_PRIOR,
        likelihoods=resolve_likelihoods_for_mdata(mdata),
        nonnegative_weights=DEFAULT_NONNEGATIVE_WEIGHTS,
        nonnegative_factors=DEFAULT_NONNEGATIVE_FACTORS,
        init_factors=DEFAULT_INIT_FACTORS,
        init_scale=DEFAULT_INIT_SCALE,
    )
    training_options = mfl.TrainingOptions(
        lr=DEFAULT_LR,
        n_particles=DEFAULT_N_PARTICLES,
        save_path=save_path,
        seed=seed,
    )
    return fit_mofaflex_model(mdata, data_options, model_options, training_options)


def extract_informed_pathway_ranking(model, gene_set_stats: pd.DataFrame) -> pd.DataFrame:
    informed_names = list(
        model.factor_names[model.n_uninformed_factors : model.n_uninformed_factors + model.n_informed_factors]
    )
    r2_by_group = model.get_r2(total=False, ordered=False)
    per_view = pd.DataFrame(index=pd.Index(model.factor_names, dtype="object"))
    for group_df in r2_by_group.values():
        for view_name in group_df.columns:
            column_name = f"r2_{view_name}"
            if column_name not in per_view.columns:
                per_view[column_name] = 0.0
            per_view.loc[group_df.index, column_name] = (
                per_view.loc[group_df.index, column_name].to_numpy(dtype=float)
                + group_df.loc[:, view_name].to_numpy(dtype=float)
            )

    ranking = pd.DataFrame({"pathway_name": informed_names})
    for column_name in per_view.columns:
        ranking[column_name] = per_view.loc[informed_names, column_name].to_numpy(dtype=float)
    r2_columns = [column_name for column_name in ranking.columns if column_name.startswith("r2_")]
    ranking["variance_explained"] = ranking.loc[:, r2_columns].sum(axis=1) if r2_columns else 0.0
    ranking = ranking.sort_values("variance_explained", ascending=False, kind="stable")
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    ranking = ranking.merge(gene_set_stats.drop(columns="features", errors="ignore"), on="pathway_name", how="left")
    missing_source = ranking["source"].isna()
    ranking.loc[missing_source, "source"] = np.where(
        ranking.loc[missing_source, "pathway_name"].str.startswith(MCA_PREFIX),
        "mca",
        "hallmark",
    )
    ranking["is_mca"] = ranking["source"].eq("mca")
    ranking["assigned_pathway"] = ranking["pathway_name"]
    return ranking.reset_index(drop=True)


def top_k_pathways(ranking: pd.DataFrame, k: int) -> list[str]:
    return ranking.sort_values("rank").head(k)["pathway_name"].tolist()


def jaccard_overlap(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def rank_series_for_universe(ranking: pd.DataFrame, universe: list[str]) -> pd.Series:
    rank_map = ranking.set_index("pathway_name")["rank"].to_dict()
    default_rank = len(ranking) + 1
    return pd.Series({pathway: float(rank_map.get(pathway, default_rank)) for pathway in universe}, dtype=float)


def spearman_rank_stability(left: pd.DataFrame, right: pd.DataFrame) -> float:
    universe = sorted(set(left["pathway_name"]) | set(right["pathway_name"]))
    if len(universe) < 2:
        return np.nan
    left_ranks = rank_series_for_universe(left, universe)
    right_ranks = rank_series_for_universe(right, universe)
    corr, _ = spearmanr(left_ranks.to_numpy(), right_ranks.to_numpy())
    return float(corr) if corr == corr else np.nan


def canonicalize_mca_names(pathways: list[str]) -> set[str]:
    normalized = set()
    for pathway in pathways:
        cleaned = pathway.strip()
        if not cleaned:
            continue
        if cleaned.startswith(MCA_PREFIX):
            normalized.add(cleaned)
        else:
            normalized.add(f"{MCA_PREFIX}{cleaned}")
    return normalized


def build_pruning_steps(
    total_gene_sets: int,
    *,
    prune_step_size: int,
    prune_direction: str = "smallest",
    min_informed_factors: int,
    min_total_factors: int | None,
    n_uninformed_factors: int,
) -> list[PruningStep]:
    if prune_step_size <= 0:
        raise ValueError("`prune_step_size` must be positive.")
    if prune_direction not in {"smallest", "largest"}:
        raise ValueError("`prune_direction` must be either 'smallest' or 'largest'.")
    if min_informed_factors <= 0:
        raise ValueError("`min_informed_factors` must be positive.")
    if min_total_factors is not None and min_total_factors <= 0:
        raise ValueError("`min_total_factors` must be positive when provided.")

    effective_min_informed = min_informed_factors
    if min_total_factors is not None:
        effective_min_informed = max(effective_min_informed, min_total_factors - n_uninformed_factors)

    steps = []
    step_index = 0
    removed = 0
    while total_gene_sets - removed >= effective_min_informed:
        n_gene_sets = total_gene_sets - removed
        steps.append(
            PruningStep(
                step_index=step_index,
                n_removed_smallest=removed if prune_direction == "smallest" else 0,
                n_removed_largest=removed if prune_direction == "largest" else 0,
                n_gene_sets=n_gene_sets,
                pruning_direction=prune_direction,
            )
        )
        step_index += 1
        removed += prune_step_size
    return steps


def retained_gene_set_stats_for_step(base_gene_set_stats: pd.DataFrame, step: PruningStep) -> pd.DataFrame:
    if step.pruning_direction == "smallest":
        return base_gene_set_stats.iloc[step.n_removed_smallest :, :].reset_index(drop=True)
    if step.pruning_direction == "largest":
        end = len(base_gene_set_stats) - step.n_removed_largest
        return base_gene_set_stats.iloc[:end, :].reset_index(drop=True)
    raise ValueError(f"Unknown pruning direction: {step.pruning_direction!r}.")


def summarize_run(
    artifact: RunArtifact,
    *,
    reference_ranking: pd.DataFrame,
    expected_mca_pathways: set[str],
    top_k_values: list[int],
) -> dict[str, Any]:
    ranking = artifact.ranking
    n_retained_mca = int((artifact.gene_set_stats["source"] == "mca").sum())
    primary_sources = artifact.gene_set_stats.loc[artifact.gene_set_stats["source"] != "mca", "source"].unique().tolist()
    primary_source_name = primary_sources[0] if primary_sources else "primary"
    top_10 = top_k_pathways(ranking, 10)
    top_10_set = set(top_10)
    top_10_mca = [pathway for pathway in top_10 if pathway.startswith(MCA_PREFIX)]
    expected_in_top_10 = sorted(expected_mca_pathways & top_10_set)
    set_sizes = artifact.gene_set_stats["set_size"]

    row: dict[str, Any] = {
        "step_index": artifact.step.step_index,
        "step_label": artifact.step.step_label,
        "pruning_direction": artifact.step.pruning_direction,
        "n_removed_smallest": artifact.step.n_removed_smallest,
        "n_removed_largest": artifact.step.n_removed_largest,
        "n_uninformed_factors": artifact.n_uninformed_factors,
        "primary_source_name": primary_source_name,
        "n_gene_sets": int(len(artifact.gene_set_stats)),
        "n_primary_gene_sets": int((artifact.gene_set_stats["source"] == primary_source_name).sum()),
        "n_mca_gene_sets": int((artifact.gene_set_stats["source"] == "mca").sum()),
        "smallest_retained_set_size": int(set_sizes.min()),
        "median_retained_set_size": float(set_sizes.median()),
        "largest_retained_set_size": int(set_sizes.max()),
        "retained_pathways": json.dumps(artifact.gene_set_stats["pathway_name"].tolist()),
        "top_10_pathways": json.dumps(top_10),
        "top_10_mca_pathways": json.dumps(top_10_mca),
        "n_mca_in_top_10": int(len(top_10_mca)),
        "mca_fraction_in_top_10": float(len(top_10_mca) / max(len(top_10), 1)),
        "mca_recall_top10": float(len(top_10_mca) / n_retained_mca) if n_retained_mca > 0 else np.nan,
        "n_expected_mca_in_top_10": int(len(expected_in_top_10)),
        "expected_mca_in_top_10": json.dumps(expected_in_top_10),
        "spearman_rank_vs_full": spearman_rank_stability(ranking, reference_ranking),
    }
    for k in top_k_values:
        top_k = top_k_pathways(ranking, k)
        n_mca_in_top_k = sum(pathway.startswith(MCA_PREFIX) for pathway in top_k)
        row[f"n_mca_in_top{k}"] = int(n_mca_in_top_k)
        row[f"mca_fraction_in_top{k}"] = float(n_mca_in_top_k / max(len(top_k), 1))
        row[f"mca_recall_top{k}"] = float(n_mca_in_top_k / n_retained_mca) if n_retained_mca > 0 else np.nan
        row[f"jaccard_top{k}_vs_full"] = jaccard_overlap(
            set(top_k),
            set(top_k_pathways(reference_ranking, k)),
        )
    return row


def build_pairwise_jaccard_matrix(run_artifacts: list[RunArtifact], *, k: int) -> pd.DataFrame:
    labels = [artifact.step.step_label for artifact in run_artifacts]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    top_sets = {
        artifact.step.step_label: set(top_k_pathways(artifact.ranking, k))
        for artifact in run_artifacts
    }
    for left in labels:
        for right in labels:
            matrix.loc[left, right] = jaccard_overlap(top_sets[left], top_sets[right])
    return matrix


def build_pairwise_spearman_matrix(run_artifacts: list[RunArtifact]) -> pd.DataFrame:
    labels = [artifact.step.step_label for artifact in run_artifacts]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    rankings = {artifact.step.step_label: artifact.ranking for artifact in run_artifacts}
    for left in labels:
        for right in labels:
            matrix.loc[left, right] = spearman_rank_stability(rankings[left], rankings[right])
    return matrix


def save_pairwise_heatmap(matrix: pd.DataFrame, *, title: str, path: Path) -> None:
    fig_size = max(7.0, min(18.0, 0.55 * len(matrix)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(matrix, ax=ax, cmap="viridis", vmin=0.0, vmax=1.0, square=True, cbar=True)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_pairwise_matrix(
    matrix: pd.DataFrame,
    *,
    metric_name: str,
    top_k_genes: int,
) -> pd.DataFrame:
    values: list[float] = []
    labels = matrix.index.tolist()
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            values.append(float(matrix.loc[left, right]))

    if not values:
        return pd.DataFrame(
            [
                {
                    "metric": metric_name,
                    "top_k_genes": int(top_k_genes),
                    "n_pairs": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            ]
        )

    arr = np.asarray(values, dtype=float)
    return pd.DataFrame(
        [
            {
                "metric": metric_name,
                "top_k_genes": int(top_k_genes),
                "n_pairs": int(len(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        ]
    )


def save_top_pathway_topk_correlation_focus_heatmaps(
    *,
    pearson_matrix: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    top_k_genes: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), squeeze=False)
    configs = [
        (axes[0, 0], pearson_matrix, "Pearson"),
        (axes[0, 1], spearman_matrix, "Spearman"),
    ]
    for ax, matrix, label in configs:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=True,
            cbar=True,
        )
        ax.set_title(f"{label} at top {top_k_genes} genes")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Top-pathway concordance across models", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_gene_set_count_plot(summary_df: pd.DataFrame, *, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=summary_df, x="n_gene_sets", y="n_mca_in_top_10", marker="o", ax=ax)
    ax.set_title("Retained prior size vs MCA pathways in the top 10")
    ax.set_xlabel("Number of retained informed pathways")
    ax.set_ylabel("MCA pathways among top 10 informed factors")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_reference_jaccard_plot(summary_df: pd.DataFrame, *, k: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=summary_df, x="n_gene_sets", y=f"jaccard_top{k}_vs_full", marker="o", ax=ax)
    ax.set_title(f"Top-{k} Jaccard overlap against the full prior")
    ax.set_xlabel("Number of retained informed pathways")
    ax.set_ylabel("Jaccard overlap")
    ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mca_recall_plot(summary_df: pd.DataFrame, *, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(
        data=summary_df,
        x="n_gene_sets",
        y="mca_recall_top10",
        marker="o",
        ax=ax,
    )
    ax.set_title("Retained MCA recall in the top 10")
    ax.set_xlabel("Number of retained informed pathways")
    ax.set_ylabel("Retained MCA recall in top 10")
    ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_selected_pathway_rank_heatmap(
    summary_df: pd.DataFrame,
    run_artifacts: list[RunArtifact],
    *,
    selected_pathways: list[str],
    path: Path,
) -> None:
    if not selected_pathways:
        return

    ranking_maps = {
        artifact.step.step_label: artifact.ranking.set_index("pathway_name")["rank"].to_dict()
        for artifact in run_artifacts
    }
    rows = []
    for pathway in selected_pathways:
        rows.append(
            {
                step_label: ranking_maps[step_label].get(pathway, np.nan)
                for step_label in summary_df["step_label"]
            }
        )
    matrix = pd.DataFrame(rows, index=selected_pathways)
    fig_width = max(8.0, 0.7 * matrix.shape[1])
    fig_height = max(3.0, 0.45 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(matrix, cmap="mako_r", ax=ax, cbar=True)
    ax.set_title("Ranks of selected MCA pathways across pruning steps")
    ax.set_xlabel("Pruning step")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_pathway_variance_heatmap(
    summary_df: pd.DataFrame,
    run_artifacts: list[RunArtifact],
    *,
    top_k: int,
    path: Path,
) -> None:
    ranking_by_step = {
        artifact.step.step_label: artifact.ranking.sort_values("rank").reset_index(drop=True)
        for artifact in run_artifacts
    }
    ordered_pathways: list[str] = []
    seen: set[str] = set()
    for step_label in summary_df["step_label"]:
        top_pathways = ranking_by_step[step_label].head(top_k)["pathway_name"].tolist()
        for pathway in top_pathways:
            if pathway in seen:
                continue
            ordered_pathways.append(pathway)
            seen.add(pathway)

    if not ordered_pathways:
        return

    matrix = pd.DataFrame(0.0, index=ordered_pathways, columns=summary_df["step_label"].tolist())
    for step_label in summary_df["step_label"]:
        values = ranking_by_step[step_label].set_index("pathway_name")["variance_explained"]
        matrix.loc[:, step_label] = values.reindex(ordered_pathways).fillna(0.0).to_numpy()

    fig_width = max(8.0, 0.7 * matrix.shape[1])
    fig_height = max(6.0, 0.35 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(matrix, cmap="mako", ax=ax, cbar=True)
    ax.set_title(f"Variance explained for the union of top-{top_k} pathways")
    ax.set_xlabel("Pruning step")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_pathway_variance_heatmaps_by_model(
    summary_df: pd.DataFrame,
    run_artifacts: list[RunArtifact],
    *,
    top_k: int,
    path: Path,
) -> None:
    ranking_by_step = {
        artifact.step.step_label: artifact.ranking.sort_values("rank").reset_index(drop=True)
        for artifact in run_artifacts
    }
    step_labels = summary_df["step_label"].tolist()
    n_panels = len(step_labels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), squeeze=False)

    value_columns = ["r2_rna", "r2_prot"]
    pretty_columns = ["RNA", "Protein"]

    for ax, step_label in zip(axes.flat, step_labels, strict=False):
        ranking = ranking_by_step[step_label].head(top_k).copy()
        available_columns = [column for column in value_columns if column in ranking.columns]
        plot_matrix = ranking.loc[:, available_columns].copy()
        plot_matrix.columns = [pretty_columns[value_columns.index(column)] for column in available_columns]
        plot_matrix.index = ranking["pathway_name"]

        sns.heatmap(plot_matrix, cmap="mako", ax=ax, cbar=True)
        ax.set_title(step_label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"Top-{top_k} pathway variance heatmaps by model", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_top_pathway_loading_plots_by_model(
    summary_df: pd.DataFrame,
    *,
    models_dir: Path,
    rankings_dir: Path,
    plots_dir: Path,
    view: str = "rna",
    n_features: int = 30,
) -> None:
    import mofaflex as mfl
    import plotnine as p9

    output_dir = plots_dir / "top_pathway_loadings_by_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_rows = list(summary_df.sort_values("step_index").itertuples(index=False))
    panel_records: list[dict[str, Any]] = []
    top_pathways = []
    for row in model_rows:
        ranking = pd.read_csv(rankings_dir / f"{row.step_label}_pathway_ranks.csv").sort_values("rank")
        pathway_name = str(ranking.iloc[0]["pathway_name"])
        if pathway_name not in top_pathways:
            top_pathways.append(pathway_name)

    if not top_pathways or not model_rows:
        return

    n_rows = len(top_pathways)
    n_cols = len(model_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 5.8 * n_rows), squeeze=False)

    for row_index, pathway_name in enumerate(top_pathways):
        safe_pathway = re.sub(r"[^A-Za-z0-9._-]+", "_", pathway_name)[:120]
        for col_index, row in enumerate(model_rows):
            model = mfl.MOFAFLEX.load(models_dir / f"{row.step_label}.h5", map_location="cpu")
            ranking = pd.read_csv(rankings_dir / f"{row.step_label}_pathway_ranks.csv").sort_values("rank")
            ax = axes[row_index, col_index]
            if pathway_name in set(ranking["pathway_name"]):
                panel_width = 5.2 if n_features > 20 else 4.3
                panel_height = 5.8 if n_features > 20 else 4.0
                plot = mfl.pl.top_weights(
                    model,
                    n_features=n_features,
                    views=view,
                    factors=pathway_name,
                    figsize=(panel_width, panel_height),
                )
                plot = plot + p9.theme_light() + p9.theme(
                    figure_size=(panel_width, panel_height),
                    axis_text_x=p9.element_text(size=10),
                    axis_text_y=p9.element_text(size=10),
                    axis_title_x=p9.element_text(size=11),
                    axis_title_y=p9.element_text(size=11),
                    legend_title=p9.element_text(size=10),
                    legend_text=p9.element_text(size=9),
                    plot_title=p9.element_text(size=9.5, ha="center", margin={"b": 1, "unit": "pt"}),
                    strip_background=p9.element_rect(fill="white", color="white"),
                    strip_text=p9.element_text(size=10),
                    panel_background=p9.element_rect(fill="white", color="white"),
                    plot_background=p9.element_rect(fill="white", color="white"),
                    legend_background=p9.element_rect(fill="white", color="white"),
                    legend_key=p9.element_rect(fill="white", color="white"),
                )
                plot = plot + p9.labs(
                    title=f"{prettify_pathway_name(pathway_name)} ({int(row.n_gene_sets)} gene programs)"
                )
                out_path = output_dir / f"{row.step_label}_{safe_pathway}_{view}_top_weights.png"
                fig_single = plot.draw(show=False)
                fig_single.patch.set_facecolor("white")
                fig_single.savefig(
                    str(out_path),
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="white",
                    transparent=False,
                )
                plt.close(fig_single)
                ax.imshow(mpimg.imread(out_path))
                ax.set_title(f"({int(row.n_gene_sets)} gene programs)", fontsize=10)
                panel_records.append(
                    {
                        "step_index": int(row.step_index),
                        "step_label": row.step_label,
                        "n_gene_sets": int(row.n_gene_sets),
                        "pathway_name": pathway_name,
                        "image_path": out_path,
                    }
                )
            else:
                ax.axis("off")
                continue
            ax.axis("off")
        axes[row_index, 0].set_ylabel(prettify_pathway_name(pathway_name), fontsize=12, rotation=90, labelpad=18)

    fig.tight_layout()
    fig.savefig(plots_dir / f"top_pathway_loadings_{view}_overview.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    if not panel_records:
        return

    panel_df = pd.DataFrame(panel_records).sort_values("step_index").drop_duplicates(
        subset=["step_label", "n_gene_sets"],
        keep="first",
    )
    n_panels = len(panel_df)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 7.0 * nrows), squeeze=False)
    for ax, (_, panel_row) in zip(axes.flat, panel_df.iterrows(), strict=False):
        ax.imshow(mpimg.imread(panel_row["image_path"]))
        ax.set_title(f"({int(panel_row['n_gene_sets'])} gene programs)", fontsize=10)
        ax.axis("off")
    for ax in axes.flat[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / f"top_pathway_loadings_{view}_overview_single_row.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    top_pathway_by_step = {}
    for row in model_rows:
        ranking = pd.read_csv(rankings_dir / f"{row.step_label}_pathway_ranks.csv").sort_values("rank")
        top_pathway_by_step[row.step_label] = str(ranking.iloc[0]["pathway_name"])

    panel_lookup = pd.DataFrame(panel_records)
    for target_pathway in top_pathways:
        selected_rows = []
        for row in model_rows:
            if top_pathway_by_step.get(row.step_label) != target_pathway:
                continue
            match = panel_lookup.loc[
                (panel_lookup["step_label"] == row.step_label) & (panel_lookup["pathway_name"] == target_pathway),
                :,
            ]
            if match.empty:
                continue
            selected_rows.append(match.iloc[0])

        if not selected_rows:
            continue

        focused_df = pd.DataFrame(selected_rows).sort_values("step_index").reset_index(drop=True)
        n_focus = len(focused_df)
        fig, axes = plt.subplots(1, n_focus, figsize=(5.6 * n_focus, 7.0), squeeze=False)
        for ax, (_, panel_row) in zip(axes.flat, focused_df.iterrows(), strict=False):
            ax.imshow(mpimg.imread(panel_row["image_path"]))
            ax.set_title(f"({int(panel_row['n_gene_sets'])} gene programs)", fontsize=10)
            ax.axis("off")
        pretty_target = prettify_pathway_name(target_pathway)
        fig.suptitle(pretty_target, fontsize=14, y=0.98)
        fig.tight_layout()
        safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", target_pathway).lower()
        fig.savefig(
            plots_dir / f"top_pathway_loadings_{view}_{safe_target}_focused.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.close(fig)


def extract_top_pathway_gene_rankings_by_model(
    summary_df: pd.DataFrame,
    *,
    models_dir: Path,
    rankings_dir: Path,
    view: str,
    top_k_values: list[int],
) -> pd.DataFrame:
    import mofaflex as mfl

    max_top_k = max(top_k_values)
    records: list[dict[str, Any]] = []

    for row in summary_df.sort_values("step_index").itertuples(index=False):
        ranking = pd.read_csv(rankings_dir / f"{row.step_label}_pathway_ranks.csv").sort_values("rank")
        top_pathway = str(ranking.iloc[0]["pathway_name"])
        model = mfl.MOFAFLEX.load(models_dir / f"{row.step_label}.h5", map_location="cpu")
        weights = model.get_weights(return_type="pandas")[view]
        factor_weights = weights.loc[top_pathway].astype(float)
        abs_sorted = factor_weights.abs().sort_values(ascending=False, kind="stable")

        for gene_rank, gene_name in enumerate(abs_sorted.index[:max_top_k], start=1):
            signed_weight = float(factor_weights.loc[gene_name])
            records.append(
                {
                    "step_index": int(row.step_index),
                    "step_label": row.step_label,
                    "top_pathway": top_pathway,
                    "view": view,
                    "gene_rank": int(gene_rank),
                    "gene_name": str(gene_name),
                    "signed_weight": signed_weight,
                    "abs_weight": float(abs(signed_weight)),
                }
            )

    return pd.DataFrame.from_records(records)


def build_top_gene_jaccard_matrices(
    top_gene_rankings: pd.DataFrame,
    *,
    top_k_values: list[int],
) -> dict[int, pd.DataFrame]:
    step_labels = (
        top_gene_rankings.loc[:, ["step_index", "step_label"]]
        .drop_duplicates()
        .sort_values("step_index")["step_label"]
        .tolist()
    )
    top_gene_sets: dict[int, dict[str, set[str]]] = {}
    for k in top_k_values:
        within_k = top_gene_rankings.loc[top_gene_rankings["gene_rank"] <= k]
        top_gene_sets[k] = {
            step_label: set(step_df["gene_name"].tolist())
            for step_label, step_df in within_k.groupby("step_label", sort=False)
        }

    matrices: dict[int, pd.DataFrame] = {}
    for k in top_k_values:
        matrix = pd.DataFrame(np.nan, index=step_labels, columns=step_labels)
        for left in step_labels:
            for right in step_labels:
                matrix.loc[left, right] = jaccard_overlap(
                    top_gene_sets[k].get(left, set()),
                    top_gene_sets[k].get(right, set()),
                )
        matrices[k] = matrix
    return matrices


def _correlation_from_series(left: pd.Series, right: pd.Series, *, method: str) -> float:
    if left.empty or right.empty:
        return np.nan
    if float(left.std(ddof=0)) == 0.0 or float(right.std(ddof=0)) == 0.0:
        return np.nan
    return float(left.corr(right, method=method))


def build_top_gene_weight_correlation_matrices(
    top_gene_rankings: pd.DataFrame,
    *,
    top_k_values: list[int],
    weight_column: str = "signed_weight",
    method: str = "pearson",
) -> dict[int, pd.DataFrame]:
    step_labels = (
        top_gene_rankings.loc[:, ["step_index", "step_label"]]
        .drop_duplicates()
        .sort_values("step_index")["step_label"]
        .tolist()
    )
    by_step = {
        step_label: step_df.copy()
        for step_label, step_df in top_gene_rankings.groupby("step_label", sort=False)
    }

    matrices: dict[int, pd.DataFrame] = {}
    for k in top_k_values:
        matrix = pd.DataFrame(np.nan, index=step_labels, columns=step_labels)
        top_k_tables = {
            step_label: (
                step_df.loc[step_df["gene_rank"] <= k, ["gene_name", weight_column]]
                .drop_duplicates(subset="gene_name", keep="first")
                .set_index("gene_name")[weight_column]
                .astype(float)
            )
            for step_label, step_df in by_step.items()
        }
        for left in step_labels:
            for right in step_labels:
                gene_union = sorted(set(top_k_tables[left].index) | set(top_k_tables[right].index))
                left_weights = top_k_tables[left].reindex(gene_union).fillna(0.0)
                right_weights = top_k_tables[right].reindex(gene_union).fillna(0.0)
                matrix.loc[left, right] = _correlation_from_series(
                    left_weights,
                    right_weights,
                    method=method,
                )
        matrices[k] = matrix
    return matrices


def save_top_gene_jaccard_heatmaps(
    top_gene_jaccard: dict[int, pd.DataFrame],
    *,
    path: Path,
) -> None:
    if not top_gene_jaccard:
        return

    ordered_k = sorted(top_gene_jaccard)
    fig, axes = plt.subplots(1, len(ordered_k), figsize=(4.8 * len(ordered_k), 4.6), squeeze=False)
    for ax, k in zip(axes.flat, ordered_k, strict=False):
        sns.heatmap(
            top_gene_jaccard[k],
            ax=ax,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            square=True,
            cbar=True,
        )
        ax.set_title(f"Top-{k} genes")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Jaccard overlap of top-pathway RNA genes across models", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_gene_weight_correlation_heatmaps(
    top_gene_correlations: dict[int, pd.DataFrame],
    *,
    path: Path,
    metric_label: str = "Pearson",
) -> None:
    if not top_gene_correlations:
        return

    ordered_k = sorted(top_gene_correlations)
    fig, axes = plt.subplots(1, len(ordered_k), figsize=(4.8 * len(ordered_k), 4.6), squeeze=False)
    for ax, k in zip(axes.flat, ordered_k, strict=False):
        sns.heatmap(
            top_gene_correlations[k],
            ax=ax,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=True,
            cbar=True,
        )
        ax.set_title(f"Top-{k} genes")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(f"{metric_label} correlation of top-pathway RNA weights across models", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_top_gene_weight_correlations(
    top_gene_correlations: dict[int, pd.DataFrame],
    *,
    summary_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for k, matrix in sorted(top_gene_correlations.items()):
        values: list[float] = []
        labels = matrix.index.tolist()
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                values.append(float(matrix.loc[left, right]))
        rows.append(
            {
                "top_k_genes": int(k),
                "comparison": "all_models",
                summary_column: float(np.mean(values)) if values else np.nan,
                "n_pairs": int(len(values)),
            }
        )

    return pd.DataFrame.from_records(rows)


def summarize_top_gene_weight_correlations_with_pathways(
    top_gene_correlations: dict[int, pd.DataFrame],
    top_gene_rankings: pd.DataFrame,
    *,
    summary_column: str,
) -> pd.DataFrame:
    pathway_by_step = (
        top_gene_rankings.sort_values(["step_index", "gene_rank"])
        .groupby("step_label", sort=False)["top_pathway"]
        .first()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []

    for k, matrix in sorted(top_gene_correlations.items()):
        labels = matrix.index.tolist()
        all_values: list[float] = []
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                all_values.append(float(matrix.loc[left, right]))
        rows.append(
            {
                "top_k_genes": int(k),
                "comparison": prettify_pathway_name("all_models"),
                summary_column: float(np.mean(all_values)) if all_values else np.nan,
                "n_pairs": int(len(all_values)),
            }
        )

        for pathway in dict.fromkeys(pathway_by_step.values()):
            group_steps = [step for step, step_pathway in pathway_by_step.items() if step_pathway == pathway]
            if len(group_steps) < 2:
                continue
            group_values: list[float] = []
            for i, left in enumerate(group_steps):
                for right in group_steps[i + 1 :]:
                    group_values.append(float(matrix.loc[left, right]))
            rows.append(
                {
                    "top_k_genes": int(k),
                    "comparison": prettify_pathway_name(pathway),
                    summary_column: float(np.mean(group_values)) if group_values else np.nan,
                    "n_pairs": int(len(group_values)),
                }
            )

    return pd.DataFrame.from_records(rows)


def save_top_gene_weight_correlation_summary_plot(
    summary_df: pd.DataFrame,
    *,
    path: Path,
    y_column: str,
    metric_label: str,
) -> None:
    if summary_df.empty:
        return

    plot_df = summary_df.loc[summary_df["top_k_genes"] >= 50, :].copy()
    if plot_df.empty:
        plot_df = summary_df.copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.lineplot(
        data=plot_df,
        x="top_k_genes",
        y=y_column,
        hue="comparison",
        marker="o",
        palette=CAT_PALETTE,
        ax=ax,
    )
    ax.set_title(f"Mean {metric_label} correlation of top-pathway RNA weights vs top-gene cutoff")
    ax.set_xlabel("Top genes per pathway")
    ax.set_ylabel(f"Mean {metric_label} correlation")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_gene_weight_correlation_summary_plot_with_pathways(
    summary_df: pd.DataFrame,
    *,
    path: Path,
    y_column: str,
    metric_label: str,
) -> None:
    if summary_df.empty:
        return

    plot_df = summary_df.loc[summary_df["top_k_genes"] >= 50, :].copy()
    if plot_df.empty:
        plot_df = summary_df.copy()
    plot_df["label"] = plot_df["comparison"]
    plot_df["label"] = plot_df["label"].replace({"All models": "Top informed gene program"})
    n_pairs_map = (
        plot_df.groupby("label", sort=False)["n_pairs"]
        .max()
        .to_dict()
    )
    preferred_order = [
        "Top informed gene program",
        prettify_pathway_name("MCA::HAN_MARGINAL ZONE B CELL"),
        prettify_pathway_name("MCA::HAN_PLASMA CELL"),
    ]
    order = [label for label in preferred_order if label in plot_df["label"].unique()]
    remaining = [label for label in plot_df["label"].unique() if label not in order]
    order.extend(remaining)
    palette = make_color_map(order, palette=CAT_PALETTE)
    style_dashes = {
        "Top informed gene program": "",
        "Marginal Zone B Cell [M]": (4, 2),
        "Plasma Cell [M]": (2, 2),
    }
    for label in remaining:
        style_dashes.setdefault(label, (3, 2))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sns.lineplot(
        data=plot_df,
        x="top_k_genes",
        y=y_column,
        hue="label",
        hue_order=order,
        style="label",
        style_order=order,
        markers=True,
        dashes=style_dashes,
        palette=palette,
        linewidth=1.8,
        ax=ax,
    )
    ax.set_xlabel("Top genes per pathway")
    ax.set_ylabel(f"Mean {metric_label} correlation")
    ax.set_ylim(0.0, 1.0)
    clean_ax(ax)
    legend = ax.legend_
    if legend is not None:
        for text in legend.get_texts():
            label = text.get_text()
            if label in n_pairs_map:
                text.set_text(f"{label} (n_pairs={n_pairs_map[label]})")
        legend.set_title("")
    savefig(fig, path)
    plt.close(fig)


def select_mca_pathways_to_plot(
    run_artifacts: list[RunArtifact],
    *,
    requested_pathways: list[str],
) -> list[str]:
    if requested_pathways:
        return sorted(canonicalize_mca_names(requested_pathways))

    reference_artifact = run_artifacts[0]
    return [
        pathway
        for pathway in top_k_pathways(reference_artifact.ranking, 10)
        if pathway.startswith(MCA_PREFIX)
    ]


def write_summary_text(summary_df: pd.DataFrame, *, path: Path, primary_source_name: str) -> None:
    median_recall_top3 = float(summary_df["mca_recall_top3"].median())
    median_recall_top5 = float(summary_df["mca_recall_top5"].median())
    median_recall_top10 = float(summary_df["mca_recall_top10"].median())
    min_recall_top10 = float(summary_df["mca_recall_top10"].min())
    max_recall_top10 = float(summary_df["mca_recall_top10"].max())
    median_jaccard = float(summary_df["jaccard_top10_vs_full"].median())
    median_spearman = float(summary_df["spearman_rank_vs_full"].median())
    last_row = summary_df.sort_values("n_gene_sets").iloc[0]

    lines = [
        f"MOFA-FLEX {primary_source_name.title()}+MCA prior-pruning sensitivity summary",
        f"Pruning steps evaluated: {len(summary_df)}",
        (
            "Across pruning steps, the retained-MCA recall was "
            f"{median_recall_top3:.3f} for top 3, "
            f"{median_recall_top5:.3f} for top 5, "
            f"and {median_recall_top10:.3f} for top 10."
        ),
        (
            "For top 10 specifically, retained-MCA recall ranged from "
            f"{min_recall_top10:.3f} to {max_recall_top10:.3f}."
        ),
        (
            "Top-10 pathway concordance against the full prior had median Jaccard overlap "
            f"{median_jaccard:.3f}, and the median rank Spearman correlation was {median_spearman:.3f}."
        ),
        (
            "The most aggressive retained prior still fitted in this sweep was "
            f"{last_row['step_label']} with {int(last_row['n_gene_sets'])} informed pathways."
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def write_outputs(
    *,
    run_artifacts: list[RunArtifact],
    summary_df: pd.DataFrame,
    pairwise_spearman: pd.DataFrame,
    pairwise_jaccard: dict[int, pd.DataFrame],
    out_dir: Path,
    selected_mca_rank_pathways: list[str],
    primary_source_name: str,
) -> None:
    plots_dir = out_dir / "plots"
    models_dir = out_dir / "models"
    rankings_dir = out_dir / "rankings"
    gene_sets_dir = out_dir / "gene_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir.mkdir(parents=True, exist_ok=True)
    gene_sets_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pairwise_spearman.to_csv(out_dir / "pairwise_spearman.csv")
    for artifact in run_artifacts:
        ranking_path = rankings_dir / f"{artifact.step.step_label}_pathway_ranks.csv"
        artifact.ranking.to_csv(ranking_path, index=False)

        gene_set_export = artifact.gene_set_stats.copy()
        gene_set_export["features"] = gene_set_export["features"].map(json.dumps)
        gene_set_export.to_csv(
            gene_sets_dir / f"{artifact.step.step_label}_retained_gene_sets.csv",
            index=False,
        )

    for k, matrix in pairwise_jaccard.items():
        matrix.to_csv(out_dir / f"pairwise_jaccard_top{k}.csv")
        save_pairwise_heatmap(
            matrix,
            title=f"Top-{k} pathway Jaccard overlap",
            path=plots_dir / f"pairwise_jaccard_top{k}.png",
        )
    save_pairwise_heatmap(
        pairwise_spearman,
        title="Pairwise pathway-rank Spearman correlation",
        path=plots_dir / "pairwise_spearman.png",
    )
    save_gene_set_count_plot(summary_df, path=plots_dir / "n_gene_sets_vs_n_mca_in_top10.png")
    save_mca_recall_plot(summary_df, path=plots_dir / "mca_recall_topk.png")
    save_reference_jaccard_plot(summary_df, k=10, path=plots_dir / "jaccard_top10_vs_full.png")

    selected_pathways = select_mca_pathways_to_plot(
        run_artifacts,
        requested_pathways=selected_mca_rank_pathways,
    )
    save_selected_pathway_rank_heatmap(
        summary_df,
        run_artifacts,
        selected_pathways=selected_pathways,
        path=plots_dir / "selected_mca_pathway_ranks.png",
    )
    save_top_pathway_variance_heatmap(
        summary_df,
        run_artifacts,
        top_k=10,
        path=plots_dir / "top10_pathway_variance_heatmap.png",
    )
    save_top_pathway_variance_heatmaps_by_model(
        summary_df,
        run_artifacts,
        top_k=10,
        path=plots_dir / "top10_pathway_variance_heatmaps_by_model.png",
    )
    save_top_pathway_loading_plots_by_model(
        summary_df,
        models_dir=models_dir,
        rankings_dir=rankings_dir,
        plots_dir=plots_dir,
        view="rna",
        n_features=20,
    )
    top_gene_k_values = list(range(20, 201, 10))
    top_gene_rankings = extract_top_pathway_gene_rankings_by_model(
        summary_df,
        models_dir=models_dir,
        rankings_dir=rankings_dir,
        view="rna",
        top_k_values=top_gene_k_values,
    )
    top_gene_rankings.to_csv(out_dir / "top_pathway_rna_top_genes.csv", index=False)
    top_gene_jaccard = build_top_gene_jaccard_matrices(
        top_gene_rankings,
        top_k_values=top_gene_k_values,
    )
    top_gene_correlations = build_top_gene_weight_correlation_matrices(
        top_gene_rankings,
        top_k_values=top_gene_k_values,
        weight_column="signed_weight",
        method="pearson",
    )
    top_gene_rank_correlations = build_top_gene_weight_correlation_matrices(
        top_gene_rankings,
        top_k_values=top_gene_k_values,
        weight_column="signed_weight",
        method="spearman",
    )
    top_gene_correlation_summary = summarize_top_gene_weight_correlations(
        top_gene_correlations,
        summary_column="mean_pearson",
    )
    top_gene_correlation_summary_with_pathways = summarize_top_gene_weight_correlations_with_pathways(
        top_gene_correlations,
        top_gene_rankings,
        summary_column="mean_pearson",
    )
    top_gene_rank_correlation_summary = summarize_top_gene_weight_correlations(
        top_gene_rank_correlations,
        summary_column="mean_spearman",
    )
    for k, matrix in top_gene_jaccard.items():
        matrix.to_csv(out_dir / f"pairwise_top_pathway_gene_jaccard_top{k}.csv")
    for k, matrix in top_gene_correlations.items():
        matrix.to_csv(out_dir / f"pairwise_top_pathway_gene_pearson_top{k}.csv")
    for k, matrix in top_gene_rank_correlations.items():
        matrix.to_csv(out_dir / f"pairwise_top_pathway_gene_spearman_top{k}.csv")
    top_gene_correlation_summary.to_csv(
        out_dir / "top_pathway_gene_pearson_summary.csv",
        index=False,
    )
    top_gene_correlation_summary_with_pathways.to_csv(
        out_dir / "top_pathway_gene_pearson_summary_with_pathways.csv",
        index=False,
    )
    top_gene_rank_correlation_summary.to_csv(
        out_dir / "top_pathway_gene_spearman_summary.csv",
        index=False,
    )
    save_top_gene_jaccard_heatmaps(
        top_gene_jaccard,
        path=plots_dir / "top_pathway_gene_jaccard_heatmaps.png",
    )
    save_top_gene_weight_correlation_heatmaps(
        top_gene_correlations,
        path=plots_dir / "top_pathway_gene_pearson_heatmaps.png",
        metric_label="Pearson",
    )
    save_top_gene_weight_correlation_heatmaps(
        top_gene_rank_correlations,
        path=plots_dir / "top_pathway_gene_spearman_heatmaps.png",
        metric_label="Spearman",
    )
    save_top_gene_weight_correlation_summary_plot(
        top_gene_correlation_summary,
        path=plots_dir / "top_pathway_gene_pearson_summary.png",
        y_column="mean_pearson",
        metric_label="Pearson",
    )
    save_top_gene_weight_correlation_summary_plot_with_pathways(
        top_gene_correlation_summary_with_pathways,
        path=plots_dir / "top_pathway_gene_pearson_summary_with_pathways.png",
        y_column="mean_pearson",
        metric_label="Pearson",
    )
    save_top_gene_weight_correlation_summary_plot(
        top_gene_rank_correlation_summary,
        path=plots_dir / "top_pathway_gene_spearman_summary.png",
        y_column="mean_spearman",
        metric_label="Spearman",
    )
    focus_top_k = top_gene_k_values[-1]
    save_top_pathway_topk_correlation_focus_heatmaps(
        pearson_matrix=top_gene_correlations[focus_top_k],
        spearman_matrix=top_gene_rank_correlations[focus_top_k],
        top_k_genes=focus_top_k,
        path=plots_dir / f"top_pathway_top{focus_top_k}_correlation_focus.png",
    )
    pd.concat(
        [
            summarize_pairwise_matrix(
                top_gene_correlations[focus_top_k],
                metric_name="pearson",
                top_k_genes=focus_top_k,
            ),
            summarize_pairwise_matrix(
                top_gene_rank_correlations[focus_top_k],
                metric_name="spearman",
                top_k_genes=focus_top_k,
            ),
        ],
        ignore_index=True,
    ).to_csv(out_dir / f"top_pathway_top{focus_top_k}_correlation_summary.csv", index=False)

    write_summary_text(summary_df, path=out_dir / "robustness_summary.txt", primary_source_name=primary_source_name)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    import mofaflex as mfl

    set_house_style()
    args = resolve_collection_preset(args)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = sorted(set([10, *args.top_k_values]))

    mdata = load_sln208_mdata(args.data_path, modalities=args.modalities)
    base_gene_set_stats = load_sorted_gene_set_stats(
        var_names=mdata.mod["rna"].var_names,
        mca_gmt_url=args.mca_gmt_url,
        msigdb_category=args.msigdb_category,
        msigdb_dbver=args.msigdb_dbver,
        gene_set_min_fraction=args.gene_set_min_fraction,
        gene_set_min_count=args.gene_set_min_count,
        gene_set_max_count=args.gene_set_max_count,
        gene_set_similarity_threshold=args.gene_set_similarity_threshold,
        expected_primary_count=args.expected_primary_count,
        expected_mca_count=args.expected_mca_count,
    )
    primary_source_name = infer_primary_source_name(args.msigdb_category)
    pruning_steps = build_pruning_steps(
        len(base_gene_set_stats),
        prune_step_size=args.prune_step_size,
        prune_direction=args.prune_direction,
        min_informed_factors=args.min_informed_factors,
        min_total_factors=args.min_total_factors,
        n_uninformed_factors=args.n_uninformed_factors,
    )

    metadata = {
        "data_path": str(args.data_path),
        "collection": args.collection,
        "modalities": list(mdata.mod.keys()),
        "n_obs": int(mdata.n_obs),
        "n_vars_by_modality": {modality: int(adata.n_vars) for modality, adata in mdata.mod.items()},
        "primary_source_name": primary_source_name,
        "initial_n_gene_sets": int(len(base_gene_set_stats)),
        "initial_n_primary_gene_sets": int((base_gene_set_stats["source"] == primary_source_name).sum()),
        "initial_n_mca_gene_sets": int((base_gene_set_stats["source"] == "mca").sum()),
        "prune_direction": args.prune_direction,
        "pruning_steps": [asdict(step) for step in pruning_steps],
        "top_k_values": top_k_values,
        "likelihoods": resolve_likelihoods_for_mdata(mdata),
        "weight_prior": DEFAULT_WEIGHT_PRIOR,
        "nonnegative_weights": DEFAULT_NONNEGATIVE_WEIGHTS,
        "nonnegative_factors": DEFAULT_NONNEGATIVE_FACTORS,
        "init_factors": DEFAULT_INIT_FACTORS,
        "init_scale": DEFAULT_INIT_SCALE,
        "lr": DEFAULT_LR,
        "n_particles": DEFAULT_N_PARTICLES,
    }
    for modality, adata in mdata.mod.items():
        metadata[f"n_vars_{modality}"] = int(adata.n_vars)
    (out_dir / "resolved_run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    run_artifacts: list[RunArtifact] = []

    for step in pruning_steps:
        current_gene_set_stats = retained_gene_set_stats_for_step(base_gene_set_stats, step)
        removed_phrase = (
            f"dropping {step.n_removed_smallest} smallest sets"
            if step.pruning_direction == "smallest"
            else f"dropping {step.n_removed_largest} largest sets"
        )
        print(
            "Fitting "
            f"{step.step_label} with {len(current_gene_set_stats)} informed pathways "
            f"after {removed_phrase}; "
            f"saving model to {models_dir / f'{step.step_label}.h5'}..."
        )
        run_mdata = attach_gene_set_mask(mdata, current_gene_set_stats)
        save_path = models_dir / f"{step.step_label}.h5"
        model = fit_model(
            mdata=run_mdata,
            n_uninformed_factors=args.n_uninformed_factors,
            seed=args.seed,
            save_path=save_path,
        )
        ranking = extract_informed_pathway_ranking(model, current_gene_set_stats)
        run_artifacts.append(
            RunArtifact(
                step=step,
                n_uninformed_factors=args.n_uninformed_factors,
                gene_set_stats=current_gene_set_stats,
                ranking=ranking,
            )
        )
        del model
        del run_mdata
        gc.collect()

    reference_ranking = run_artifacts[0].ranking
    expected_mca_pathways = canonicalize_mca_names(args.expected_mca_pathways)
    summary_rows = [
        summarize_run(
            artifact,
            reference_ranking=reference_ranking,
            expected_mca_pathways=expected_mca_pathways,
            top_k_values=top_k_values,
        )
        for artifact in run_artifacts
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("n_gene_sets", ascending=False).reset_index(drop=True)
    pairwise_jaccard = {k: build_pairwise_jaccard_matrix(run_artifacts, k=k) for k in top_k_values}
    pairwise_spearman = build_pairwise_spearman_matrix(run_artifacts)

    write_outputs(
        run_artifacts=run_artifacts,
        summary_df=summary_df,
        pairwise_spearman=pairwise_spearman,
        pairwise_jaccard=pairwise_jaccard,
        out_dir=out_dir,
        selected_mca_rank_pathways=args.selected_mca_rank_pathways,
        primary_source_name=primary_source_name,
    )

    return {
        "n_runs": int(len(run_artifacts)),
        "initial_n_gene_sets": int(len(base_gene_set_stats)),
        "final_n_gene_sets": int(run_artifacts[-1].step.n_gene_sets),
        "out_dir": str(out_dir),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_analysis(args)
    print(f"Finished MCA prior-pruning sensitivity analysis in {result['out_dir']}")
    print(f"n_runs={result['n_runs']}")
    print(f"initial_n_gene_sets={result['initial_n_gene_sets']}")
    print(f"final_n_gene_sets={result['final_n_gene_sets']}")


if __name__ == "__main__":
    main()
