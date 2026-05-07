# MOFAFLEX Benchmark

This repository is a lightweight benchmark harness for comparing multi-omic single-cell integration methods, with a primary focus on MOFAFLEX and a few strong baselines on RNA+protein and RNA+ATAC datasets.

The codebase is organized around reproducible experiment scripts rather than a large reusable package. In practice, the workflow is:

1. Prepare a dataset into the format expected by each method.
2. Run one or more model scripts from `runs/` using YAML configs from `configs/`.
3. Save each method's latent representation under `outputs/<run_name>/`.
4. Evaluate and visualize those latents with the scripts in `runs/`.

## Repository Layout

```text
configs/         Experiment definitions
runs/            Training, preparation, evaluation, and plotting entry points
src/data/        Data loaders and preprocessing helpers
src/models/      Thin wrappers around each benchmarked method
src/metrics/     Clustering and scIB evaluation helpers
notebooks/       Exploratory analysis, especially informed MOFAFLEX work
outputs/         Saved benchmark results and plots
```

## Main Benchmark Scenarios

The repository currently has a few concrete benchmark setups wired in:

- `sln_208`: main CITE-seq benchmark on mouse spleen/lymph node data
- `sln_208_d1`: single-batch follow-up benchmark on `SLN208-D1`
- `spleen_111`: smaller RNA+protein benchmark
- `pbmc3k`: 10x multiome RNA+ATAC benchmark
- horizontal integration examples for Multigrate

The `sln_208` workflow is the most complete and is the best place to start.

## Supported Methods

Implemented runners:

- MOFAFLEX
- totalVI
- Multigrate
- MultiVI
- UINMF
- PCA baseline
- NMF baseline

Scaffolded but not fully implemented yet:

- scGPT
- scFoundation

The `scGPT` and `scFoundation` runners currently raise `NotImplementedError` until model-specific inference wiring is added.

## How The Benchmark Works

Each training script in `runs/` follows the same high-level pattern:

1. Load a YAML config from `configs/`.
2. Read input data as `h5ad`, `h5mu`, or a 10x multiome directory.
3. Preprocess the data into the format expected by the target model.
4. Train the model and export its latent space.

Each run writes an output directory containing some combination of:

- `latent.npy`: latent representation used for downstream evaluation
- `runinfo.json`: metadata about the run
- `cell_ids.txt` or `cell_ids.npy`: cell identity alignment for evaluation
- `latent_groups.npz`: MOFAFLEX group-wise factors

Evaluation scripts align each latent matrix back to the original labels, using saved `cell_ids` when necessary.

## Important Entry Points

### Data preparation

- `runs/prepare_sln208_mofaflex.py`
- `runs/prepare_sln208_single_batch.py`
- `runs/prepare_sln208_single_batch_h5mu.py`

These create the derived benchmark datasets used by the downstream training scripts.

### Training

- `runs/run_mofaflex.py`
- `runs/run_totalvi.py`
- `runs/run_multigrate.py`
- `runs/run_multigrate_horizontal.py`
- `runs/run_multivi.py`
- `runs/run_uinmf.py`
- `runs/run_baselines.py`

### Evaluation and plots

- `runs/evaluate_all.py`
- `runs/evaluate_latent.py`
- `runs/evaluate_scib_benchmarker.py`
- `runs/evaluate_scgraph.py`
- `runs/merge_metric_tables.py`
- `runs/plot_latent_umap.py`
- `runs/plot_nmi_results.py`
- `runs/plot_scib_results.py`

### End-to-end orchestration

- `runs/run_sln208_train_all.sh`
- `runs/run_sln208_evaluate_all.sh`
- `runs/run_sln208_plot_all.sh`
- `runs/run_sln208_scib_benchmarker.sh`
- `runs/run_sln208_scgraph.sh`
- `runs/run_sln208_metric_comparison.sh`
- `runs/run_sln208_evaluate_clusters_legacy.sh`
- `runs/run_sln208_plot_umaps_legacy.sh`
- `runs/run_sln208_d1_train_all.sh`
- `runs/run_sln208_d1_evaluate_all.sh`
- `runs/run_sln208_d1_plot_all.sh`
- `runs/run_sln208_d1_scib_benchmarker.sh`
- `runs/run_sln208_d1_evaluate_clusters_legacy.sh`
- `runs/run_sln208_d1_plot_umaps_legacy.sh`

These are the closest thing this repo has to a canonical benchmark workflow.

## Data Loading Model

The loading layer is split across:

- `src/data/loaders.py`
- `src/data/load_multiome.py`

Supported input styles:

- `h5ad` for RNA+protein workflows
- `h5mu` for MOFAFLEX-style multi-view workflows
- 10x multiome directory layouts for RNA+ATAC workflows

There are method-specific preprocessing choices built into the loaders:

- totalVI expects a raw counts layer
- Multigrate applies modality-specific normalization before model setup
- MOFAFLEX can apply a paper-style normalization path for 10x data
- horizontal RNA+protein and RNA+ATAC loaders prepare per-batch views

## Informed MOFAFLEX

One of the more specialized parts of the repository is the informed MOFAFLEX workflow.

`runs/run_mofaflex.py` can attach gene-set annotations to the RNA modality via `varm["gene_set_mask"]`. This is used to guide factor learning with prior biological knowledge. The exploratory notebook for that workflow is:

- `notebooks/mouse_citeseq_informed.ipynb`

That notebook appears to be the prototype for the scripted `sln_208` informed benchmark.

## Evaluation

The default `sln_208` main-results workflow is now the merged metric comparison produced by:

- `runs/run_sln208_evaluate_all.sh`

This alias calls `runs/run_sln208_metric_comparison.sh`, which evaluates:

- scIB `Bio conservation`
- Islander/scGraph `Corr-Weighted`
- a merged comparison table aligning both rankings

The primary full-benchmark result files are:

- `outputs/sln_208_metric_comparison.csv`
- `outputs/sln_208_metric_comparison.json`
- `outputs/sln_208_scib_bio_benchmarker.csv`
- `outputs/sln_208_scgraph.csv`

The primary full-benchmark result plots are:

- `outputs/sln_208_scib_bio_conservation.png`
- `outputs/sln_208_scgraph_corr_weighted.png`

Clustering-based comparison is still available, but is now treated as a legacy / secondary view through:

- `runs/run_sln208_evaluate_clusters_legacy.sh`
- `runs/run_sln208_plot_umaps_legacy.sh`

The legacy clustering evaluation script `runs/evaluate_all.py` computes metrics such as:

- NMI
- ARI
- silhouette metrics
- homogeneity/completeness/v-measure
- Calinski-Harabasz
- Davies-Bouldin

Optional scIB-style metrics are implemented in `src/metrics/scib.py` and exposed through `runs/evaluate_all.py`.

For broader embedding benchmarking, `runs/evaluate_scib_benchmarker.py` uses the `scib_metrics.benchmark.Benchmarker` API when available.

The repository also now supports Islander-style `scGraph` evaluation through `runs/evaluate_scgraph.py`, using the official `scgraph-eval` package. This is most meaningful on multi-batch datasets such as `sln_208`, where the metric compares embedding-derived cell-type geometry against a cross-batch consensus.

For the smaller single-batch `sln_208_d1` workflow, the default main-results view is now the scIB biological-conservation benchmark produced by:

- `runs/run_sln208_d1_evaluate_all.sh`

This alias calls `runs/run_sln208_d1_scib_benchmarker.sh`, which writes:

- `outputs/sln_208_d1_scib_benchmarker.csv`
- `outputs/sln_208_d1_scib_benchmarker.json`
- `outputs/sln_208_d1_scib_bio_conservation.png`

Clustering tables and UMAPs are still available for `sln_208_d1`, but are now treated as a legacy / secondary view through:

- `runs/run_sln208_d1_evaluate_clusters_legacy.sh`
- `runs/run_sln208_d1_plot_umaps_legacy.sh`

## Current Results Snapshot

Saved outputs under `outputs/` include benchmark tables, latent arrays, and UMAP plots from previous runs. These are useful as references, but they should not always be treated as fully current or canonical.

For example, some saved comparison files include only a subset of methods mentioned by the shell scripts, which suggests the repository contains historical outputs from different benchmark passes.

## Environments

This repo uses different environments for different methods. The shell scripts assume names like:

- `mfl_bench`
- `scvi_env`
- `multigrate_env`
- `rliger_env`
- `scgpt_env`
- `scfoundation_env`

There is also a minimal Conda spec in `environment_multigrate.yml` for the Multigrate stack.

In practice, the benchmark depends on method-specific environments more than on a single universal environment.

For the `scGPT` path, a reproducible bootstrap script is now available at:

- `runs/setup_scgpt_env.sh`

## Quick Start

### 1. Prepare the main dataset

```bash
PYTHONPATH=. python runs/prepare_sln208_mofaflex.py
```

### 2. Train the main `sln_208` benchmark

```bash
bash runs/run_sln208_train_all.sh
```

### 3. Compute Main Results

```bash
conda activate scvi_env
bash runs/run_sln208_evaluate_all.sh
```

### 4. Refresh Main Result Plots

```bash
conda activate scvi_env
bash runs/run_sln208_plot_all.sh
```

### 5. Legacy clustering / UMAP view

```bash
conda activate scvi_env
bash runs/run_sln208_evaluate_clusters_legacy.sh
bash runs/run_sln208_plot_umaps_legacy.sh
```

### 6. `sln_208_d1` default main results

```bash
conda activate scvi_env
bash runs/run_sln208_d1_evaluate_all.sh
bash runs/run_sln208_d1_plot_all.sh
```

### 7. `sln_208_d1` legacy clustering / UMAP view

```bash
conda activate scvi_env
bash runs/run_sln208_d1_evaluate_clusters_legacy.sh
bash runs/run_sln208_d1_plot_umaps_legacy.sh
```

## Recommended Places To Extend

If you want to continue the benchmark, the safest extension points are:

- add a new YAML config in `configs/`
- add a new runner in `runs/`
- add a method wrapper in `src/models/`
- add a loader or preprocessing path in `src/data/`
- add a new metric or aggregation step in `src/metrics/`

For most benchmark additions, the lowest-friction path is:

1. add a config
2. add or adapt a run script
3. make sure it writes `latent.npy` and `runinfo.json`
4. add `cell_ids.txt` if row order may differ from the source dataset
5. include the run in `evaluate_all.py` calls

## Known Gaps and Caveats

- There is no formal test suite yet.
- Several scripts use absolute local filesystem paths.
- Saved results in `outputs/` are partly historical and may not match the latest scripts.
- `scGPT` and `scFoundation` are placeholders, not finished integrations.
- Reproducibility currently depends heavily on local environment management.

## Benchmark Context

The repository also includes `datalab-output-s41592-024-02429-w.pdf.md`, which appears to be notes or extracted text from the 2024 Nature Methods benchmark paper on single-cell multi-omics prediction and integration. It provides useful background context for the benchmarking direction of this repo.

## Practical Advice

If you are new to the codebase, start with the `sln_208_d1` workflow. It is smaller than the full `sln_208` benchmark, still representative of the repository design, and easier to iterate on when changing loaders, model wrappers, or evaluation code.
