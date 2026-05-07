from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import tomllib


@dataclass
class ExperimentConfig:
    seed: int = 0
    output_dir: str = "artifacts/synthetic_baseline"
    device: str = "auto"
    save_model: bool = True
    save_mudata: bool = True


@dataclass
class SyntheticConfig:
    n_samples: int = 200
    n_features: list[int] = field(default_factory=lambda: [400, 400, 400, 400])
    likelihoods: list[str] = field(default_factory=lambda: ["Normal", "Normal", "Normal", "Normal"])
    n_fully_shared_factors: int = 2
    n_partially_shared_factors: int = 15
    n_private_factors: int = 3
    factor_size_dist: str = "Uniform"
    all_combinations: bool = True
    noise_fraction: float = 0.5
    informed_view_indices: list[int] = field(default_factory=lambda: [0])
    nmf: bool | list[bool] = False
    normalize_generated: bool = True
    normalize_with_std: bool = False


@dataclass
class ModelConfig:
    n_factors: int = 0
    weight_prior: str = "Horseshoe"
    factor_prior: str = "Normal"
    nonnegative_weights: bool = False
    nonnegative_factors: bool = False
    annotation_confidence: float = 0.97
    init_factors: float | str = 0.0
    init_scale: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 0
    max_epochs: int = 10_000
    n_particles: int = 5
    lr: float = 0.005
    early_stopper_patience: int = 100
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class EvaluationConfig:
    sparse_type: str = "mix"
    factor_activity_indices: list[int] = field(default_factory=lambda: [0, 4, 8, 12])
    dpi: int = 200


@dataclass
class PipelineConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        n_views = len(self.synthetic.n_features)
        if n_views == 0:
            raise ValueError("`synthetic.n_features` must contain at least one view.")
        if len(self.synthetic.likelihoods) != n_views:
            raise ValueError("`synthetic.likelihoods` must match the number of views.")
        if isinstance(self.synthetic.nmf, list) and len(self.synthetic.nmf) != n_views:
            raise ValueError("`synthetic.nmf` must be a bool or a list matching the number of views.")
        if not 0.0 <= self.synthetic.noise_fraction <= 1.0:
            raise ValueError("`synthetic.noise_fraction` must be between 0 and 1.")
        if not 0.0 <= self.model.annotation_confidence <= 1.0:
            raise ValueError("`model.annotation_confidence` must be between 0 and 1.")
        invalid_views = [idx for idx in self.synthetic.informed_view_indices if idx < 0 or idx >= n_views]
        if invalid_views:
            raise ValueError(f"Invalid informed view indices: {invalid_views}")
        invalid_factor_indices = [idx for idx in self.evaluation.factor_activity_indices if idx < 0]
        if invalid_factor_indices:
            raise ValueError(f"Factor activity indices must be non-negative: {invalid_factor_indices}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_section(instance: Any, overrides: dict[str, Any] | None) -> Any:
    if not overrides:
        return instance
    for key, value in overrides.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key `{key}` in `{type(instance).__name__}`.")
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text())
    config = PipelineConfig()
    config.experiment = _merge_section(config.experiment, raw.get("experiment"))
    config.synthetic = _merge_section(config.synthetic, raw.get("synthetic"))
    config.model = _merge_section(config.model, raw.get("model"))
    config.training = _merge_section(config.training, raw.get("training"))
    config.evaluation = _merge_section(config.evaluation, raw.get("evaluation"))
    config.validate()
    return config
