"""Typed config loading with fail-closed validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.constants import (
    CORRUPTION_OPERATORS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    MIN_SEEDS_PER_CELL,
)


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "cifar100"
    root: str = "data"
    image_size: int = 32
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 2
    subset_fit: int | None = None


@dataclass(frozen=True)
class QualityConfig:
    """How data quality is manipulated.

    `operator` selects the corruption mechanism and `quality` its rate. Holding
    rate fixed while varying operator is the artifact test, so both are explicit
    and never inferred.
    """

    operator: str = "label_flip"
    quality: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet20"
    num_classes: int = 100
    width: int = 16


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer: str = "sgd"
    momentum: float = 0.9
    seed: int = 0
    checkpoint_dir: str = "outputs/runs"


@dataclass(frozen=True)
class StudyConfig:
    """A study is a grid of (model size, dataset size, quality, seed) runs."""

    name: str = "study"
    n_val: int = 2000
    grid_sizes: tuple[int, ...] = (15, 30, 60, 120)
    min_seeds: int = MIN_SEEDS_PER_CELL
    operators: tuple[str, ...] = CORRUPTION_OPERATORS


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    study: StudyConfig = field(default_factory=StudyConfig)


_ALLOWED_DATASETS = {"cifar100", "cifar10", "tinyimagenet", "fake"}
_ALLOWED_MODELS = {"resnet20", "resnet32"}
_ALLOWED_OPTIMIZERS = {"sgd", "adam", "adamw"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_config(path: str | Path) -> RunConfig:
    """Load YAML config with strict validation. Raises ValueError on any violation."""
    p = Path(path)
    _require(p.is_file(), f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = RunConfig(
        data=DataConfig(**raw.get("data", {})),
        quality=QualityConfig(**raw.get("quality", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        study=StudyConfig(**raw.get("study", {})),
    )
    _require(cfg.data.dataset in _ALLOWED_DATASETS, f"bad data.dataset: {cfg.data.dataset}")
    _require(cfg.model.name in _ALLOWED_MODELS, f"bad model.name: {cfg.model.name}")
    _require(cfg.train.optimizer in _ALLOWED_OPTIMIZERS, f"bad optimizer: {cfg.train.optimizer}")
    _require(cfg.quality.operator in CORRUPTION_OPERATORS,
             f"bad quality.operator: {cfg.quality.operator}; allowed {CORRUPTION_OPERATORS}")
    _require(0.0 <= cfg.quality.quality <= 1.0, "quality.quality must be in [0, 1]")
    _require(cfg.data.batch_size >= 1, "batch_size must be >= 1")
    _require(cfg.train.epochs >= 1, "epochs must be >= 1")
    _require(cfg.train.lr > 0, "lr must be > 0")
    _require(cfg.train.weight_decay >= 0, "weight_decay must be >= 0")
    _require(cfg.data.subset_fit is None or cfg.data.subset_fit >= 1, "subset_fit must be >= 1")
    _require(cfg.train.seed >= 0, "train.seed must be >= 0")
    _require(cfg.model.width >= 1, "model.width must be >= 1")
    _require(cfg.study.grid_sizes, "study.grid_sizes must be non-empty")
    _require(all(g >= 3 for g in cfg.study.grid_sizes), "grid sizes must be >= 3")
    return cfg