"""Grid construction.

A scaling grid is a set of (N, D, Q, seed) runs. Two properties make the quality
exponent identifiable at all:

1. **Nesting.** Every dataset-size point is a subset of the next larger one, so
   dataset size is not confounded with which examples happen to be present.
2. **Crossing.** Model size, dataset size and quality each vary independently
   across the grid, so the exponent is not aliased onto any single axis.

Both are enforced here rather than left to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import MIN_SEEDS_PER_CELL, REQUIRE_NESTED
from src.data.datasets import nested_subset


@dataclass(frozen=True)
class GridPoint:
    """One training run in the grid."""

    n_params: int
    n_examples: int
    quality: float
    operator: str
    seed: int
    width: int
    blocks: int

    def as_dict(self) -> dict:
        return {"n_params": self.n_params, "n_examples": self.n_examples,
                "quality": self.quality, "operator": self.operator, "seed": self.seed,
                "width": self.width, "blocks": self.blocks}


def param_count_for(width: int, blocks: int, num_classes: int) -> int:
    """Analytic parameter count for the CIFAR ResNet, used to label the size axis."""
    if width < 1 or blocks < 1:
        raise ValueError("width and blocks must be >= 1")
    c = width
    conv1 = 3 * c * 9 + 2 * c
    def stage(in_planes: int, planes: int) -> int:
        total = planes * in_planes * 9 + 2 * planes          # block 1 conv1 + bn
        total += planes * planes * 9 + 2 * planes             # block 1 conv2 + bn
        total += planes * in_planes + 2 * planes             # shortcut
        for _ in range(blocks - 1):
            total += planes * planes * 9 + 2 * planes         # conv1 + bn
            total += planes * planes * 9 + 2 * planes         # conv2 + bn
        return total
    total = conv1 + stage(c, c) + stage(c, 2 * c) + stage(2 * c, 4 * c)
    total += 4 * c * num_classes + num_classes
    return int(total)


def nested_dataset_sizes(total: int, n_sizes: int) -> list[int]:
    """Increasing dataset sizes, each a subset of the next."""
    if n_sizes < 2:
        raise ValueError("n_sizes must be >= 2")
    if total < n_sizes:
        raise ValueError("total corpus smaller than the requested number of sizes")
    sizes = np.unique(np.round(np.geomspace(total // 20, total, n_sizes)).astype(int))
    if sizes.size < n_sizes:
        sizes = np.linspace(max(n_sizes, total // 20), total, n_sizes).astype(int)
    return sorted(int(s) for s in sizes)


def build_grid(widths: tuple[int, ...], blocks: int, num_classes: int, dataset_sizes,
               qualities: tuple[float, ...], operators: tuple[str, ...],
               seeds: tuple[int, ...]) -> list[GridPoint]:
    """Cartesian grid with nesting enforced by construction.

    Sizes, qualities and operators must all vary for the exponent to be
    identifiable, so a degenerate axis is rejected rather than silently accepted.
    """
    if len(widths) < 2:
        raise ValueError("need at least two model sizes for a scaling grid")
    if len(dataset_sizes) < 2:
        raise ValueError("need at least two dataset sizes for a scaling grid")
    if len(qualities) < 2:
        raise ValueError("need at least two quality levels for a quality exponent")
    if len(operators) < 1:
        raise ValueError("need at least one operator")
    if len(seeds) < MIN_SEEDS_PER_CELL:
        raise ValueError(
            f"this project measures variance; need at least {MIN_SEEDS_PER_CELL} seeds, got {len(seeds)}")
    if REQUIRE_NESTED and any(b <= a for a, b in zip(dataset_sizes, dataset_sizes[1:])):
        raise ValueError(f"dataset sizes must be strictly increasing and nested: {dataset_sizes}")

    points: list[GridPoint] = []
    for width in widths:
        n_params = param_count_for(width, blocks, num_classes)
        for n_examples in dataset_sizes:
            for quality in qualities:
                for operator in operators:
                    for seed in seeds:
                        points.append(GridPoint(n_params=n_params, n_examples=n_examples,
                                                quality=float(quality), operator=operator,
                                                seed=int(seed), width=width, blocks=blocks))
    return points


def build_nested_arrays(base_images: np.ndarray, base_labels: np.ndarray,
                         dataset_sizes: list[int], total: int, seed: int):
    """Materialise nested corpora so larger sizes contain smaller ones.

    Returns a dict mapping size -> (images, labels). Every subset is drawn from
    the same permutation, which is what nesting requires.
    """
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for size in dataset_sizes:
        fraction = min(1.0, size / total)
        out[size] = nested_subset(base_images, base_labels, fraction, seed)
    return out