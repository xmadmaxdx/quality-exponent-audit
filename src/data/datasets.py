"""Datasets for the quality-exponent audit.

CIFAR-100 is the default because it is small enough for a Colab T4, large
enough that a quality gradient is meaningful, and public. Tiny-ImageNet is the
scale-up check. Nothing here is medical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
import torchvision.datasets as tvd

from src.constants import EVAL_SPLIT_SEED, MIN_LOGIT

_TORCHVISION = {
    "cifar10": ("CIFAR10", 32, 10),
    "cifar100": ("CIFAR100", 32, 100),
    "tinyimagenet": ("TinyImageNet", 64, 200),
    # "fake" is a synthetic stand-in with CIFAR-100 geometry. It is registered
    # here rather than special-cased at each call site so that the smoke config
    # exercises the same code paths as a real dataset, with no downloads.
    "fake": ("Fake", 32, 100),
}

_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    "fake": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def image_size_for(dataset: str) -> int:
    if dataset not in _TORCHVISION:
        raise ValueError(f"unknown dataset: {dataset}")
    return _TORCHVISION[dataset][1]


def num_classes_for(dataset: str) -> int:
    if dataset not in _TORCHVISION:
        raise ValueError(f"unknown dataset: {dataset}")
    return _TORCHVISION[dataset][2]


def _transforms(dataset: str, train: bool):
    mean, std = _STATS[dataset]
    size = image_size_for(dataset)
    if train:
        return T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


class ArrayDataset(Dataset):
    """In-memory dataset. Quality operators act on the arrays before wrapping."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError(f"images {images.shape[0]} and labels {labels.shape[0]} disagree")
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        from PIL import Image

        array = self.images[index]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        image = Image.fromarray(array)
        return self.transform(image), int(self.labels[index])


class FakeImages(Dataset):
    """Deterministic synthetic set for offline tests. No downloads."""

    def __init__(self, n: int, num_classes: int, image_size: int, transform, seed: int = 0) -> None:
        self.n = int(n)
        self.transform = transform
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, size=(self.n, image_size, image_size, 3), dtype=np.uint8)
        self.labels = rng.integers(0, num_classes, size=(self.n,))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def load_arrays(cfg, download: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return the full training arrays, uncorrupted.

    Quality is applied afterwards by the caller so that every run in a grid
    derives its corpus from the same source arrays, which is what makes the
    grid nested and the exponent identifiable.
    """
    name = cfg.data.dataset
    if name == "fake":
        n = cfg.data.subset_fit or 1000
        size, classes = _TORCHVISION[name][1], num_classes_for(name)
        rng = np.random.default_rng(0)
        return (rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8),
                rng.integers(0, classes, size=(n,)))
    cls_name, size, _ = _TORCHVISION[name]
    root = Path(cfg.data.root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = getattr(tvd, cls_name)(root=str(root), train=True, download=download)
    images = np.asarray(dataset.data)
    labels = np.asarray(dataset.targets)
    if name == "tinyimagenet":
        # Tiny-ImageNet ships as a folder structure, not arrays; materialise it
        # once into an array so quality operators can act on it uniformly.
        if images.ndim == 4:
            images = images
        else:
            raise ValueError("tinyimagenet must be materialised by scripts/download_data.py")
    return images, labels


def nested_subset(images: np.ndarray, labels: np.ndarray, fraction: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Take a random `fraction` of the corpus, stratified by class.

    Nesting is what makes the exponent identifiable: if dataset size and quality
    varied independently, a fitted exponent would confound the two. Every grid
    point in a study must therefore be drawn from the same random permutation.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    keep_parts: list[np.ndarray] = []
    for class_id in np.unique(labels):
        member = np.flatnonzero(labels == class_id)
        member = member[rng.permutation(member.size)]
        take = max(1, int(round(member.size * fraction)))
        keep_parts.append(member[:take])
    keep = np.sort(np.concatenate(keep_parts))
    return images[keep], labels[keep]


def split_indices(labels: np.ndarray, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified disjoint fit/eval indices. The single source of truth for the split.

    Returning indices rather than materialised copies keeps the split verifiable:
    disjointness is a claim about indices, and it cannot be checked on pixel
    arrays.

    The eval split is drawn from a fixed permutation, so it is identical across
    every run in a study. That matters twice over:

    1. Eval loss is the estimand fitted by the scaling law, so the split must be
       held constant or the fitted exponent partly measures the split.
    2. The split must be disjoint from the fit pool. Evaluating on training
       examples inflates the loss, and under label corruption it is worse than
       merely optimistic: the eval labels are the clean originals for examples
       whose training labels were deliberately damaged, so leakage manufactures
       an apparent quality signal out of nothing.

    Returns (fit_indices, eval_indices).
    """
    labels = np.asarray(labels)
    total = labels.shape[0]
    if n_val < 1:
        raise ValueError("n_val must be >= 1")
    if n_val >= total:
        raise ValueError(f"n_val={n_val} must be smaller than the corpus ({total})")
    rng = np.random.default_rng(seed)
    class_ids = np.unique(labels)
    if class_ids.size < 2:
        raise ValueError("need at least two classes to build a stratified split")
    eval_parts, fit_parts = [], []
    for class_id in class_ids:
        member = np.flatnonzero(labels == class_id)
        member = member[rng.permutation(member.size)]
        take = max(1, int(round(member.size * n_val / total)))
        take = min(take, member.size - 1)  # always leave >= 1 fit example
        eval_parts.append(member[:take])
        fit_parts.append(member[take:])
    fit_idx = np.sort(np.concatenate(fit_parts))
    eval_idx = np.sort(np.concatenate(eval_parts))
    if fit_idx.size == 0 or eval_idx.size == 0:
        raise ValueError("stratified split produced an empty side")
    overlap = np.intersect1d(fit_idx, eval_idx)
    if overlap.size:
        raise RuntimeError(f"fit/eval split leaked {overlap.size} overlapping indices")
    return fit_idx, eval_idx


def split_eval_fit(images: np.ndarray, labels: np.ndarray, n_val: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialise the split as arrays. See :func:`split_indices` for the guarantees."""
    fit_idx, eval_idx = split_indices(labels, n_val, seed)
    return images[fit_idx], labels[fit_idx], images[eval_idx], labels[eval_idx]


def build_loaders(cfg, download: bool = False) -> dict[str, DataLoader]:
    """Build fit / eval loaders with quality applied to the fit split only."""
    from src.data.quality import apply_quality

    images, labels = load_arrays(cfg, download)
    size = image_size_for(cfg.data.dataset)
    train_t, eval_t = _transforms(cfg.data.dataset, True), _transforms(cfg.data.dataset, False)
    fit_images, fit_labels, eval_images, eval_labels = split_eval_fit(
        images, labels, min(cfg.study.n_val, images.shape[0] - 1), seed=EVAL_SPLIT_SEED)
    damaged_images, damaged_labels = apply_quality(
        fit_images, fit_labels, cfg.quality.operator, cfg.quality.quality,
        cfg.quality.seed, num_classes_for(cfg.data.dataset), size)
    fit = ArrayDataset(damaged_images, damaged_labels, train_t)
    ev = ArrayDataset(eval_images, eval_labels, eval_t)
    if cfg.data.subset_fit is not None:
        fit = Subset(fit, list(range(min(cfg.data.subset_fit, len(fit)))))
    common = {"batch_size": cfg.data.batch_size, "num_workers": cfg.data.num_workers,
              "pin_memory": torch.cuda.is_available()}
    return {"fit": DataLoader(fit, shuffle=True, **common),
            "eval": DataLoader(ev, shuffle=False, **common)}


def logit_safe(prob: np.ndarray) -> np.ndarray:
    """Clip probabilities away from 0 and 1 before taking a logarithm."""
    return np.clip(prob, MIN_LOGIT, 1.0 - MIN_LOGIT)