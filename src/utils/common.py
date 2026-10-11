"""Determinism, device, and logging helpers."""
from __future__ import annotations

import logging
import random

import numpy as np
import torch

LOGGER_NAME = "quality-exponent-audit"


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger(LOGGER_NAME)


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch.

    Seeding alone is not sufficient: cuDNN autotuning and non-deterministic
    kernels are separate sources of variance, which is exactly what this project
    measures. Both are therefore pinned here rather than left to defaults.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_rng_state() -> dict:
    """Snapshot every RNG stream so a run is reproducible from its bundle."""
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore a snapshot from :func:`get_rng_state`. Raises on a malformed state."""
    required = {"torch", "numpy", "python"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"rng state missing keys: {sorted(missing)}")
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])