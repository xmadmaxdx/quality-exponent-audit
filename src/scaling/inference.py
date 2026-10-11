"""Identifiability machinery: how precisely can the quality exponent be measured?

The audit's central object is the *sampling distribution* of a fitted quality
exponent. Two questions follow from it, and both are answerable in closed form:

1. What is the standard error of the exponent at a given grid size and
   per-run variance?
2. Given that standard error, how many runs are needed to separate two candidate
   exponents by a given gap?

Both answers depend on quantities that must be measured, never assumed. In
particular `sigma_run`, the between-seed standard deviation of log-loss at a
fixed grid cell, is the load-bearing input, and the project's first experiment is
to measure it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.constants import Z_ONE_SIDED, Z_TWO_SIDED


@dataclass(frozen=True)
class SigmaRun:
    """Per-cell between-seed dispersion of log-loss."""

    mean: float
    std: float
    n_seeds: int

    @property
    def well_estimated(self) -> bool:
        return self.n_seeds >= 5

    def as_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "n_seeds": self.n_seeds,
                "well_estimated": self.well_estimated}


def measure_sigma_run(losses: np.ndarray) -> SigmaRun:
    """Between-seed standard deviation of log-loss within one grid cell.

    `losses` is the array of per-seed log-losses at fixed (N, D, Q). This is the
    single most important measured quantity in the project: every standard error
    downstream scales with it.
    """
    values = np.asarray(losses, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("need at least two seeds to measure sigma_run")
    if np.any(~np.isfinite(values)):
        raise ValueError("sigma_run received non-finite losses")
    return SigmaRun(mean=float(values.mean()), std=float(values.std(ddof=1)),
                    n_seeds=int(values.size))


def exponent_standard_error(sigma_run: float, grid_size: int, spread_log_d: float,
                            quality_spread_log_q: float = 1.0,
                            correlation: float = 0.0) -> float:
    """Standard error of a slope estimated from `grid_size` noisy points.

    For a regression slope, SE = sigma_resid / (sqrt(g) * sd(x)), where
    `spread_log_d` is the standard deviation of the predictor across the grid and
    `sigma_run` the per-observation noise. Quality enters as an additional factor
    because the exponent is a difference of two slopes, which roughly doubles the
    variance.

    This is the honest standard error under the stated model. If the residual
    noise is not the fitted slope's noise, the fitter's own bias term must be
    added -- which is precisely what the audit measures empirically.
    """
    if grid_size < 3:
        raise ValueError("grid_size must be >= 3")
    if sigma_run < 0:
        raise ValueError("sigma_run must be >= 0")
    if spread_log_d <= 0 or quality_spread_log_q <= 0:
        raise ValueError("predictor spreads must be > 0")
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must be in [0, 1)")
    # Var(slope) ~ 2 sigma^2 / (g * var(x)) for a difference of two slopes.
    variance = 2.0 * sigma_run ** 2 / (grid_size * (spread_log_d ** 2) * quality_spread_log_q ** 2)
    return float(math.sqrt(variance))


def minimum_detectable_gap(se_gamma: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest true gap in the exponent that `se_gamma` can resolve.

    Standard sample-size-inverting form: (z_{1-alpha/2} + z_{power}) * SE.
    """
    if se_gamma < 0:
        raise ValueError("se_gamma must be >= 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    return (Z_TWO_SIDED + Z_ONE_SIDED) * se_gamma


def runs_needed(sigma_run: float, gap: float, spread_log_d: float, alpha: float = 0.05,
                power: float = 0.80) -> int:
    """Grid points required to resolve a gap in the exponent, in closed form.

    Inverts :func:`exponent_standard_error` and
    :func:`minimum_detectable_gap`. This is the number the literature has never
    reported: how many runs a quality claim actually requires.
    """
    if gap <= 0:
        raise ValueError("gap must be > 0")
    if sigma_run < 0:
        raise ValueError("sigma_run must be >= 0")
    target = gap / (Z_TWO_SIDED + Z_ONE_SIDED)
    if sigma_run == 0:
        return 3
    required = 2.0 * sigma_run ** 2 / (target ** 2 * spread_log_d ** 2)
    return int(max(3, math.ceil(required)))


def bootstrap_gamma(log_d: np.ndarray, log_q: np.ndarray, log_n: np.ndarray,
                    loss: np.ndarray, fitter, n_replicates: int = 400,
                    seed: int = 0) -> dict:
    """Sampling distribution of a fitted exponent, by resampling grid points.

    Resamples the grid with replacement and refits. This captures the fitter's
    bias implicitly: if the parabola fitter is biased, the bootstrap distribution
    is centred off the truth, and the interval is centred off too, which is the
    behaviour the coverage test in the paper exposes.
    """
    log_d = np.asarray(log_d, dtype=float)
    log_q = np.asarray(log_q, dtype=float)
    log_n = np.asarray(log_n, dtype=float)
    loss = np.asarray(loss, dtype=float)
    n = log_d.size
    if n < 3:
        raise ValueError("need at least 3 grid points to bootstrap")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_replicates, dtype=float)
    for rep in range(n_replicates):
        idx = rng.integers(0, n, n)
        try:
            estimates[rep] = fitter(log_d[idx], log_q[idx], log_n[idx], loss[idx]).gamma
        except (ValueError, np.linalg.LinAlgError):
            estimates[rep] = np.nan
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan"),
                "n_valid": 0, "estimates": []}
    return {"mean": float(finite.mean()), "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "se": float(finite.std(ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else 0.0,
            "n_valid": int(finite.size), "estimates": [float(v) for v in finite]}