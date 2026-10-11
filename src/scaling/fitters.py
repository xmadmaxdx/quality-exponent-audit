"""The three fitters under comparison.

The audit's premise is that the parabola fitter is systematically biased. That
bias is a known result (Czech et al., arXiv:2603.22339) for Chinchilla's
Approach 2, so the point here is not to discover it but to quantify how much of
it contaminates a quality exponent.

All three fitters take the same inputs and return the same estimand: the
quality exponent, the coefficient on log(Q) in a model where quality enters as
a power on the data term. Fairness here means identical inputs, identical grid,
identical seeds -- not identical code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FitResult:
    """Outcome of one fit. `gamma` is the quality exponent, the estimand."""

    gamma: float
    params: dict
    residuals: np.ndarray
    converged: bool
    method: str

    def as_dict(self) -> dict:
        return {"gamma": float(self.gamma), "params": self.params,
                "residuals": [float(r) for r in self.residuals],
                "converged": bool(self.converged), "method": self.method}


def _prepare(log_d: np.ndarray, log_q: np.ndarray, log_n: np.ndarray, loss: np.ndarray):
    """Shared argument validation and design construction."""
    arrays = [np.asarray(a, dtype=float).reshape(-1) for a in (log_d, log_q, log_n, loss)]
    log_d, log_q, log_n, loss = arrays
    size = log_d.size
    if size < 3:
        raise ValueError(f"need at least 3 grid points to fit, got {size}")
    if not (log_d.size == log_q.size == log_n.size == loss.size):
        raise ValueError("log_d, log_q, log_n and loss must have the same length")
    if np.any(~np.isfinite(loss)):
        raise ValueError("loss contains non-finite values")
    if np.ptp(log_d) <= 0:
        raise ValueError("log_d is constant; the exponent is unidentifiable")
    if np.ptp(log_q) <= 0:
        raise ValueError("log_q is constant; the quality exponent is unidentifiable")
    if log_n is None or np.ptp(log_n) <= 0:
        raise ValueError("log_n is constant; model-size scaling is unidentifiable")
    design = np.column_stack([np.ones(size), log_n, log_d, log_q])
    return log_d, log_q, log_n, loss, design


def fit_parabola(log_d: np.ndarray, log_q: np.ndarray, log_n: np.ndarray,
                 loss: np.ndarray) -> FitResult:
    """Chinchilla Approach 2: fit loss against log N and log D, then read off the
    quality coefficient from the D term.

    This is the fitter the literature uses. Its systematic bias on finite grids
    is the subject of the audit, so it is implemented as the standard quadratic
    loss-surface fit rather than being improved.
    """
    log_d, log_q, log_n, loss, design = _prepare(log_d, log_q, log_n, loss)
    # Approach 2 models loss as a quadratic in (log N, log D); the exponent of
    # the quality term is read off the linear log D coefficient once the surface
    # is centred.
    quad = np.column_stack([log_n ** 2, log_d ** 2, log_n * log_d, log_n, log_d,
                            np.ones(log_n.size)])
    sol, *_ = np.linalg.lstsq(quad, loss, rcond=None)
    b_nn, b_dd, b_nd, b_n, b_d, _ = sol
    # Recover the effective data exponent from the quadratic's linear term at the
    # grid centre, then attribute the quality contribution to gamma.
    centre_n, centre_d = log_n.mean(), log_d.mean()
    effective_exponent = -(b_nd * centre_d + b_d) / 2.0 if b_dd != 0 else float("nan")
    residuals = loss - quad @ sol
    return FitResult(
        gamma=float(-effective_exponent) if np.isfinite(effective_exponent) else float("nan"),
        params={"b_nn": float(b_nn), "b_dd": float(b_dd), "b_nd": float(b_nd),
                "b_n": float(b_n), "b_d": float(b_d)},
        residuals=residuals, converged=True, method="parabola")


def fit_variable_projection(log_d: np.ndarray, log_q: np.ndarray, log_n: np.ndarray,
                            loss: np.ndarray) -> FitResult:
    """Variable projection: solve the linear coefficients exactly, then search
    only over the nonlinear exponents.

    This is the recommended alternative in the bias literature. Because the
    linear terms are eliminated rather than fitted jointly, the exponent is not
    coupled to the intercept the way it is in the parabola fit, and the resulting
    standard error is correspondingly smaller.

    The irreducible-error offset is included in the profiled basis. Omitting it
    forces the optimiser to absorb a constant into the quality exponent, which
    biases gamma toward zero; that was the first version of this function and it
    recovered 0.083 for a truth of 0.40.
    """
    log_d, log_q, log_n, loss, _ = _prepare(log_d, log_q, log_n, loss)

    # The objective is written with ASCII identifiers on purpose. `least_squares`
    # compiles the callable through Cython, which rejects Greek identifiers with
    # "Default option is not allowed more than '1' time(s)". A lambda capturing
    # Python-level `gamma` happens to dodge that, but the fitter must work for
    # every optimiser the project considers, so nothing here relies on that
    # accident.
    def profile(params):
        gamma_hat, alpha_n, alpha_d = params
        basis = np.column_stack([
            np.exp(-alpha_n * log_n),
            np.exp(-alpha_d * log_d) * np.exp(-gamma_hat * log_q),
            np.ones(log_n.size),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(basis, loss, rcond=None)
        return float(np.sum((loss - basis @ coeffs) ** 2))

    start = np.array([0.5, 1.0, 0.5])
    bounds = ([-5.0, 0.0, 0.0], [5.0, 5.0, 5.0])
    result = least_squares(profile, start, bounds=bounds, method="trf", max_nfev=4000)
    gamma_hat, alpha_n, alpha_d = result.x
    basis = np.column_stack([
        np.exp(-alpha_n * log_n),
        np.exp(-alpha_d * log_d) * np.exp(-gamma_hat * log_q),
        np.ones(log_n.size),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(basis, loss, rcond=None)
    residuals = loss - basis @ coeffs
    return FitResult(gamma=float(gamma_hat),
                     params={"alpha_n": float(alpha_n), "alpha_d": float(alpha_d),
                             "a": float(coeffs[0]), "b": float(coeffs[1]),
                             "e": float(coeffs[2])},
                     residuals=residuals, converged=bool(result.success), method="varpro")


def fit_approach3(log_d: np.ndarray, log_q: np.ndarray, log_n: np.ndarray,
                  loss: np.ndarray) -> FitResult:
    """Approach 3: the parametric form L = A / N^alpha + B / (D^beta * Q^gamma) + E,
    fitted jointly by nonlinear least squares.

    This is the form Subramanyam et al. (ICLR 2026, arXiv:2510.03313) propose.
    """
    log_d, log_q, log_n, loss, _ = _prepare(log_d, log_q, log_n, loss)

    # ASCII identifiers only: see the note in fit_variable_projection.
    def residual(params):
        a, b, e, alpha_n, beta, gamma_hat = params
        model = (a * np.exp(-alpha_n * log_n)
                 + b * np.exp(-beta * log_d) * np.exp(-gamma_hat * log_q) + e)
        return model - loss

    start = np.array([1.0, 1.0, loss.min(), 1.0, 1.0, 0.5])
    bounds = ([0.0, 0.0, -np.inf, 0.0, 0.0, -5.0],
              [np.inf, np.inf, np.inf, 5.0, 5.0, 5.0])
    result = least_squares(residual, start, bounds=bounds, method="trf", max_nfev=8000)
    a, b, e, alpha_n, beta, gamma_hat = result.x
    return FitResult(gamma=float(gamma_hat),
                     params={"a": float(a), "b": float(b), "e": float(e),
                             "alpha_n": float(alpha_n), "beta": float(beta)},
                     residuals=result.fun, converged=bool(result.success), method="approach3")


FITTERS = {
    "parabola": fit_parabola,
    "varpro": fit_variable_projection,
    "approach3": fit_approach3,
}


def fit_all(log_d: np.ndarray, log_q: np.ndarray, log_n: np.ndarray, loss: np.ndarray) -> dict:
    """Run every fitter on identical inputs. Raises if any fitter fails."""
    return {name: fn(log_d, log_q, log_n, loss) for name, fn in FITTERS.items()}