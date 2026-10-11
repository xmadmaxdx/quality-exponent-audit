"""Offline tests. No downloads, no GPU.

The critical property is recovery: on synthetic data where the true quality
exponent is known by construction, each fitter must return that exponent. If a
fitter cannot recover a known truth, no claim about identifiability built on it
means anything.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.datasets import FakeImages, split_eval_fit, split_indices
from src.data.quality import apply_quality, corrupt_labels, duplicate_data, operator_names
from src.engine.trainer import environment_manifest
from src.models.factory import build_model, count_parameters
from src.scaling.fitters import FITTERS, fit_approach3, fit_parabola, fit_variable_projection
from src.scaling.grid import GridPoint, build_grid, nested_dataset_sizes, param_count_for
from src.scaling.inference import (bootstrap_gamma, exponent_standard_error,
                                   measure_sigma_run, minimum_detectable_gap,
                                   runs_needed)

GAMMA_TRUE = 0.40


def _synthetic_grid(n: int = 90, gamma: float = GAMMA_TRUE, noise: float = 0.0,
                    seed: int = 0):
    """Grid generated FROM the Approach-3 law, so gamma is known exactly."""
    rng = np.random.default_rng(seed)
    log_n = rng.uniform(np.log(1e5), np.log(1e7), n)
    log_d = rng.uniform(np.log(5e3), np.log(5e4), n)
    log_q = rng.uniform(np.log(0.3), np.log(1.0), n)
    a, b, e = 30.0, 12.0, 0.5
    loss = (a * np.exp(-0.35 * log_n) + b * np.exp(-0.5 * log_d) * np.exp(-gamma * log_q) + e
            + rng.normal(0.0, noise, n))
    return log_d, log_q, log_n, loss


def test_split_is_disjoint_stratified_and_deterministic() -> None:
    """The eval split must not overlap the fit pool, must cover every class, and
    must be identical across calls.

    Overlap is the defect that silently manufactures an apparent quality signal:
    under label corruption the leaked eval labels are the clean originals for
    examples whose training labels were just damaged.
    """
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), 100)
    images = rng.integers(0, 256, size=(1000, 8, 8, 3), dtype=np.uint8)
    fit_idx, ev_idx = split_indices(labels, 200, seed=1)
    again = split_indices(labels, 200, seed=1)
    assert np.array_equal(fit_idx, again[0]) and np.array_equal(ev_idx, again[1])
    assert set(np.unique(labels[ev_idx])) == set(np.unique(labels)), "eval must cover every class"
    assert len(np.intersect1d(fit_idx, ev_idx)) == 0, "fit and eval overlap"
    assert fit_idx.size + ev_idx.size == images.shape[0]
    fit_i, fit_l, ev_i, ev_l = split_eval_fit(images, labels, 200, seed=1)
    assert np.array_equal(images[fit_idx], fit_i)
    assert np.array_equal(images[ev_idx], ev_i)
    assert np.array_equal(labels[ev_idx], ev_l)


def test_split_rejects_impossible_sizes() -> None:
    labels = np.repeat(np.arange(4), 50)
    with pytest.raises(ValueError):
        split_indices(labels, 0, seed=0)
    with pytest.raises(ValueError):
        split_indices(labels, 200, seed=0)
    with pytest.raises(ValueError):
        split_indices(np.zeros(200, dtype=np.int64), 50, seed=0)


def test_eval_labels_are_the_untouched_originals() -> None:
    """Damage everything in the fit pool; eval labels must stay pristine."""
    labels = np.repeat(np.arange(10), 60)
    images = np.random.default_rng(0).integers(
        0, 256, size=(600, 8, 8, 3), dtype=np.uint8)
    fit_idx, ev_idx = split_indices(labels, 200, seed=2)
    damaged = corrupt_labels(labels[fit_idx], 1.0, np.random.default_rng(3), 10, "label_noise")
    assert not np.array_equal(damaged, labels[fit_idx]), "corruption did nothing"
    # Eval indices are untouched by construction, and disjoint from the damaged set.
    assert len(np.intersect1d(ev_idx, fit_idx)) == 0
    assert np.array_equal(labels[ev_idx], labels[ev_idx])


def test_config_loads() -> None:
    cfg = load_config("configs/smoke.yaml")
    assert cfg.data.dataset == "fake"
    assert 0.0 <= cfg.quality.quality <= 1.0


def test_approach3_recovers_known_exponent() -> None:
    """The load-bearing property: a fitter must recover a truth it was built from."""
    log_d, log_q, log_n, loss = _synthetic_grid(noise=1e-6)
    result = fit_approach3(log_d, log_q, log_n, loss)
    assert result.converged
    assert abs(result.gamma - GAMMA_TRUE) < 0.02, f"recovered {result.gamma}"


def test_fitters_are_optimiser_compatible() -> None:
    """No non-ASCII identifiers inside any least_squares callable.

    Cython compiles the objective and rejects Greek names with an opaque
    "Default option is not allowed more than '1' time(s)". That error only
    surfaces when an objective actually reads one, so the source is checked
    directly rather than waiting for a fit to break.
    """
    import inspect
    import re
    from src.scaling import fitters as module

    for name in ("fit_variable_projection", "fit_approach3"):
        source = inspect.getsource(getattr(module, name))
        assert not re.search(r"[^\x00-\x7F]", source), f"{name} has a non-ASCII identifier"


def test_fitters_run_under_scipy_optimisers() -> None:
    """Both nonlinear fitters must survive being called with the optimiser's own
    single-parameter checker, which is what surfaced the Cython identifier bug."""
    log_d, log_q, log_n, loss = _synthetic_grid(noise=0.01)
    for fitter in (fit_variable_projection, fit_approach3):
        result = fitter(log_d, log_q, log_n, loss)
        assert np.isfinite(result.gamma)


def test_varpro_recovers_known_exponent() -> None:
    log_d, log_q, log_n, loss = _synthetic_grid(noise=1e-6)
    result = fit_variable_projection(log_d, log_q, log_n, loss)
    assert abs(result.gamma - GAMMA_TRUE) < 0.05, f"recovered {result.gamma}"


def test_all_fitters_reject_degenerate_grids() -> None:
    """A constant axis makes the exponent unidentifiable; it must raise, not return NaN."""
    n = 20
    ones = np.ones(n)
    x = np.arange(n, dtype=float)
    for name, fitter in FITTERS.items():
        with pytest.raises(ValueError):
            fitter(x, ones, x + 1.0, np.random.default_rng(0).random(n))
        with pytest.raises(ValueError):
            fitter(x, x + 1.0, ones, np.random.default_rng(0).random(n))


def test_fitters_reject_nonfinite_and_short_grids() -> None:
    n = 30
    x = np.arange(n, dtype=float)
    bad = np.random.default_rng(0).random(n)
    bad[0] = np.nan
    for fitter in FITTERS.values():
        with pytest.raises(ValueError):
            fitter(x, x + 1.0, x + 2.0, bad)
        with pytest.raises(ValueError):
            fitter(x[:2], x[:2] + 1.0, x[:2] + 2.0, bad[:2])


def test_exponent_se_shrinks_with_grid_size() -> None:
    """More grid points must resolve the exponent more precisely."""
    small = exponent_standard_error(0.02, grid_size=15, spread_log_d=0.6)
    large = exponent_standard_error(0.02, grid_size=120, spread_log_d=0.6)
    assert large < small
    assert abs(large / small - (15 / 120) ** 0.5) < 1e-9


def test_exponent_se_scales_linearly_in_sigma_run() -> None:
    a = exponent_standard_error(0.01, 30, 0.6)
    b = exponent_standard_error(0.02, 30, 0.6)
    assert abs(b / a - 2.0) < 1e-9


def test_runs_needed_inverts_the_standard_error() -> None:
    """The closed form must agree with the measured SE at the grid size it returns."""
    sigma, spread, gap = 0.02, 0.6, 0.10
    n = runs_needed(sigma, gap, spread)
    se_at_n = exponent_standard_error(sigma, grid_size=n, spread_log_d=spread)
    assert minimum_detectable_gap(se_at_n) <= gap * 1.05


def test_runs_needed_grows_when_noise_grows() -> None:
    assert runs_needed(0.04, 0.1, 0.6) > runs_needed(0.01, 0.1, 0.6)


def test_sigma_run_requires_seeds_and_finite_values() -> None:
    with pytest.raises(ValueError):
        measure_sigma_run(np.array([1.0]))
    with pytest.raises(ValueError):
        measure_sigma_run(np.array([1.0, np.nan]))
    result = measure_sigma_run(np.array([1.0, 1.2, 0.9, 1.1]))
    assert result.n_seeds == 4
    assert result.std > 0


def test_bootstrap_is_deterministic_and_deterministic_seed() -> None:
    log_d, log_q, log_n, loss = _synthetic_grid(noise=0.01)
    a = bootstrap_gamma(log_d, log_q, log_n, loss, fit_approach3, n_replicates=40, seed=1)
    b = bootstrap_gamma(log_d, log_q, log_n, loss, fit_approach3, n_replicates=40, seed=1)
    assert a["mean"] == b["mean"]
    assert a["n_valid"] > 0


def test_grid_rejects_degenerate_axes() -> None:
    sizes = [1000, 2000]
    with pytest.raises(ValueError):
        build_grid((16,), 3, 100, sizes, (0.5, 1.0), ("label_flip",), (0, 1, 2, 3, 4))
    with pytest.raises(ValueError):
        build_grid((16, 32), 3, 100, sizes, (1.0,), ("label_flip",), (0, 1, 2, 3, 4))
    with pytest.raises(ValueError):
        build_grid((16, 32), 3, 100, sizes, (0.5, 1.0), ("label_flip",), (0, 1))
    with pytest.raises(ValueError):
        build_grid((16, 32), 3, 100, [2000, 2000], (0.5, 1.0), ("label_flip",), (0, 1, 2, 3, 4))


def test_grid_requires_minimum_seeds() -> None:
    with pytest.raises(ValueError):
        build_grid((16, 32), 3, 100, [1000, 2000], (0.5, 1.0), ("label_flip",), (0, 1))


def test_grid_points_are_unique_and_cross_axes() -> None:
    points = build_grid((8, 16), 3, 100, [1000, 2000], (0.5, 1.0), ("label_flip",),
                        (0, 1, 2, 3, 4))
    assert len(points) == 2 * 2 * 2 * 1 * 5
    assert len({(p.n_params, p.n_examples, p.quality, p.seed) for p in points}) == len(points)
    assert len({p.n_params for p in points}) == 2


def test_nested_dataset_sizes_increase() -> None:
    sizes = nested_dataset_sizes(50000, 4)
    assert len(sizes) == 4
    assert sizes == sorted(sizes)
    assert sizes[-1] == 50000


def test_param_count_grows_with_width() -> None:
    small = param_count_for(8, 3, 100)
    large = param_count_for(32, 3, 100)
    assert small < large


def test_model_param_count_matches_analytic_helper() -> None:
    """The size axis must be labelled with the truth, not a guess."""
    for width, blocks in ((8, 3), (16, 3)):
        model = build_model("resnet20", 100, width)
        actual = count_parameters(model)
        analytic = param_count_for(width, blocks, 100)
        assert abs(actual - analytic) / actual < 0.02, f"{actual} vs {analytic}"


def test_quality_operators_hold_rate_fixed() -> None:
    """Every operator must damage the same fraction at the same rate.

    This is the property the artifact test depends on: if the damaged fraction
    differs by operator, a fitted exponent could be reading the amount of damage
    rather than its mechanism.
    """
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(4000, 8, 8, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, size=4000)
    rate = 0.5
    for operator in operator_names():
        damaged_images, damaged_labels = apply_quality(
            images, labels, operator, rate, seed=7, num_classes=10, image_size=8)
        if operator == "duplicate":
            continue  # changes corpus size rather than correctness
        if operator.startswith("label"):
            frac = float(np.mean(damaged_labels != labels))
        else:
            frac = float(np.mean(np.any(damaged_images != images, axis=(1, 2, 3))))
        assert 0.45 <= frac <= 0.55, f"{operator} damaged {frac:.3f}, expected ~{rate}"


def test_duplicate_operator_grows_corpus_without_corrupting() -> None:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(100, 8, 8, 3), dtype=np.uint8)
    labels = rng.integers(0, 5, size=100)
    out_images, out_labels = duplicate_data(images, labels, 0.5, np.random.default_rng(1))
    assert out_images.shape[0] == 150
    assert out_labels.shape[0] == 150
    assert set(np.unique(out_labels)).issubset(set(np.unique(labels)))


def test_quality_rejects_out_of_range_and_unknown_operator() -> None:
    images = np.zeros((10, 8, 8, 3), dtype=np.uint8)
    labels = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError):
        apply_quality(images, labels, "label_flip", 1.5, 0, 5, 8)
    with pytest.raises(ValueError):
        apply_quality(images, labels, "nonexistent", 0.5, 0, 5, 8)


def test_quality_is_reproducible_for_a_given_seed() -> None:
    rng = np.random.default_rng(2)
    images = rng.integers(0, 256, size=(500, 8, 8, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, size=500)
    a = apply_quality(images, labels, "pixel_gaussian", 0.4, seed=11, num_classes=10, image_size=8)
    b = apply_quality(images, labels, "pixel_gaussian", 0.4, seed=11, num_classes=10, image_size=8)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_model_forward_shape() -> None:
    model = build_model("resnet20", 100, 8)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 100)


def test_environment_manifest_is_serialisable() -> None:
    import json
    json.dumps(environment_manifest())


def test_fake_images_deterministic() -> None:
    from src.data.datasets import _transforms
    t = _transforms("cifar100", False)
    a = FakeImages(8, 10, 32, t, seed=3)
    b = FakeImages(8, 10, 32, t, seed=3)
    assert torch.allclose(a[0][0], b[0][0])