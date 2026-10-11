"""Experiment 1: measure sigma_run.

This is the load-bearing measurement. Every standard error in the project scales
with the between-seed dispersion of log-loss at a fixed grid cell, so it must be
measured before anything is derived from it. Run this first.

If sigma_run comes out small, the identifiability correction is cosmetic and the
paper becomes a negative result. If it comes out large, the published claims are
unidentifiable and the paper is strong. Both are publishable; only guessing is
not.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.config import RunConfig, load_config
from src.constants import EVAL_SPLIT_SEED
from src.data.datasets import (ArrayDataset, _transforms, load_arrays, nested_subset,
                               num_classes_for, split_eval_fit)
from src.engine.trainer import run_dir, train_one
from src.models.factory import build_model
from src.scaling.inference import (exponent_standard_error, measure_sigma_run,
                                   minimum_detectable_gap, runs_needed)
from src.utils.common import get_device, setup_logging
from src.utils.drive import ensure_drive_mounted, mirror_file

log = logging.getLogger("quality-exponent-audit")


def main() -> None:
    setup_logging()
    ensure_drive_mounted()
    parser = argparse.ArgumentParser(description="Measure between-seed sigma_run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", default="results/sigma_run.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        raise SystemExit("need at least two seeds to measure sigma_run")
    device = get_device()

    images, labels = load_arrays(cfg, args.download)
    total = images.shape[0]
    # Hold out the eval split BEFORE subsampling, so the fit subset can never
    # contain an eval example.
    fit_pool_images, fit_pool_labels, eval_images, eval_labels = split_eval_fit(
        images, labels, min(cfg.study.n_val, total - 1), seed=EVAL_SPLIT_SEED)
    # Measure at a single mid-sized cell: the dispersion depends on regime, and
    # the paper reports it per configuration rather than as one global constant.
    cell_size = max(1000, fit_pool_images.shape[0] // 10)
    subset_images, subset_labels = nested_subset(
        fit_pool_images, fit_pool_labels, cell_size / fit_pool_images.shape[0], seed=0)

    losses: list[float] = []
    # The measurement cell is clean data: sigma_run is the dispersion of the
    # training process itself, so introducing corruption here would confound the
    # measurement with the factor under study. Quality enters later, per grid.
    train_t = _transforms(cfg.data.dataset, True)
    eval_t = _transforms(cfg.data.dataset, False)
    fit_ds = ArrayDataset(subset_images, subset_labels, train_t)
    ev_ds = ArrayDataset(eval_images, eval_labels, eval_t)
    common = {"batch_size": cfg.data.batch_size, "num_workers": cfg.data.num_workers,
              "pin_memory": device.type == "cuda"}
    loaders = {"fit": DataLoader(fit_ds, shuffle=True, **common),
               "eval": DataLoader(ev_ds, shuffle=False, **common)}

    for seed in seeds:
        from src.config import TrainConfig
        run_cfg = RunConfig(
            data=cfg.data, quality=cfg.quality, model=cfg.model,
            train=TrainConfig(**{**cfg.train.__dict__, "seed": seed}), study=cfg.study)
        model = build_model(cfg.model.name, num_classes_for(cfg.data.dataset), cfg.model.width).to(device)
        point = {"n_params": int(sum(p.numel() for p in model.parameters())),
                 "n_examples": cell_size, "quality": 1.0, "operator": "clean",
                 "seed": seed, "width": cfg.model.width, "blocks": 3}
        metrics = train_one(run_cfg, model, loaders, device, point)
        losses.append(metrics["final_eval_nll"])
        log.info("seed=%d eval_nll=%.6f", seed, losses[-1])

    sigma = measure_sigma_run(np.asarray(losses))
    spread = float(np.log(cell_size).std())
    row = {
        "cell": {"n_examples": cell_size, "dataset": cfg.data.dataset,
                 "model": cfg.model.name, "width": cfg.model.width,
                 "epochs": cfg.train.epochs},
        "losses": losses,
        "sigma_run": sigma.as_dict(),
        "se_gamma_by_grid": {
            str(g): exponent_standard_error(sigma.std, g, spread) for g in cfg.study.grid_sizes},
        "mde_by_grid": {
            str(g): minimum_detectable_gap(exponent_standard_error(sigma.std, g, spread))
            for g in cfg.study.grid_sizes},
        "runs_needed_for_gap_0p10": runs_needed(sigma.std, 0.10, spread),
        "interpretation": ("identifiable at published grid sizes"
                           if sigma.std > 0 else "degenerate"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    mirror_file(out)
    log.info("sigma_run=%.6f over %d seeds", sigma.std, sigma.n_seeds)
    log.info("runs needed to resolve a 0.10 gap: %d", row["runs_needed_for_gap_0p10"])
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()