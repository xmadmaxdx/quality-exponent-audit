"""Experiment 2: the corruption-operator artifact.

The claim under test: the fitted quality exponent should be a property of the
corpus, not of the instrument used to damage it. If the exponent moves when the
corruption *mechanism* changes while the *quality axis* stays the same, then any
law built on it is partly measuring the corrupter.

Design note. An exponent cannot be identified from a single quality level: with
quality fixed, log(Q) has no variance and no slope exists. The experiment
therefore characterises every operator over the same ladder of corruption rates,
fits the law within each operator, and then asks whether the resulting exponents
agree across operators. The comparison is at matched severity *axis*, not at a
single severity.

The `duplicate` operator is reported separately. Duplication grows the corpus
without adding information, so it perturbs the dataset-size axis rather than
holding it fixed, and its exponent is therefore not comparable to the rest.
Pooling it into the headline spread would manufacture agreement or disagreement
out of a confound.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.config import RunConfig, load_config
from src.constants import EVAL_SPLIT_SEED
from src.data.datasets import (ArrayDataset, _transforms, load_arrays, nested_subset,
                               num_classes_for, split_eval_fit)
from src.data.quality import PIXEL_OPERATORS, LABEL_OPERATORS, apply_quality, operator_names
from src.engine.trainer import train_one
from src.models.factory import build_model
from src.scaling.fitters import FITTERS as FITTERS_BY_NAME, fit_all
from src.scaling.inference import bootstrap_gamma
from src.utils.common import get_device, setup_logging
from src.utils.drive import ensure_drive_mounted, mirror_file

log = logging.getLogger("quality-exponent-audit")

# Operators that damage examples in place and therefore leave the dataset-size
# axis untouched. Only these are comparable to one another.
SAME_AXIS_OPERATORS = tuple(sorted(PIXEL_OPERATORS | LABEL_OPERATORS))

# Reference fitter for the headline. Approach 3 is the form the literature
# proposes, so it is the one the critique must be made against.
HEADLINE_FITTER = "approach3"

# Spread below which the exponents are treated as indistinguishable. This is a
# declared threshold, not a fitted one; the bootstrap SE is reported alongside
# it so a reader can apply their own.
SPREAD_THRESHOLD = 0.05


def main() -> None:
    setup_logging()
    ensure_drive_mounted()
    parser = argparse.ArgumentParser(description="Corruption-operator artifact test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rates", default="0.1,0.3,0.5,0.7,0.9",
                        help="corruption-rate ladder, identical for every operator")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", default="results/artifact.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    if len(rates) < 3:
        raise SystemExit("need at least three corruption rates to identify an exponent")
    if not all(0.0 < r < 1.0 for r in rates):
        raise SystemExit("every rate must be in (0, 1)")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        raise SystemExit("need at least two seeds per cell")
    device = get_device()
    classes = num_classes_for(cfg.data.dataset)

    images, labels = load_arrays(cfg, args.download)
    total = images.shape[0]
    # Hold out eval once, for the whole study, before any corruption is applied.
    fit_pool_images, fit_pool_labels, eval_images, eval_labels = split_eval_fit(
        images, labels, min(cfg.study.n_val, total - 1), seed=EVAL_SPLIT_SEED)
    pool = fit_pool_images.shape[0]

    # Nested dataset sizes so the D axis is crossed rather than aliased onto Q.
    sizes = sorted({max(500, pool // 20), max(1000, pool // 5), max(2000, pool // 2)})
    sizes = [s for s in sizes if s <= pool] or [max(500, pool // 2)]
    sizes = list(dict.fromkeys(sizes))
    while len(sizes) < 3:
        sizes.append(min(pool, sizes[-1] * 2))
    # Two model sizes so the N axis is crossed too.
    widths = (cfg.model.width, cfg.model.width * 2)

    train_t = _transforms(cfg.data.dataset, True)
    eval_t = _transforms(cfg.data.dataset, False)
    common = {"batch_size": cfg.data.batch_size, "num_workers": cfg.data.num_workers,
              "pin_memory": device.type == "cuda"}
    # The eval set is fixed for the whole study, so its loader is built once.
    eval_loader = DataLoader(ArrayDataset(eval_images, eval_labels, eval_t),
                             shuffle=False, **common)

    per_operator: dict[str, dict] = {}
    for operator in operator_names():
        log_n, log_d, log_q, losses = [], [], [], []
        for width in widths:
            n_params = int(sum(p.numel() for p in build_model(
                cfg.model.name, classes, width).parameters()))
            for size in sizes:
                sub_images, sub_labels = nested_subset(
                    fit_pool_images, fit_pool_labels, min(1.0, size / pool), seed=0)
                for rate in rates:
                    dmg_images, dmg_labels = apply_quality(
                        sub_images, sub_labels, operator, rate, seed=1234,
                        num_classes=classes, image_size=cfg.data.image_size)
                    loaders = {
                        "fit": DataLoader(ArrayDataset(dmg_images, dmg_labels, train_t),
                                          shuffle=True, **common),
                        "eval": eval_loader,
                    }
                    cell_losses = []
                    for seed in seeds:
                        run_cfg = RunConfig(
                            data=cfg.data, quality=cfg.quality, model=cfg.model,
                            train=cfg.train, study=cfg.study)
                        model = build_model(cfg.model.name, classes, width).to(device)
                        point = {"n_params": n_params, "n_examples": int(size),
                                 "quality": 1.0 - rate, "operator": operator,
                                 "seed": seed, "width": width, "blocks": 3}
                        metrics = train_one(run_cfg, model, loaders, device, point)
                        cell_losses.append(metrics["final_eval_nll"])
                    log_n.append(np.log(n_params))
                    log_d.append(np.log(size))
                    log_q.append(np.log(max(1.0 - rate, 1e-6)))
                    losses.append(float(np.mean(cell_losses)))

        log_n_a, log_d_a, log_q_a, loss_a = (np.asarray(v) for v in (log_n, log_d, log_q, losses))
        fits = fit_all(log_d_a, log_q_a, log_n_a, loss_a)
        boot = {name: bootstrap_gamma(log_d_a, log_q_a, log_n_a, loss_a,
                                      FITTERS_BY_NAME[name],
                                      n_replicates=args.bootstrap, seed=0)
                for name in fits}
        per_operator[operator] = {
            "gamma": {name: fit.gamma for name, fit in fits.items()},
            "gamma_se": {name: boot[name]["se"] for name in boot},
            "converged": {name: fit.converged for name, fit in fits.items()},
            "mean_losses": [float(v) for v in loss_a],
            "comparable": operator in SAME_AXIS_OPERATORS,
        }
        log.info("operator=%-14s gamma=%s", operator,
                 {k: round(v, 4) for k, v in per_operator[operator]["gamma"].items()})

    comparable = {op: per_operator[op]["gamma"][HEADLINE_FITTER]
                  for op in per_operator if per_operator[op]["comparable"]}
    valid = {op: g for op, g in comparable.items() if np.isfinite(g)}
    spread = (max(valid.values()) - min(valid.values())) if len(valid) > 1 else float("nan")
    # Paired against the pooled bootstrap SE: a spread is only meaningful if it
    # exceeds the noise in each individual estimate.
    se_max = max((per_operator[op]["gamma_se"][HEADLINE_FITTER] for op in valid),
                 default=float("nan"))
    payload = {
        "config": str(args.config),
        "dataset": cfg.data.dataset,
        "rate_ladder": rates,
        "seeds_per_cell": len(seeds),
        "grid": {"widths": list(widths), "dataset_sizes": sizes,
                 "n_cells_per_operator": len(rates) * len(sizes) * len(widths)},
        "per_operator": per_operator,
        "headline": {
            "fitter": HEADLINE_FITTER,
            "comparable_operators": sorted(valid),
            "gamma_by_operator": valid,
            "excluded_operators": sorted(set(comparable) - set(valid)) or
                                   sorted(op for op in per_operator if not per_operator[op]["comparable"]),
            "spread_across_operators": spread,
            "max_bootstrap_se": se_max,
            "spread_exceeds_noise": bool(np.isfinite(spread) and spread > se_max),
            "claim_supported": bool(np.isfinite(spread) and spread > max(SPREAD_THRESHOLD, se_max)),
            "threshold": SPREAD_THRESHOLD,
            "note": ("gamma spread across operators fitted over the same rate ladder. "
                     "If the spread exceeds the bootstrap noise, the exponent tracks "
                     "the corruption mechanism rather than the corpus alone."),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mirror_file(out)
    log.info("gamma spread across comparable operators: %s (max SE %s)",
             round(spread, 4) if np.isfinite(spread) else spread,
             round(se_max, 4) if np.isfinite(se_max) else se_max)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
