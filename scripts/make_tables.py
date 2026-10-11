"""Generate every table in the paper from stored run artifacts. Never hand-typed."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from src.scaling.fitters import FITTERS
from src.scaling.inference import measure_sigma_run


def load_runs(root: str) -> list[dict]:
    """Load every completed run's metrics, skipping anything unfinished."""
    runs: list[dict] = []
    for path in sorted(glob.glob(f"{root}/**/metrics.json", recursive=True)):
        run_dir = Path(path).parent
        if not (run_dir / "DONE").is_file():
            print(f"skipping incomplete run: {run_dir.name}", flush=True)
            continue
        runs.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return runs


def cell_losses(runs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average log-loss within each (N, D, Q, operator) cell across seeds."""
    cells: dict[tuple, list[float]] = {}
    for run in runs:
        point = run["grid_point"]
        key = (point["n_params"], point["n_examples"], round(point["quality"], 6),
               point["operator"])
        cells.setdefault(key, []).append(run["final_eval_nll"])
    log_n, log_d, log_q, loss = [], [], [], []
    for (n_params, n_examples, quality, _operator), values in sorted(cells.items()):
        log_n.append(np.log(n_params))
        log_d.append(np.log(n_examples))
        log_q.append(np.log(max(quality, 1e-6)))
        loss.append(float(np.mean(values)))
    return (np.asarray(log_n), np.asarray(log_d), np.asarray(log_q), np.asarray(loss))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper tables from stored runs")
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--outdir", default="results/tables")
    args = parser.parse_args()
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no completed runs under {args.runs}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Table 1: seeds per cell, and sigma_run per cell.
    cells: dict[tuple, list[float]] = {}
    for run in runs:
        point = run["grid_point"]
        key = (point["operator"], round(point["quality"], 4))
        cells.setdefault(key, []).append(run["final_eval_nll"])
    rows = []
    for (operator, quality), values in sorted(cells.items()):
        sigma = measure_sigma_run(np.asarray(values)) if len(values) > 1 else None
        rows.append({"operator": operator, "quality": quality, "n_seeds": len(values),
                     "mean_eval_nll": float(np.mean(values)),
                     "sigma_run": sigma.std if sigma else float("nan")})
    (outdir / "sigma_by_cell.csv").write_text(
        "operator,quality,n_seeds,mean_eval_nll,sigma_run\n" +
        "\n".join(f"{r['operator']},{r['quality']},{r['n_seeds']},"
                  f"{r['mean_eval_nll']:.6f},{r['sigma_run']:.6f}" for r in rows),
        encoding="utf-8")

    # Table 2: fitted exponents per fitter, from the cell-averaged grid.
    log_n, log_d, log_q, loss = cell_losses(runs)
    if log_d.size >= 3:
        fitted = {}
        for name, fitter in FITTERS.items():
            try:
                result = fitter(log_d, log_q, log_n, loss)
                fitted[name] = {"gamma": result.gamma, "converged": result.converged}
            except ValueError as exc:
                fitted[name] = {"gamma": float("nan"), "converged": False, "error": str(exc)}
        (outdir / "gamma_by_fitter.json").write_text(
            json.dumps({"n_cells": int(log_d.size), "fits": fitted}, indent=2), encoding="utf-8")
        print(json.dumps(fitted, indent=2))
    else:
        print(f"only {log_d.size} cells; need >= 3 to fit. Run more grid points.")

    print(f"wrote tables to {outdir} from {len(runs)} completed runs")


if __name__ == "__main__":
    main()