"""Single-run training with complete, resumable artifact storage.

Every run writes a self-contained bundle so the paper's tables can be
regenerated without retraining, and so a Colab disconnect costs at most one run
rather than a whole grid. Raw per-example predictions are stored so that every
metric in the paper is recomputable after the fact.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.common import cleanup_cuda, get_rng_state, restore_rng_state, set_seed

log = logging.getLogger("quality-exponent-audit")


def _make_optimizer(cfg, model: torch.nn.Module):
    if cfg.train.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.train.lr,
                               momentum=cfg.train.momentum,
                               weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                             weight_decay=cfg.train.weight_decay)


def environment_manifest() -> dict:
    """Everything needed to explain why a run is not bit-reproducible elsewhere."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "cpu"),
    }


def run_dir(cfg, grid_point: dict) -> Path:
    """Canonical on-disk location for one run. Sorts lexicographically by design."""
    stem = (f"N{grid_point['n_params']:>9d}_D{grid_point['n_examples']:>7d}"
            f"_Q{grid_point['quality']:.4f}_{grid_point['operator']}"
            f"_s{grid_point['seed']:>3d}")
    return Path(cfg.train.checkpoint_dir) / cfg.study.name / stem


def train_one(cfg, model: torch.nn.Module, loaders: dict, device: torch.device,
              grid_point: dict) -> dict:
    """Train one grid point. Returns its metrics dict. Resumes if interrupted."""
    directory = run_dir(cfg, grid_point)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    done_path = directory / "DONE"
    if done_path.is_file() and metrics_path.is_file():
        log.info("run already complete: %s", directory.name)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    seed = int(grid_point["seed"])
    set_seed(seed)
    optimizer = _make_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    loss_fn = nn.CrossEntropyLoss()
    start_epoch = 0
    best_eval_nll = float("inf")
    history: list[dict] = []
    last_path = directory / "last.pt"
    if last_path.is_file():
        try:
            state = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["opt"])
            scheduler.load_state_dict(state["sched"])
            start_epoch = int(state["epoch"])
            best_eval_nll = float(state["best_eval_nll"])
            history = list(state["history"])
            restore_rng_state(state["rng_state"])
            log.info("resumed %s at epoch %d", directory.name, start_epoch)
        except Exception as exc:
            # A silent restart would discard completed epochs invisibly.
            log.error("resume failed for %s (%s); restarting from scratch", directory.name, exc)
            start_epoch, history, best_eval_nll = 0, [], float("inf")

    log_path = directory / "train_log.jsonl"
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            model.train()
            running, seen = 0.0, 0
            for x, y in loaders["fit"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch + 1} in {directory.name}")
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * x.size(0)
                seen += x.size(0)
            scheduler.step()
            train_nll = running / max(1, seen)
            eval_nll, eval_acc, preds, labels = evaluate(model, loaders["eval"], device)
            if eval_nll < best_eval_nll:
                best_eval_nll = eval_nll
                torch.save(model.state_dict(), directory / "best.pt")
            record = {"epoch": epoch + 1, "train_nll": train_nll, "eval_nll": eval_nll,
                      "eval_acc": eval_acc, "seconds": time.perf_counter()}
            history.append(record)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            np.save(directory / "preds.npy", preds.astype(np.float32))
            np.save(directory / "labels.npy", labels.astype(np.int64))
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                        "sched": scheduler.state_dict(), "epoch": epoch + 1,
                        "best_eval_nll": best_eval_nll, "history": history,
                        "rng_state": get_rng_state()}, last_path)
    finally:
        cleanup_cuda()

    metrics = {
        "grid_point": grid_point,
        "epochs": len(history),
        "final_train_nll": history[-1]["train_nll"] if history else float("nan"),
        "final_eval_nll": history[-1]["eval_nll"] if history else float("nan"),
        "final_eval_acc": history[-1]["eval_acc"] if history else float("nan"),
        "best_eval_nll": best_eval_nll,
        "n_params": grid_point["n_params"],
        "seed": seed,
        "environment": environment_manifest(),
        "config": {name: asdict(getattr(cfg, name))
                   for name in ("data", "quality", "model", "train", "study")},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    done_path.write_text(json.dumps({"completed": True}), encoding="utf-8")
    return metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Eval NLL, accuracy, and raw per-example predictions.

    Predictions are stored rather than only the scalar so that a different metric
    can be computed later without retraining.
    """
    model.eval()
    total_nll, seen, correct = 0.0, 0, 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_nll += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct += int((logits.argmax(1) == y).sum().item())
        all_preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        seen += x.size(0)
    if seen == 0:
        raise ValueError("eval loader produced no batches")
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_nll / seen, correct / seen, preds, labels