"""Download the datasets. Colab-only entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.datasets import _TORCHVISION


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-100 / Tiny-ImageNet")
    parser.add_argument("--root", default="data")
    parser.add_argument("--sets", default="cifar100")
    args = parser.parse_args()
    import torchvision.datasets as tvd

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        if name not in _TORCHVISION:
            raise SystemExit(f"unknown dataset: {name}")
        if name == "tinyimagenet":
            raise SystemExit(
                "Tiny-ImageNet is not a torchvision dataset; fetch it manually "
                "(http://cs231n.stanford.edu/tiny-imagenet-200.zip) and place it "
                "under the root, then pass an explicit path.")
        getattr(tvd, _TORCHVISION[name][0])(root=str(root), train=True, download=True)
        print(f"ok: {name} -> {root}", flush=True)


if __name__ == "__main__":
    main()