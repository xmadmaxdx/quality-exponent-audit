"""Small backbones sized by `width` so the model-size axis of a scaling grid is real."""
from __future__ import annotations

import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nn.functional.relu(out + self.shortcut(x))


class ResNetCifar(nn.Module):
    """CIFAR-style ResNet. `width` scales every stage, so it is the size axis."""

    def __init__(self, blocks_per_stage: int, width: int, num_classes: int) -> None:
        super().__init__()
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be >= 1")
        if width < 1:
            raise ValueError("width must be >= 1")
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, blocks_per_stage, 1)
        self.layer2 = self._make_layer(2 * width, blocks_per_stage, 2)
        self.layer3 = self._make_layer(4 * width, blocks_per_stage, 2)
        self.fc = nn.Linear(4 * width, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        layers += [BasicBlock(planes, planes, 1) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = nn.functional.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(name: str, num_classes: int, width: int = 16) -> nn.Module:
    """Build a backbone. Raises ValueError on an unknown name."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if name == "resnet20":
        return ResNetCifar(3, width, num_classes)
    if name == "resnet32":
        return ResNetCifar(5, width, num_classes)
    raise ValueError(f"unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))