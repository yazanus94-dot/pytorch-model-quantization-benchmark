from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Accuracy:
    top1: float
    top5: float
    samples: int


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> list[int]:
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [int(correct[:k].reshape(-1).float().sum().item()) for k in ks]


def evaluate_accuracy(model, loader) -> Accuracy:
    top1 = 0
    top5 = 0
    samples = 0
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            c1, c5 = topk_correct(logits, targets, ks=(1, 5))
            top1 += c1
            top5 += c5
            samples += targets.numel()
    if samples == 0:
        return Accuracy(0.0, 0.0, 0)
    return Accuracy(100.0 * top1 / samples, 100.0 * top5 / samples, samples)
