from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Accuracy:
    top1: float
    top5: float
    samples: int


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> list[int]:
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor")
    if targets.ndim != 1 or targets.numel() != logits.shape[0]:
        raise ValueError("targets must be a 1D tensor with one entry per logits row")
    ks = tuple(ks)
    if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if max(ks) > logits.shape[1]:
        raise ValueError("k cannot exceed the number of logits columns")

    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [int(correct[:k].reshape(-1).float().sum().item()) for k in ks]


def accuracy_delta_pp(reference_accuracy: float | None, measured_accuracy: float | None) -> float | None:
    """Return an accuracy delta in percentage points, or None if either input is missing."""
    if reference_accuracy is None or measured_accuracy is None:
        return None
    return measured_accuracy - reference_accuracy


def evaluate_accuracy(model, loader) -> Accuracy:
    if not callable(model):
        raise TypeError("model must be callable")
    if loader is None:
        raise ValueError("loader must not be None")

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
