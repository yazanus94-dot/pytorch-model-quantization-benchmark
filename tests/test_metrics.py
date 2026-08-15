import torch

from quant_bench.metrics import topk_correct


def test_topk_correct():
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    targets = torch.tensor([0, 2])
    top1, top5 = topk_correct(logits, targets, ks=(1, 5))
    assert top1 == 1
    assert top5 == 2
