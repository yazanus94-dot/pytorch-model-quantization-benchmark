import torch

from quant_bench.benchmark import _percentile
from quant_bench.config import BenchmarkConfig
from quant_bench.metrics import accuracy_delta_pp, topk_correct


def test_benchmark_config_defaults():
    config = BenchmarkConfig()
    assert config.batch_size == 32
    assert config.warmup == 10
    assert config.iterations == 50
    assert config.limit == 1000
    assert config.calibration_samples == 256


def test_percentile_and_topk_helpers():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    logits = torch.tensor([[5.0, 4.0, 1.0], [1.0, 2.0, 3.0]])
    targets = torch.tensor([0, 2])
    assert topk_correct(logits, targets, ks=(1, 3)) == [2, 2]
    assert accuracy_delta_pp(90.0, 89.5) == -0.5
    assert accuracy_delta_pp(None, 89.5) is None