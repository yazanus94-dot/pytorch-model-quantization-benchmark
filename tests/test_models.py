import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.quantization import resnet18 as quantizable_resnet18

from quant_bench.models import (
    build_dynamic_int8_model,
    build_static_int8_model,
    choose_quantized_backend,
)


def _make_loader():
    images = torch.rand(4, 3, 224, 224)
    targets = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(images, targets), batch_size=2)


def test_backend_is_supported():
    backend = choose_quantized_backend()
    assert backend in torch.backends.quantized.supported_engines


def test_dynamic_and_static_conversion():
    fp32 = quantizable_resnet18(weights=None, quantize=False).eval()
    dynamic = build_dynamic_int8_model(fp32)

    loader = _make_loader()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        static, backend = build_static_int8_model(fp32, loader)

    batch, _ = next(iter(loader))
    for model in (fp32, dynamic, static):
        with torch.inference_mode():
            output = model(batch)
        assert tuple(output.shape) == (2, 1000)
        assert torch.isfinite(output).all().item()

    assert backend in torch.backends.quantized.supported_engines