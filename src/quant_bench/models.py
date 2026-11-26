from __future__ import annotations

import copy
import platform
import warnings

import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models.quantization import resnet18 as quantizable_resnet18


def build_fp32_model() -> nn.Module:
    """Quantizable ResNet18 loaded with standard pretrained FP32 weights."""
    return quantizable_resnet18(
        weights=ResNet18_Weights.DEFAULT,
        quantize=False,
    ).eval()


def choose_quantized_backend() -> str:
    supported = set(torch.backends.quantized.supported_engines)
    if not supported:
        raise RuntimeError("PyTorch reports no supported quantized backends")
    machine = platform.machine().lower()

    if machine in {"x86_64", "amd64"}:
        for candidate in ("onednn", "x86", "fbgemm"):
            if candidate in supported:
                return candidate
    for candidate in ("qnnpack", "onednn", "x86", "fbgemm"):
        if candidate in supported:
            return candidate
    return sorted(supported)[0]


def build_dynamic_int8_model(fp32_model: nn.Module) -> nn.Module:
    """Dynamic INT8 for supported layers (ResNet18: principally the final Linear)."""
    if not isinstance(fp32_model, nn.Module):
        raise TypeError("fp32_model must be a torch.nn.Module")
    from torch.ao.quantization import quantize_dynamic

    backend = choose_quantized_backend()
    torch.backends.quantized.engine = backend

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return quantize_dynamic(
            copy.deepcopy(fp32_model).eval(),
            {nn.Linear},
            dtype=torch.qint8,
            inplace=False,
        )


def build_static_int8_model(
    fp32_model: nn.Module,
    calibration_loader,
    *,
    reduce_range: bool | None = None,
) -> tuple[nn.Module, str]:
    """Post-training static INT8 using eager-mode observers + calibration."""
    if not isinstance(fp32_model, nn.Module):
        raise TypeError("fp32_model must be a torch.nn.Module")
    if calibration_loader is None:
        raise ValueError("calibration_loader must not be None")
    from torch.ao.quantization import convert, get_default_qconfig, prepare

    backend = choose_quantized_backend()
    torch.backends.quantized.engine = backend

    model = copy.deepcopy(fp32_model).eval()
    model.fuse_model(is_qat=False)
    if reduce_range is None:
        model.qconfig = get_default_qconfig(backend)
    else:
        import functools

        from torch.ao.quantization import QConfig
        from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver

        model.qconfig = QConfig(
            activation=functools.partial(HistogramObserver, reduce_range=reduce_range),
            weight=functools.partial(
                PerChannelMinMaxObserver,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            ),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepare(model, inplace=True)
        with torch.inference_mode():
            for images, _ in calibration_loader:
                model(images)
        convert(model, inplace=True)

    return model.eval(), backend
