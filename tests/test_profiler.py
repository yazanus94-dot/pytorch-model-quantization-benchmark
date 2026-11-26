import tempfile
from pathlib import Path

import torch
from torchvision.models.quantization import resnet18 as quantizable_resnet18

from quant_bench.profiler import profile_model


def test_profile_model_writes_trace_and_csv():
    model = quantizable_resnet18(weights=None, quantize=False).eval()
    batch = torch.rand(2, 3, 224, 224)

    with tempfile.TemporaryDirectory() as directory:
        trace_path, csv_path = profile_model(model, batch, Path(directory), "fp32", row_limit=3)

        assert trace_path.exists()
        assert trace_path.stat().st_size > 0
        assert csv_path.exists()
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "op,self_cpu_time_total_us,cpu_time_total_us,self_cpu_memory_bytes,cpu_memory_bytes,calls"
        assert len(lines) >= 2