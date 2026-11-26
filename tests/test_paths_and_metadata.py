from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from quant_bench.config import DEFAULT_ARTIFACT_ROOT, PROJECT_ROOT
from quant_bench.data import make_calibration_loader, make_val_loader
from quant_bench.profiler import profile_model
from quant_bench.report import environment_metadata


def test_default_artifact_root():
    assert DEFAULT_ARTIFACT_ROOT == PROJECT_ROOT / "artifacts" / "personal_benchmark"


def test_profiler_output_directory(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 2 * 2, 4))
    batch = torch.rand(2, 3, 2, 2)
    custom_dir = tmp_path / "custom_profiler"
    trace_path, csv_path = profile_model(model, batch, custom_dir, "tiny")
    assert trace_path.parent == custom_dir
    assert csv_path.parent == custom_dir


def test_calibration_samples_validation():
    try:
        make_calibration_loader(Path("data/imagenette2-160"), batch_size=2, calibration_samples=0, num_workers=0)
    except ValueError as exc:
        assert "calibration_samples" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid calibration sample count")


def test_environment_metadata_provenance():
    metadata = environment_metadata()
    assert metadata["source"] == "fork_local"