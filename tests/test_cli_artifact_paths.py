import os
from pathlib import Path

import torch

from quant_bench.benchmark import BenchmarkResult
from quant_bench.cli import _run
from quant_bench.config import DEFAULT_ARTIFACT_ROOT


def _make_args(artifact_root, tmp_path, profile=False):
    data_root = tmp_path / "data"
    (data_root / "imagenette2-160" / "val").mkdir(parents=True, exist_ok=True)
    return type(
        "Args",
        (),
        {
            "data_root": str(data_root),
            "artifact_root": str(artifact_root),
            "download": False,
            "limit": 2,
            "calibration_samples": 2,
            "batch_size": 2,
            "warmup": 1,
            "iterations": 1,
            "num_workers": 0,
            "threads": 1,
            "seed": 7,
            "profile": profile,
            "reduce_range": True,
        },
    )()


def _patch_cli(monkeypatch, captured):
    import quant_bench.cli as cli

    monkeypatch.setattr(cli, "make_val_loader", lambda *a, **k: [(torch.rand(2, 3, 2, 2), torch.zeros(2, dtype=torch.long))])
    monkeypatch.setattr(cli, "make_calibration_loader", lambda *a, **k: None)
    monkeypatch.setattr(cli, "dataset_fingerprint", lambda *a, **k: "fingerprint")
    monkeypatch.setattr(cli, "build_fp32_model", lambda: torch.nn.Identity())
    monkeypatch.setattr(cli, "build_dynamic_int8_model", lambda model: model)
    monkeypatch.setattr(cli, "build_static_int8_model", lambda *a, **k: (torch.nn.Identity(), "onednn"))

    def fake_benchmark(*a, **k):
        return BenchmarkResult(
            variant="fp32",
            dtype="float32",
            latency_ms_mean=1.0,
            latency_ms_p50=1.0,
            latency_ms_p95=1.0,
            throughput_images_per_s=1.0,
            serialized_size_mb=1.0,
            peak_process_rss_mb=1.0,
            top1_accuracy=1.0,
            top5_accuracy=1.0,
            accuracy_samples=2,
        )

    monkeypatch.setattr(cli, "benchmark_model", fake_benchmark)

    def fake_profile(model, batch, output_dir, name):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        captured["profile_dir"] = output_dir
        return output_dir / f"{name}_trace.json", output_dir / f"{name}_top_ops.csv"

    monkeypatch.setattr(cli, "profile_model", fake_profile)

    def fake_write(results, artifact_dir, metadata):
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        captured["artifact_dir"] = artifact_dir
        return artifact_dir / "benchmark_results.json", artifact_dir / "benchmark_results.csv"

    monkeypatch.setattr(cli, "write_machine_readable", fake_write)

    def fake_results_md(path, results, metadata):
        path = Path(path)
        captured["results_md"] = path

    monkeypatch.setattr(cli, "write_results_md", fake_results_md)


def test_default_artifact_root(monkeypatch, tmp_path):
    captured = {}
    _patch_cli(monkeypatch, captured)
    args = _make_args(DEFAULT_ARTIFACT_ROOT, tmp_path, profile=True)
    assert _run(args) == 0
    assert captured["artifact_dir"] == DEFAULT_ARTIFACT_ROOT.resolve()
    assert captured["results_md"] == DEFAULT_ARTIFACT_ROOT.resolve() / "RESULTS.md"
    assert captured["profile_dir"] == DEFAULT_ARTIFACT_ROOT.resolve() / "profiler"


def test_absolute_custom_artifact_root(monkeypatch, tmp_path):
    captured = {}
    _patch_cli(monkeypatch, captured)
    custom = tmp_path / "absolute_output"
    args = _make_args(custom, tmp_path, profile=True)
    assert _run(args) == 0
    assert captured["artifact_dir"] == custom.resolve()
    assert captured["results_md"] == custom.resolve() / "RESULTS.md"
    assert captured["profile_dir"] == custom.resolve() / "profiler"


def test_relative_custom_artifact_root(monkeypatch, tmp_path):
    captured = {}
    _patch_cli(monkeypatch, captured)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    args = _make_args(Path("relative_output"), tmp_path, profile=True)
    expected = (work / "relative_output").resolve()
    assert _run(args) == 0
    assert captured["artifact_dir"] == expected
    assert captured["results_md"] == expected / "RESULTS.md"
    assert captured["profile_dir"] == expected / "profiler"


def test_working_directory_independence(monkeypatch, tmp_path):
    captured = {}
    _patch_cli(monkeypatch, captured)
    work = tmp_path / "elsewhere"
    work.mkdir()
    monkeypatch.chdir(work)
    args = _make_args(DEFAULT_ARTIFACT_ROOT, tmp_path, profile=False)
    assert _run(args) == 0
    assert captured["artifact_dir"] == DEFAULT_ARTIFACT_ROOT.resolve()
    assert captured["results_md"] == DEFAULT_ARTIFACT_ROOT.resolve() / "RESULTS.md"