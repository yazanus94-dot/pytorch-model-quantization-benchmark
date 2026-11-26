from __future__ import annotations

import gc
import io
import statistics
import threading
import time
from dataclasses import asdict, dataclass

import psutil
import torch

from .metrics import evaluate_accuracy


@dataclass
class BenchmarkResult:
    variant: str
    dtype: str
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    throughput_images_per_s: float
    serialized_size_mb: float
    peak_process_rss_mb: float
    top1_accuracy: float | None
    top5_accuracy: float | None
    accuracy_samples: int
    backend: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def serialized_model_size_mb(model) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getbuffer()) / (1024 ** 2)


def _rss_sampler(stop: threading.Event, samples: list[int], interval_s: float = 0.005) -> None:
    process = psutil.Process()
    while not stop.is_set():
        samples.append(process.memory_info().rss)
        time.sleep(interval_s)
    samples.append(process.memory_info().rss)


def measure_latency_and_rss(
    model,
    sample_batch: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], float]:
    if not callable(model):
        raise TypeError("model must be callable")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if not torch.is_tensor(sample_batch) or sample_batch.ndim < 1:
        raise ValueError("sample_batch must be a non-empty tensor")
    model.eval()
    gc.collect()

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample_batch)

    rss_samples: list[int] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_rss_sampler, args=(stop, rss_samples), daemon=True)
    sampler.start()

    times_ms: list[float] = []
    try:
        with torch.inference_mode():
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(sample_batch)
                end = time.perf_counter_ns()
                times_ms.append((end - start) / 1e6)
    finally:
        stop.set()
        sampler.join()

    peak_rss_mb = (max(rss_samples) if rss_samples else psutil.Process().memory_info().rss) / (1024 ** 2)
    return times_ms, peak_rss_mb


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if q < 0.0 or q > 1.0:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def benchmark_model(
    name: str,
    dtype: str,
    model,
    val_loader,
    *,
    warmup: int,
    iterations: int,
    backend: str | None = None,
    accuracy: bool = True,
) -> BenchmarkResult:
    try:
        sample_batch, _ = next(iter(val_loader))
    except StopIteration as exc:
        raise ValueError("val_loader must contain at least one batch") from exc
    times_ms, peak_rss_mb = measure_latency_and_rss(
        model, sample_batch, warmup=warmup, iterations=iterations
    )
    mean_ms = statistics.fmean(times_ms)
    batch_size = sample_batch.shape[0]
    throughput = batch_size / (mean_ms / 1000.0)

    acc = evaluate_accuracy(model, val_loader) if accuracy else None
    return BenchmarkResult(
        variant=name,
        dtype=dtype,
        latency_ms_mean=mean_ms,
        latency_ms_p50=_percentile(times_ms, 0.50),
        latency_ms_p95=_percentile(times_ms, 0.95),
        throughput_images_per_s=throughput,
        serialized_size_mb=serialized_model_size_mb(model),
        peak_process_rss_mb=peak_rss_mb,
        top1_accuracy=acc.top1 if acc else None,
        top5_accuracy=acc.top5 if acc else None,
        accuracy_samples=acc.samples if acc else 0,
        backend=backend,
    )
