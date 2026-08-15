from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


def profile_model(model, sample_batch: torch.Tensor, output_dir: Path, name: str, row_limit: int = 20) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{name}_trace.json"
    csv_path = output_dir / f"{name}_top_ops.csv"

    model.eval()
    with torch.inference_mode():
        # One warmup pass keeps initialization out of the trace where possible.
        model(sample_batch)
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(5):
                model(sample_batch)

    prof.export_chrome_trace(str(trace_path))
    events = list(prof.key_averages())
    events.sort(key=lambda event: float(getattr(event, "self_cpu_time_total", 0.0)), reverse=True)

    fields = [
        "op",
        "self_cpu_time_total_us",
        "cpu_time_total_us",
        "self_cpu_memory_bytes",
        "cpu_memory_bytes",
        "calls",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for event in events[:row_limit]:
            writer.writerow(
                {
                    "op": event.key,
                    "self_cpu_time_total_us": round(float(getattr(event, "self_cpu_time_total", 0.0)), 3),
                    "cpu_time_total_us": round(float(getattr(event, "cpu_time_total", 0.0)), 3),
                    "self_cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
                    "cpu_memory_bytes": int(getattr(event, "cpu_memory_usage", 0)),
                    "calls": int(getattr(event, "count", 0)),
                }
            )

    return trace_path, csv_path
