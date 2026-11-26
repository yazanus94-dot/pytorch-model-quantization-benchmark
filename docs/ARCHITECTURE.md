# Architecture

## Package layout
```
src/quant_bench/
  config.py      # defaults + Imagenette label mapping
  data.py        # download/load + label remapping
  models.py      # FP32 + dynamic/static quantization
  benchmark.py   # timing, throughput, size, RSS
  metrics.py     # top-k accuracy
  profiler.py    # torch.profiler traces + CSV
  report.py      # JSON/CSV/Markdown output
  cli.py         # entrypoint
```

## Data flow
1. `cli._run` downloads (optional) Imagenette, builds val + calibration loaders.
2. `models` builds FP32, dynamic INT8, static INT8 (calibrated).
3. `benchmark.benchmark_model` measures latency/throughput/size/RSS/accuracy.
4. `profiler.profile_model` (optional) writes traces + top-op CSVs.
5. `report` writes JSON/CSV/RESULTS.md with machine metadata.
