# PyTorch Model Quantization Benchmark

A reproducible CPU inference benchmark comparing **FP32 vs dynamic INT8 vs post-training static INT8 ResNet18** using PyTorch. It measures latency, throughput, serialized model size, process RSS, top-1/top-5 accuracy, and operator-level CPU/memory behavior with `torch.profiler`.

The project writes benchmark numbers only after a real run. `RESULTS.md` is generated from measurements on your machine, so the repository contains no invented performance claims.

## Technical overview

This repository implements a complete quantization benchmarking pipeline:

- **Model quantization:** FP32 baseline, dynamic INT8 (weight-only for supported modules), post-training static INT8 with calibration.
- **Performance benchmarking:** CPU inference latency (mean, p50, p95), throughput (images/s), serialized model size, peak process RSS.
- **Model evaluation:** Top-1 and top-5 accuracy, accuracy deltas vs FP32 baseline.
- **Profiling:** PyTorch Profiler with CPU operator analysis, memory profiling, Chrome trace / Perfetto-compatible output.
- **Reproducibility:** Fixed dataset slice, seed, CPU thread count, raw JSON/CSV artifacts, machine metadata.

## Main features

- Three quantization paths: FP32, Dynamic INT8, Static INT8
- Calibration workflow with observer-driven post-training static quantization
- Warmup + repeated timed execution with `torch.inference_mode()`
- Configurable CPU thread count for reproducibility
- Process-wide peak RSS sampling during timed inference
- Top-1 / top-5 accuracy on real Imagenette validation data
- Chrome trace export + top-op CSV for Perfetto analysis
- Raw JSON/CSV artifacts + auto-generated `RESULTS.md`
- Unit tests for config, metrics, and report generation
- GitHub Actions CI for unit tests and smoke test

## Architecture overview

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The package is organized as:

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

## Technologies and dependencies

- Python >= 3.10
- PyTorch >= 2.6
- torchvision >= 0.21
- Pillow >= 10.0
- psutil >= 5.9
- pytest >= 8.0 (dev)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Apple Silicon, quantized CPU execution generally uses QNNPACK. On x86, the runner prefers x86/FBGEMM-style backends. The selected backend is recorded in `RESULTS.md`.

## Execution instructions

### Fast correctness check (no download)

```bash
quant-bench smoke
# or
make smoke
```

This validates that all three conversion/inference paths work without downloading pretrained weights or Imagenette.

### Run the real benchmark

```bash
quant-bench run \
  --download \
  --limit 1000 \
  --calibration-samples 256 \
  --batch-size 32 \
  --warmup 10 \
  --iterations 50 \
  --threads 1 \
  --profile
```

Faster first pass:

```bash
quant-bench run --download --limit 250 --calibration-samples 64 --batch-size 16 --warmup 3 --iterations 10
```

## Quantization methodology

| Variant | What changes | Why it is included |
|---|---|---|
| FP32 | Pretrained ResNet18 baseline | Reference latency/size/accuracy |
| Dynamic INT8 | Supported weight-heavy modules dynamically quantized | Shows weight-only/dynamic quantization behavior |
| Static INT8 | Calibrated activations + weights quantized through CNN path | Main deployment-oriented INT8 comparison |

**Dataset:** Imagenette2-160 validation data, mapped back to original ImageNet-1K class indices. Small enough for a laptop while still computing real classification accuracy.

**Calibration:** Observer-driven post-training static quantization using a deterministic sample of Imagenette training images.

## Benchmarking methodology

See [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md).

- Same pretrained ResNet18 weights and preprocessing for each variant.
- Same validation slice, batch size, CPU thread count, warmup count, timed iteration count.
- Timing performed after warmup under `torch.inference_mode()`.
- Serialized size = byte size of `torch.save(model.state_dict())`.
- Peak RSS = max process resident memory during timed inference (process-wide, includes runtime overhead).
- Accuracy on Imagenette validation slice using original ImageNet-1K output indices.

## Results and comparison tables

`RESULTS.md` is generated from actual measurements. See the current file for the last run's numbers. Re-run to regenerate on your hardware.

## Accuracy versus compression analysis

Static INT8 typically reduces serialized model size ~75% (44.67 MB → 11.19 MB upstream) with accuracy within ~0.7 pp of FP32. Dynamic INT8 has limited leverage on ResNet18 because convolution dominates compute and dynamic quantization targets modules such as `Linear`.

## Profiling instructions

When `--profile` is enabled, Chrome traces and top-op CSVs are written under `artifacts/profiler/`.

```text
artifacts/profiler/fp32_trace.json
artifacts/profiler/fp32_top_ops.csv
...
```

Open any `*_trace.json` in Perfetto or Chrome tracing. The companion `*_top_ops.csv` is sorted by self CPU time.

Useful analysis questions:
1. Which operators dominate FP32 inference?
2. Which operations become quantized in static INT8?
3. Does lower serialized model size translate to lower process RSS?
4. Is latency improvement consistent at your chosen batch size?
5. How much top-1 accuracy is lost, if any?

## Testing instructions

```bash
pytest -q
```

Tests cover: Imagenette label mapping, top-k accuracy counting, report markdown generation.

## Limitations and future development

1. Imagenette is a 10-class subset of ImageNet; measured accuracy is not official ImageNet-1K validation accuracy.
2. Dynamic quantization has limited leverage on ResNet18 (convolution-heavy).
3. Peak RSS is process-wide, not isolated tensor memory.
4. Eager `torch.ao.quantization` is a legacy API family; PyTorch is moving toward torchao/PT2E. A natural extension is adding a `torchao` PT2E backend for comparison.
5. No LICENSE file existed upstream; redistribution follows GitHub's fork mechanism.

## Upstream project attribution

This repository is a fork of [Gopal3746/model-quantization](https://github.com/Gopal3746/model-quantization). Original authorship and Git history are preserved. Measurements originally reported upstream (Apple Silicon / QNNPACK) are retained only as reference in [docs/UPSTREAM_REFERENCE.md](docs/UPSTREAM_REFERENCE.md) and are not presented as results of this fork.

See also: [docs/UPSTREAM.md](docs/UPSTREAM.md), [VALIDATION.md](VALIDATION.md) (upstream).

## Outputs

After a successful run:

```text
RESULTS.md                         # measured summary
artifacts/benchmark_results.json   # raw results + machine metadata
artifacts/benchmark_results.csv    # table-friendly raw results
artifacts/profiler/*_trace.json
artifacts/profiler/*_top_ops.csv
```

## Repository structure

```
src/quant_bench/   # package source
tests/             # unit tests
docs/              # architecture, methodology, upstream
artifacts/         # generated benchmark outputs
.github/workflows/ # CI
```
