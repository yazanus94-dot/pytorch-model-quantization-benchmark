# Model Quantization Benchmark

A reproducible CPU inference benchmark for **FP32 vs dynamic INT8 vs post-training static INT8 ResNet18** using PyTorch. It measures latency, throughput, serialized model size, process RSS, top-1/top-5 accuracy, and operator-level CPU/memory behavior with `torch.profiler`.

The project intentionally writes benchmark numbers only after a real run. `RESULTS.md` is generated from measurements on your machine, so the repository does not contain invented performance claims.


## Measured benchmark result

On a 1,000-image Imagenette evaluation using Apple Silicon CPU / QNNPACK, calibrated static INT8 reduced ResNet18 serialized size from **44.67 MB to 11.19 MB (~75%)** and peak process RSS from **1,074.5 MB to 867.1 MB (~19%)**. Throughput increased from **25.95 to 26.63 images/s**, while Top-1 accuracy remained within **0.7 percentage points** of the FP32 baseline.

Dynamic INT8 produced substantially less benefit because ResNet18 is convolution-heavy and dynamic quantization primarily affected supported weight-heavy modules such as the final linear layer.

See [`RESULTS.md`](RESULTS.md) for the full measurements and methodology.

## What it demonstrates

- **Model compression:** dynamic and static INT8 quantization
- **Calibration:** observer-driven post-training static quantization
- **Performance engineering:** warmup, repeated CPU timing, throughput, model storage footprint, process RSS
- **Accuracy/performance tradeoffs:** measured top-1/top-5 accuracy delta vs FP32
- **Profiling:** Chrome trace export plus top CPU/memory operations from `torch.profiler`
- **Reproducibility:** fixed dataset slice, seed, CPU thread count, raw JSON/CSV artifacts, machine metadata

## Benchmark design

| Variant | What changes | Why it is included |
|---|---|---|
| FP32 | Pretrained ResNet18 baseline | Reference latency/size/accuracy |
| Dynamic INT8 | Supported weight-heavy modules are dynamically quantized | Shows weight-only/dynamic quantization behavior |
| Static INT8 | Calibrated activations + weights quantized through the CNN path | Main deployment-oriented INT8 comparison |

**Dataset:** Imagenette2-160 validation data, mapped back to the original ImageNet-1K class indices. This makes the benchmark small enough for a laptop while still computing real classification accuracy for a pretrained ImageNet model.

## Setup

Recommended: Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Apple Silicon, quantized CPU execution generally uses a supported backend such as QNNPACK when available. On x86, the runner prefers x86/FBGEMM-style backends. The selected backend is recorded in `RESULTS.md`.

## Fast correctness check

This does **not** download pretrained weights or Imagenette. It just validates that all three conversion/inference paths work:

```bash
quant-bench smoke
```

Or:

```bash
make smoke
```

## Run the real benchmark

A good laptop-sized run:

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

For a faster first pass:

```bash
quant-bench run --download --limit 250 --calibration-samples 64 --batch-size 16 --warmup 3 --iterations 10
```

## Outputs

After a successful run:

```text
RESULTS.md                         # recruiter-readable measured summary
artifacts/benchmark_results.json   # raw results + machine metadata
artifacts/benchmark_results.csv    # table-friendly raw results
artifacts/profiler/fp32_trace.json
artifacts/profiler/fp32_top_ops.csv
artifacts/profiler/dynamic_int8_trace.json
artifacts/profiler/dynamic_int8_top_ops.csv
artifacts/profiler/static_int8_trace.json
artifacts/profiler/static_int8_top_ops.csv
```

The generated table includes:

- mean and p95 batch latency
- images/second throughput
- serialized model size
- peak process RSS during timed inference
- top-1 accuracy
- accuracy delta vs FP32

## Read a profiler trace

Open any `*_trace.json` in a Chrome-trace-compatible viewer such as Perfetto. The companion `*_top_ops.csv` is easier to scan in GitHub or a spreadsheet and is sorted by self CPU time.

Useful questions to answer in your write-up:

1. Which operators dominate FP32 inference?
2. Which operations become quantized in static INT8?
3. Does lower serialized model size translate to lower process RSS?
4. Is latency improvement consistent at your chosen batch size?
5. How much top-1 accuracy is lost, if any, on the same evaluation slice?

## Why both dynamic and static quantization?

Dynamic quantization is intentionally included even though it is not the strongest technique for a convolution-heavy network. For ResNet18, dynamic quantization mostly affects supported modules such as the final `Linear` layer, so its latency improvement may be small. Static quantization is the more meaningful CNN comparison because calibration enables integer execution across a larger part of the network.

That difference is useful in an interview: it shows you understand that "INT8" is not a single implementation and that model architecture determines which quantization method is appropriate.

## Tests

```bash
pytest -q
```

## Repository structure

```text
src/quant_bench/
  benchmark.py   # latency, throughput, model size, RSS
  cli.py         # command-line entrypoint
  config.py      # benchmark defaults + Imagenette mapping
  data.py        # download/load Imagenette and map labels
  metrics.py     # top-k accuracy
  models.py      # FP32 + dynamic/static quantization
  profiler.py    # torch.profiler trace + top-op CSV
  report.py      # raw outputs + RESULTS.md generation

tests/
RESULTS.md
VALIDATION.md
```

## API note

This implementation uses `torch.ao.quantization` eager APIs to make the requested dynamic/static quantization workflow explicit. PyTorch has been moving active quantization development toward **torchao/PT2E**, so treat the eager path here as a transparent benchmark implementation rather than a claim that it is the newest production API. A natural follow-up project extension is to add a `torchao` PT2E backend and compare it against the eager baseline.
