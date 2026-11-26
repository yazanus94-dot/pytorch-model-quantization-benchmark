# Calibration Experiments

These experiments use the same Imagenette validation set, pretrained ResNet18 weights, batch size, warmup, iterations, CPU thread count, and random seed. Calibration uses Imagenette training images only.

FP32 baseline:
- Top-1: 72.2%
- Top-5: 91.7%

Earlier static INT8 result using the oneDNN default `reduce_range=False`:
- Top-1: 67.9%
- Top-5: 87.8%

Root cause: the default oneDNN `HistogramObserver` qconfig uses `reduce_range=False`. On this CPU, that configuration produced inaccurate INT8 activations. Setting `reduce_range=True` restored accuracy.

| Calibration Samples | Static Top-1 | Static Top-5 | Mean Latency (ms/batch) | P95 Latency (ms) | Throughput (img/s) | Static Size (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 72.4 | 92.3 | 809.65 | 825.49 | 39.52 | 11.29 |
| 512 | 72.4 | 92.2 | 818.13 | 859.86 | 39.11 | 11.29 |
| 1024 | 71.7 | 92.2 | 827.00 | 920.44 | 38.69 | 11.29 |

Observations:
- Increasing calibration samples from 256 to 512 did not change Top-1 accuracy.
- 1024 calibration samples slightly reduced Top-1 accuracy relative to 256/512.
- Model size was unchanged across calibration sample counts.
- Latency and throughput were similar across calibration sample counts.
- The dominant accuracy issue was the oneDNN `reduce_range` setting, not calibration sample count.