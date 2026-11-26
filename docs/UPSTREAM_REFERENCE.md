# Upstream Reference Measurements

The following numbers were reported by the upstream project on Apple Silicon CPU / QNNPACK and are reproduced here for reference only. They are NOT measurements from this fork owner's hardware.

| Variant | Throughput (img/s) | Size (MB) | Peak RSS (MB) | Top-1 (%) |
|---|---:|---:|---:|---:|
| fp32 | 25.95 | 44.67 | 1074.50 | 72.20 |
| dynamic_int8 | 25.36 | 43.20 | 1081.00 | 72.50 |
| static_int8 | 26.63 | 11.19 | 867.06 | 72.90 |

Generate your own results with `quant-bench run`.
