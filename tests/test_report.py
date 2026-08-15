from quant_bench.benchmark import BenchmarkResult
from quant_bench.report import render_results_markdown


def _result(name, acc, latency, size):
    return BenchmarkResult(
        variant=name,
        dtype="int8" if "int8" in name else "float32",
        latency_ms_mean=latency,
        latency_ms_p50=latency,
        latency_ms_p95=latency * 1.1,
        throughput_images_per_s=100.0,
        serialized_size_mb=size,
        peak_process_rss_mb=200.0,
        top1_accuracy=acc,
        top5_accuracy=99.0,
        accuracy_samples=100,
    )


def test_report_contains_accuracy_delta():
    md = render_results_markdown(
        [_result("fp32", 90.0, 10.0, 40.0), _result("static_int8", 89.5, 6.0, 12.0)],
        {"batch_size": 32},
    )
    assert "-0.50" in md
    assert "static_int8" in md
    assert "Peak process RSS" in md
