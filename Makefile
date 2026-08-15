PYTHON ?= python

.PHONY: install test smoke benchmark clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m quant_bench.cli smoke --iterations 2 --warmup 1

benchmark:
	$(PYTHON) -m quant_bench.cli run --download --limit 1000 --calibration-samples 256 --batch-size 32 --warmup 10 --iterations 50 --profile

clean:
	rm -f artifacts/benchmark_results.json artifacts/benchmark_results.csv
	rm -f artifacts/profiler/*.json artifacts/profiler/*.csv
