from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "personal_benchmark"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_ARCHIVE = "imagenette2-160.tgz"
IMAGENETTE_DIRNAME = "imagenette2-160"

# Imagenette WordNet IDs mapped to the corresponding ImageNet-1K output index.
IMAGENETTE_TO_IMAGENET_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int = 32
    warmup: int = 10
    iterations: int = 50
    limit: int | None = 1000
    calibration_samples: int = 256
    num_workers: int = 0
    seed: int = 7
