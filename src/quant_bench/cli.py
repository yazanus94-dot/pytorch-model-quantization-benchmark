from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark import benchmark_model
from .config import BenchmarkConfig, DEFAULT_ARTIFACT_ROOT, DEFAULT_DATA_ROOT, PROJECT_ROOT
from .data import dataset_fingerprint, download_imagenette, make_calibration_loader, make_val_loader
from .models import build_dynamic_int8_model, build_fp32_model, build_static_int8_model
from .profiler import profile_model
from .report import environment_metadata, write_machine_readable, write_results_md


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _run(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    data_root = Path(args.data_root).resolve()
    dataset_root = data_root / "imagenette2-160"
    if args.download or not (dataset_root / "val").is_dir():
        dataset_root = download_imagenette(data_root)
    if not (dataset_root / "val").is_dir():
        raise FileNotFoundError(
            f"Imagenette not found at {dataset_root}. Re-run with --download."
        )

    val_loader = make_val_loader(
        dataset_root,
        batch_size=args.batch_size,
        limit=args.limit,
        num_workers=args.num_workers,
    )
    calibration_loader = make_calibration_loader(
        dataset_root,
        batch_size=args.batch_size,
        calibration_samples=args.calibration_samples,
        num_workers=args.num_workers,
    )

    print("Loading pretrained FP32 ResNet18...")
    fp32 = build_fp32_model()
    print("Building dynamic INT8 model...")
    dynamic = build_dynamic_int8_model(fp32)
    print("Calibrating and building static INT8 model...")
    static, backend = build_static_int8_model(fp32, calibration_loader)

    variants = [
        ("fp32", "float32", fp32, None),
        ("dynamic_int8", "qint8 weights / fp32 activations", dynamic, torch.backends.quantized.engine),
        ("static_int8", "int8", static, backend),
    ]

    results = []
    for name, dtype, model, model_backend in variants:
        print(f"Benchmarking {name}...")
        result = benchmark_model(
            name,
            dtype,
            model,
            val_loader,
            warmup=args.warmup,
            iterations=args.iterations,
            backend=model_backend,
            accuracy=True,
        )
        results.append(result)

        if args.profile:
            sample_batch, _ = next(iter(val_loader))
            trace, ops = profile_model(model, sample_batch, Path(args.artifact_root) / "profiler", name)
            print(f"  profiler trace: {trace}")
            print(f"  top ops:        {ops}")

    metadata = environment_metadata(
        dataset="Imagenette2-160 validation",
        dataset_fingerprint=dataset_fingerprint(dataset_root),
        evaluation_limit=args.limit,
        calibration_samples=args.calibration_samples,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        quantized_backend=backend,
    )
    artifact_root = Path(args.artifact_root).resolve()
    json_path, csv_path = write_machine_readable(results, artifact_root, metadata)
    results_md = PROJECT_ROOT / "RESULTS.md"
    write_results_md(results_md, results, metadata)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {results_md}")
    return 0


def _smoke(args: argparse.Namespace) -> int:
    """No dataset/pretrained-weight download: validates quantization + inference paths."""
    import warnings
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models.quantization import resnet18 as quantizable_resnet18

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    images = torch.rand(4, 3, 224, 224)
    targets = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)

    fp32 = quantizable_resnet18(weights=None, quantize=False).eval()
    dynamic = build_dynamic_int8_model(fp32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        static, backend = build_static_int8_model(fp32, loader)

    summary = []
    for name, model in [("fp32", fp32), ("dynamic_int8", dynamic), ("static_int8", static)]:
        batch, _ = next(iter(loader))
        with torch.inference_mode():
            out = model(batch)
        summary.append({"variant": name, "output_shape": list(out.shape), "finite": bool(torch.isfinite(out).all())})
    print(json.dumps({"backend": backend, "checks": summary}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ResNet18 FP32/INT8 quantization benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the real Imagenette benchmark")
    run.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    run.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    run.add_argument("--download", action="store_true", help="Download Imagenette if needed")
    run.add_argument("--limit", type=_positive_int, default=1000, help="Validation images to evaluate")
    run.add_argument("--calibration-samples", type=_positive_int, default=256)
    run.add_argument("--batch-size", type=_positive_int, default=32)
    run.add_argument("--warmup", type=_positive_int, default=10)
    run.add_argument("--iterations", type=_positive_int, default=50)
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--threads", type=_positive_int, default=1, help="CPU intra-op threads for reproducibility")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--profile", action="store_true")
    run.set_defaults(func=_run)

    smoke = sub.add_parser("smoke", help="Validate model conversion without downloads")
    smoke.add_argument("--threads", type=_positive_int, default=1)
    smoke.add_argument("--warmup", type=_positive_int, default=1)
    smoke.add_argument("--iterations", type=_positive_int, default=2)
    smoke.set_defaults(func=_smoke)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
