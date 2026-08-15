from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights

from .config import (
    DEFAULT_DATA_ROOT,
    IMAGENETTE_ARCHIVE,
    IMAGENETTE_DIRNAME,
    IMAGENETTE_TO_IMAGENET_INDEX,
    IMAGENETTE_URL,
)


class ImagenetteImageNetLabels(Dataset):
    """ImageFolder wrapper that returns ImageNet-1K target indices."""

    def __init__(self, root: Path, transform: Callable):
        self.base = ImageFolder(root=str(root), transform=transform)
        unknown = set(self.base.class_to_idx) - set(IMAGENETTE_TO_IMAGENET_INDEX)
        if unknown:
            raise ValueError(f"Unexpected Imagenette classes: {sorted(unknown)}")

        self.class_to_imagenet = {
            local_idx: IMAGENETTE_TO_IMAGENET_INDEX[wnid]
            for wnid, local_idx in self.base.class_to_idx.items()
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, local_target = self.base[index]
        return image, self.class_to_imagenet[local_target]


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    root = path.resolve()
    for member in tar.getmembers():
        destination = (path / member.name).resolve()
        if root not in destination.parents and destination != root:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    tar.extractall(path)


def download_imagenette(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_root = data_root / IMAGENETTE_DIRNAME
    if (dataset_root / "val").is_dir():
        return dataset_root

    archive = data_root / IMAGENETTE_ARCHIVE
    if not archive.exists():
        print(f"Downloading Imagenette from {IMAGENETTE_URL}")
        urllib.request.urlretrieve(IMAGENETTE_URL, archive)

    print(f"Extracting {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extract(tar, data_root)
    return dataset_root


def dataset_fingerprint(dataset_root: Path) -> str:
    """Stable lightweight fingerprint based on validation relative paths."""
    val_root = dataset_root / "val"
    paths = sorted(str(p.relative_to(val_root)) for p in val_root.rglob("*") if p.is_file())
    digest = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()
    return digest[:16]


def make_val_loader(
    dataset_root: Path,
    *,
    batch_size: int,
    limit: int | None,
    num_workers: int,
) -> DataLoader:
    transform = ResNet18_Weights.DEFAULT.transforms()
    dataset: Dataset = ImagenetteImageNetLabels(dataset_root / "val", transform)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def make_calibration_loader(
    dataset_root: Path,
    *,
    batch_size: int,
    calibration_samples: int,
    num_workers: int,
) -> DataLoader:
    transform = ResNet18_Weights.DEFAULT.transforms()
    train = ImagenetteImageNetLabels(dataset_root / "train", transform)
    generator = torch.Generator().manual_seed(7)
    count = min(calibration_samples, len(train))
    indices = torch.randperm(len(train), generator=generator)[:count].tolist()
    subset = Subset(train, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
