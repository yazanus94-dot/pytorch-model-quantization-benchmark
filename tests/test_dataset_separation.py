from pathlib import Path

from PIL import Image

from quant_bench.data import make_calibration_loader, make_val_loader


def _make_dataset(root: Path) -> None:
    for split in ("train", "val"):
        class_dir = root / split / "n01440764"
        class_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(10, 20, 30)).save(class_dir / "image.png")


def test_calibration_and_validation_use_different_splits(tmp_path):
    _make_dataset(tmp_path)
    calibration = make_calibration_loader(tmp_path, batch_size=1, calibration_samples=1, num_workers=0, seed=7)
    validation = make_val_loader(tmp_path, batch_size=1, limit=1, num_workers=0)

    calibration_root = str(calibration.dataset.dataset.base.root)
    validation_root = str(validation.dataset.dataset.base.root)
    assert "train" in calibration_root
    assert "val" in validation_root
    assert calibration_root != validation_root