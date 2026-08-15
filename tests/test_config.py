from quant_bench.config import IMAGENETTE_TO_IMAGENET_INDEX


def test_imagenette_mapping_has_ten_classes():
    assert len(IMAGENETTE_TO_IMAGENET_INDEX) == 10
    assert IMAGENETTE_TO_IMAGENET_INDEX["n01440764"] == 0
    assert IMAGENETTE_TO_IMAGENET_INDEX["n03888257"] == 701
