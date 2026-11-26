from quant_bench.data import calibration_indices


def test_calibration_indices_are_reproducible():
    first = calibration_indices(100, 10, seed=7)
    second = calibration_indices(100, 10, seed=7)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


def test_calibration_indices_vary_by_seed():
    assert calibration_indices(100, 10, seed=7) != calibration_indices(100, 10, seed=11)


def test_calibration_indices_validation():
    try:
        calibration_indices(100, 0, seed=7)
    except ValueError as exc:
        assert "calibration_samples" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        calibration_indices(0, 10, seed=7)
    except ValueError as exc:
        assert "dataset_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")