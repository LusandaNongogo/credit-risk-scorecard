import numpy as np
import pytest

from src.metrics import auc, calibration_table, gini, ks, psi


def test_perfect_model():
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert auc(y, s) == pytest.approx(1.0)
    assert gini(y, s) == pytest.approx(1.0)
    assert ks(y, s) == pytest.approx(1.0)


def test_reversed_model_has_negative_gini():
    y = [0, 0, 1, 1]
    s = [0.9, 0.8, 0.2, 0.1]
    assert gini(y, s) == pytest.approx(-1.0)


def test_random_scores_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    s = rng.random(20000)
    assert auc(y, s) == pytest.approx(0.5, abs=0.02)
    assert ks(y, s) < 0.03


def test_ks_hand_example():
    # goods at 1 and 3, bads at 2 and 4: gaps are 0.5, 0, 0.5, 0
    y = [0, 1, 0, 1]
    s = [1, 2, 3, 4]
    assert ks(y, s) == pytest.approx(0.5)


def test_ks_does_not_split_ties():
    y = [0, 1]
    s = [0.5, 0.5]
    assert ks(y, s) == pytest.approx(0.0)


def test_psi_zero_for_same_sample():
    x = np.arange(1000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_picks_up_a_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(500, 50, 5000)
    assert psi(base, rng.normal(500, 50, 5000)) < 0.02
    assert psi(base, rng.normal(560, 50, 5000)) > 0.25


def test_calibration_table_shape_and_counts():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = calibration_table(y, p, -p)
    assert len(t) == 10
    assert t["count"].sum() == 1000
    assert (t["predicted_bad_rate"] - t["actual_bad_rate"]).abs().max() < 0.1
