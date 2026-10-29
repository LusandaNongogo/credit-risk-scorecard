import numpy as np
import pytest

from src.scorecard import pd_from_score, score_from_pd


def test_base_odds_give_base_score():
    pd_at_50_to_1 = 1 / 51
    assert score_from_pd(pd_at_50_to_1) == pytest.approx(600)


def test_pdo_doubles_odds():
    s = score_from_pd(1 / 51)
    s2 = score_from_pd(1 / 101)  # 100:1
    assert s2 - s == pytest.approx(20)


def test_round_trip():
    p = np.array([0.01, 0.1, 0.3, 0.7])
    assert pd_from_score(score_from_pd(p)) == pytest.approx(p)
