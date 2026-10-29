import numpy as np
import pytest

from src.woe import information_value, iv_summary, woe_table


def test_woe_matches_hand_calculation():
    # bin A: 1 bad, 3 good. bin B: 3 bad, 1 good.
    bins = ["A"] * 4 + ["B"] * 4
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    t = woe_table(bins, y).set_index("bin")

    assert t.loc["A", "woe"] == pytest.approx(np.log((1 / 4) / (3 / 4)))
    assert t.loc["B", "woe"] == pytest.approx(np.log((3 / 4) / (1 / 4)))


def test_riskier_bin_has_higher_woe():
    bins = ["low"] * 10 + ["high"] * 10
    y = [0] * 9 + [1] + [1] * 6 + [0] * 4
    t = woe_table(bins, y).set_index("bin")
    assert t.loc["high", "woe"] > 0 > t.loc["low", "woe"]


def test_iv_is_zero_when_feature_has_no_signal():
    bins = ["A", "A", "B", "B"]
    y = [0, 1, 0, 1]
    assert information_value(bins, y) == pytest.approx(0.0)


def test_iv_is_positive_and_symmetric():
    bins = ["A"] * 4 + ["B"] * 4
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    iv = information_value(bins, y)
    flipped = information_value(bins, [1 - v for v in y])
    assert iv > 0
    assert iv == pytest.approx(flipped)


def test_empty_bin_does_not_give_infinite_woe():
    bins = ["A", "A", "B", "B"]
    y = [0, 0, 1, 1]
    t = woe_table(bins, y)
    assert np.isfinite(t["woe"]).all()


def test_iv_summary_keep_and_flag():
    import pandas as pd

    fake_bins = {
        "weak": pd.DataFrame({"total_iv": [0.01]}),
        "ok": pd.DataFrame({"total_iv": [0.2]}),
        "too_good": pd.DataFrame({"total_iv": [0.7]}),
    }
    s = iv_summary(fake_bins).set_index("feature")
    assert not s.loc["weak", "keep"]
    assert s.loc["ok", "keep"] and not s.loc["ok", "flag_high"]
    assert s.loc["too_good", "flag_high"]
