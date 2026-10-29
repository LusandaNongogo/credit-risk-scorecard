import pytest

from src.ecl import assign_stage, ecl, lifetime_pd, marginal_pds


def test_stage_1_is_twelve_month():
    assert ecl(0.05, 0.45, 10000, stage=1, remaining_years=5) == pytest.approx(225)


def test_stage_2_higher_than_stage_1():
    s1 = ecl(0.05, 0.45, 10000, stage=1, remaining_years=5)
    s2 = ecl(0.05, 0.45, 10000, stage=2, remaining_years=5)
    assert s2 > s1


def test_stage_3_is_full_loss_given_default():
    assert ecl(0.05, 0.45, 10000, stage=3) == pytest.approx(4500)


def test_lifetime_pd():
    # 1 - 0.9^3
    assert lifetime_pd(0.1, 3) == pytest.approx(1 - 0.9 ** 3)
    assert len(marginal_pds(0.1, 2.5)) == 3


def test_discounting_lowers_ecl():
    plain = ecl(0.05, 0.45, 10000, stage=2, remaining_years=3)
    disc = ecl(0.05, 0.45, 10000, stage=2, remaining_years=3, eir=0.15)
    assert disc < plain


def test_staging_rules():
    assert assign_stage(0, 0.05, 0.04) == 1
    assert assign_stage(45, 0.05, 0.04) == 2
    assert assign_stage(0, 0.10, 0.04) == 2
    assert assign_stage(120, 0.05, 0.04) == 3


def test_bad_pd_rejected():
    with pytest.raises(ValueError):
        marginal_pds(1.5, 2)
