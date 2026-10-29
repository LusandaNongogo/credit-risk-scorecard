import pytest

from src.affordability import (Applicant, assess, discretionary_income,
                               min_expense_norm, monthly_instalment)


def test_norm_band_edges():
    assert min_expense_norm(0) == pytest.approx(0)
    assert min_expense_norm(800) == pytest.approx(800)
    # R10 000: 1167.88 + 9% of (10000 - 6250.01)
    assert min_expense_norm(10000) == pytest.approx(1167.88 + 0.09 * 3749.99)


def test_norm_rises_with_income():
    incomes = [1000, 5000, 15000, 30000, 80000]
    norms = [min_expense_norm(i) for i in incomes]
    assert norms == sorted(norms)


def test_negative_income_rejected():
    with pytest.raises(ValueError):
        min_expense_norm(-1)


def test_instalment_zero_rate():
    assert monthly_instalment(12000, 0.0, 12) == pytest.approx(1000)


def test_instalment_known_value():
    # R100 000 over 60 months at 12% a year is about R2 224.44
    assert monthly_instalment(100000, 0.12, 60) == pytest.approx(2224.44, abs=0.01)


def test_declared_expenses_floored_at_norm():
    a = Applicant(gross_income=20000, net_income=16000,
                  declared_expenses=500, existing_debt=2000)
    assert discretionary_income(a) == pytest.approx(
        16000 - min_expense_norm(20000) - 2000)


def test_decisions():
    a = Applicant(20000, 16000, 6000, 2000)  # about R8 000 spare
    assert assess(a, 560, 50000, 0.18, 36)["decision"] == "approve"
    assert assess(a, 450, 50000, 0.18, 36)["decision"] == "decline: score"
    assert assess(a, 560, 400000, 0.18, 36)["decision"] == "decline: affordability"
