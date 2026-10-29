"""Affordability check in the style of the South African NCR regulations.

A score tells you how likely someone is to default. Under the National
Credit Act a lender also has to show the customer can afford the new
repayment, otherwise the loan can be ruled reckless credit. This module
does a simplified version of that check.

The minimum expense norms below are Table 1 of regulation 23A(10) of the
National Credit Regulations (Affordability Assessment Regulations, in force
since 2015). Note that one widely used online copy shows the second band's
fixed amount as R900; the correct figure is R800, which is the only value
that makes R800 + 6.75% x R5 450 equal the next band's R1 167.88.
"""

from dataclasses import dataclass

# (lower bound, upper bound, fixed amount, rate on income above lower bound)
# all amounts are gross monthly income in rand
MIN_EXPENSE_NORMS = [
    (0.00, 800.00, 0.00, 1.0000),
    (800.01, 6250.00, 800.00, 0.0675),
    (6250.01, 25000.00, 1167.88, 0.0900),
    (25000.01, 50000.00, 2855.38, 0.0820),
    (50000.01, float("inf"), 4905.38, 0.0675),
]


def min_expense_norm(gross_income):
    """Minimum monthly living expenses the lender must assume."""
    if gross_income < 0:
        raise ValueError("income cannot be negative")
    for lo, hi, fixed, rate in MIN_EXPENSE_NORMS:
        if gross_income <= hi:
            return fixed + rate * (gross_income - max(lo, 0.0))
    raise AssertionError("unreachable")


def monthly_instalment(principal, annual_rate, months):
    """Standard annuity repayment."""
    if months <= 0:
        raise ValueError("months must be positive")
    r = annual_rate / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


@dataclass
class Applicant:
    gross_income: float        # monthly, rand
    net_income: float          # after tax and statutory deductions
    declared_expenses: float   # what the applicant says they spend
    existing_debt: float       # current monthly debt repayments


def discretionary_income(a: Applicant):
    """Net income left after living expenses and existing debt.

    Uses the higher of declared expenses and the minimum norm, so an
    applicant cannot understate their way into a loan.
    """
    expenses = max(a.declared_expenses, min_expense_norm(a.gross_income))
    return a.net_income - expenses - a.existing_debt


def assess(a: Applicant, score, principal, annual_rate, months,
           cutoff=500, buffer=0.0):
    """Approve only if the score clears the cutoff AND the new instalment
    fits inside discretionary income (less an optional safety buffer)."""
    instalment = monthly_instalment(principal, annual_rate, months)
    room = discretionary_income(a) * (1 - buffer)
    score_ok = score >= cutoff
    affordable = instalment <= room

    if score_ok and affordable:
        decision = "approve"
    elif not score_ok:
        decision = "decline: score"
    else:
        decision = "decline: affordability"

    return {
        "instalment": round(instalment, 2),
        "discretionary_income": round(room, 2),
        "score_ok": score_ok,
        "affordable": affordable,
        "decision": decision,
    }
