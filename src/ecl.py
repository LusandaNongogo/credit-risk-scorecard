"""Simplified IFRS 9 expected credit loss.

ECL = PD x LGD x EAD, with the horizon set by the stage:
  stage 1: 12 month ECL (performing, no big increase in risk)
  stage 2: lifetime ECL (significant increase in credit risk)
  stage 3: lifetime ECL with PD = 1 (credit impaired, in default)

Simplifications: lifetime PD assumes a constant annual hazard, EAD is
flat over the life of the loan and losses are discounted yearly at the
effective interest rate. Real IFRS 9 models add forward looking
macroeconomic scenarios, amortising exposure and term structures of PD.
"""

import numpy as np

DPD_STAGE_2 = 30
DPD_STAGE_3 = 90
SICR_MULTIPLE = 2.0  # PD at least doubled since origination counts as significant


def assign_stage(days_past_due, pd_now, pd_origination, sicr_multiple=SICR_MULTIPLE):
    if days_past_due > DPD_STAGE_3:
        return 3
    if days_past_due > DPD_STAGE_2 or pd_now >= sicr_multiple * pd_origination:
        return 2
    return 1


def marginal_pds(pd_12m, years):
    """Probability of defaulting in each year, given survival to that year."""
    if not 0 <= pd_12m <= 1:
        raise ValueError("pd_12m must be between 0 and 1")
    years = max(1, int(np.ceil(years)))
    survival = (1 - pd_12m) ** np.arange(years)
    return survival * pd_12m


def lifetime_pd(pd_12m, years):
    return float(marginal_pds(pd_12m, years).sum())


def ecl(pd_12m, lgd, ead, stage, remaining_years=1.0, eir=0.0):
    """Expected credit loss for one exposure."""
    if stage == 3:
        return lgd * ead
    horizon = 1 if stage == 1 else max(1, int(np.ceil(remaining_years)))
    marg = marginal_pds(pd_12m, horizon)
    discount = (1 + eir) ** -np.arange(1, horizon + 1)
    return float(np.sum(marg * lgd * ead * discount))
