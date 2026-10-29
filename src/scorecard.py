"""Turn the logistic regression into scorecard points.

Scaling: 600 points at good:bad odds of 50:1, and every 20 points
doubles the odds (PDO = 20).
"""

import numpy as np
import pandas as pd

BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20


def scaling(base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    """Return (factor, offset) so that score = offset + factor * ln(good:bad odds)."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def score_from_pd(pd_bad, **kw):
    factor, offset = scaling(**kw)
    pd_bad = np.clip(np.asarray(pd_bad, dtype=float), 1e-9, 1 - 1e-9)
    return offset + factor * np.log((1 - pd_bad) / pd_bad)


def pd_from_score(score, **kw):
    factor, offset = scaling(**kw)
    odds = np.exp((np.asarray(score, dtype=float) - offset) / factor)
    return 1 / (1 + odds)


def build_scorecard(result, bins, **kw):
    """Points for every bin of every feature in the model, plus base points.

    Points for a bin = -factor * coef * woe. The intercept goes into
    the base points so the bin points are easy to read on their own.
    """
    factor, offset = scaling(**kw)
    rows = [{"feature": "base", "bin": "", "woe": np.nan,
             "points": round(offset - factor * result.params["const"])}]
    for col, coef in result.params.drop("const").items():
        feat = col.removesuffix("_woe")
        for _, b in bins[feat].iterrows():
            rows.append({
                "feature": feat,
                "bin": b["bin"],
                "woe": b["woe"],
                "points": round(-factor * coef * b["woe"]),
            })
    return pd.DataFrame(rows)


def score_woe(result, X_woe, **kw):
    """Total points for each row of a WoE matrix (before rounding)."""
    factor, offset = scaling(**kw)
    params = result.params
    logit = params["const"] + X_woe[params.drop("const").index] @ params.drop("const")
    return np.asarray(offset - factor * logit)
