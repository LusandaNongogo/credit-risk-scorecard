"""Logistic regression PD model on WoE features."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_logit(X, y):
    """Fit a logistic regression with statsmodels so we get p values too."""
    return sm.Logit(np.asarray(y), sm.add_constant(X)).fit(disp=0)


def coef_table(result):
    """Coefficients with a sign check.

    With WoE = ln(%bad / %good) every feature coefficient should be
    positive: a riskier bin should push the PD up. A negative sign
    usually means the feature is fighting a correlated one.
    """
    t = pd.DataFrame({
        "coef": result.params,
        "std_err": result.bse,
        "p_value": result.pvalues,
    })
    t["sign_ok"] = (t["coef"] > 0) | (t.index == "const")
    return t


def fit_with_sign_check(X, y, max_rounds=10):
    """Refit, dropping the worst wrong signed feature each round, until
    every coefficient is positive. Returns the result and dropped columns."""
    cols = list(X.columns)
    dropped = []
    for _ in range(max_rounds):
        res = fit_logit(X[cols], y)
        params = res.params.drop("const")
        bad = params[params <= 0]
        if bad.empty:
            return res, dropped
        worst = bad.idxmin()
        cols.remove(worst)
        dropped.append(worst)
    raise RuntimeError("signs still wrong after max_rounds")


def predict_pd(result, X):
    return np.asarray(result.predict(sm.add_constant(X, has_constant="add")))
