"""Weight of evidence (WoE) and information value (IV).

WoE here follows the scorecardpy convention: ln(%bad / %good), so a
positive WoE means the bin is riskier than average.
"""

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import scorecardpy as sc

IV_MIN = 0.02
IV_SUSPICIOUS = 0.5


def woe_table(bins, y, smoothing=0.5):
    """WoE and IV for each bin of one feature.

    bins: bin label per row. y: 0/1 target with 1 = bad.
    Bins with no goods or no bads get `smoothing` added to both counts
    so the log stays finite.
    """
    df = pd.DataFrame({"bin": np.asarray(bins), "y": np.asarray(y)})
    t = df.groupby("bin", sort=True)["y"].agg(count="size", bad="sum")
    t["good"] = t["count"] - t["bad"]

    good = t["good"].astype(float)
    bad = t["bad"].astype(float)
    empty = (good == 0) | (bad == 0)
    good[empty] += smoothing
    bad[empty] += smoothing

    t["bad_rate"] = t["bad"] / t["count"]
    t["dist_good"] = good / good.sum()
    t["dist_bad"] = bad / bad.sum()
    t["woe"] = np.log(t["dist_bad"] / t["dist_good"])
    t["iv"] = (t["dist_bad"] - t["dist_good"]) * t["woe"]
    return t.reset_index()


def information_value(bins, y):
    return float(woe_table(bins, y)["iv"].sum())


def bin_features(train, target):
    """Fit WoE bins on the training data with scorecardpy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sc.woebin(train, y=target, print_info=False)


def iv_summary(bins, iv_min=IV_MIN, iv_flag=IV_SUSPICIOUS):
    """One row per feature with its IV, whether we keep it and a flag
    for IV high enough to be worth a second look."""
    rows = [
        {"feature": name, "iv": float(b["total_iv"].iloc[0])}
        for name, b in bins.items()
    ]
    s = pd.DataFrame(rows).sort_values("iv", ascending=False)
    s["keep"] = s["iv"] >= iv_min
    s["flag_high"] = s["iv"] > iv_flag
    return s.reset_index(drop=True)


def apply_woe(df, bins, features):
    """Replace raw values with their WoE. Returns columns named <feature>_woe."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sc.woebin_ply(df[features], {f: bins[f] for f in features},
                            print_info=False)
    return out[[f + "_woe" for f in features]]
