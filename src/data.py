"""Loading and splitting the German Credit data."""

import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import scorecardpy as sc

TARGET = "creditability"
SEED = 42

# Left out on purpose, see README (fairness section).
EXCLUDED = ["personal_status_and_sex", "foreign_worker"]


def load_data(drop_cols=EXCLUDED):
    """Return the German Credit data with a 0/1 target (1 = bad)."""
    df = sc.germancredit()
    df = df.drop(columns=list(drop_cols))
    df[TARGET] = (df[TARGET] == "bad").astype(int)
    return df


def split_data(df, test_size=0.3, seed=SEED):
    """Stratified train/test split so both sides keep the 30% bad rate."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
