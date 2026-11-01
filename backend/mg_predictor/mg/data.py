"""Dataset loading, validation and feature engineering."""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_dataset(path=C.DATA_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{path} not found. Run `python generate_dataset.py` to create it."
        ) from exc
    required = [C.TARGET, C.COHORT_COL, C.DIAGNOSIS_COL, "age", "Gender", "duration_months",
                *C.MG_SIGNS, *C.SYMPTOMS, *C.BLOOD_FLAGS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def build_xy(df: pd.DataFrame):
    """Return (X, y) using only information a patient can report at screening time."""
    derived = (df[C.COHORT_COL] == C.MG_COHORT).astype(int)
    if not (derived == df[C.TARGET]).all():
        raise ValueError("MG_Label disagrees with the Cohort column")

    X = pd.DataFrame(index=df.index)
    for key in [*C.MG_SIGNS, *C.SYMPTOMS, *C.BLOOD_FLAGS]:
        X[key] = df[key].astype(int)
    X["age"] = df["age"].astype(float)
    X["gender_male"] = (df["Gender"].str.strip().str.lower() == "male").astype(int)
    X["duration_months"] = df["duration_months"].astype(float)

    if X.isna().any().any():
        raise ValueError("Missing values found in model features")
    return X[C.FEATURES], df[C.TARGET].astype(int)


def dataset_breakdown(df: pd.DataFrame) -> dict:
    """MG vs Normal vs other autoimmune conditions (for the pie chart)."""
    counts = df[C.COHORT_COL].value_counts()
    return {
        C.MG_COHORT: int(counts.get(C.MG_COHORT, 0)),
        C.HEALTHY_COHORT: int(counts.get(C.HEALTHY_COHORT, 0)),
        C.OTHER_COHORT: int(counts.get(C.OTHER_COHORT, 0)),
    }
