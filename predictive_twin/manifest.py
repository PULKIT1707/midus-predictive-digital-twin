from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class ManifestSelection:
    predictors_m1: List[str]
    predictors_m2: List[str]
    prior_cognition_code: Optional[str]
    outcome_code: str


def read_manifest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def filter_included(df: pd.DataFrame) -> pd.DataFrame:
    if "include_decision" not in df.columns:
        raise ValueError("Manifest missing required column: include_decision")
    return df[df["include_decision"].eq("include")].copy()


def select_features(
    manifest_df: pd.DataFrame,
    *,
    m1_dataset_name: str,
    m2_dataset_name: str,
    outcome_code: str,
    prior_cognition_code: str,
    include_prior: bool,
) -> ManifestSelection:
    df = filter_included(manifest_df)

    # M3 outcome is not a predictor
    df_pred = df[df["code"].ne(outcome_code)].copy()

    m1 = df_pred[df_pred["dataset"].eq(m1_dataset_name) & df_pred["wave"].eq("M1")]
    m2 = df_pred[df_pred["dataset"].eq(m2_dataset_name) & df_pred["wave"].eq("M2")]

    predictors_m1 = sorted(m1["code"].astype(str).unique().tolist())
    predictors_m2 = sorted(m2["code"].astype(str).unique().tolist())

    if include_prior:
        if prior_cognition_code not in predictors_m2:
            predictors_m2 = predictors_m2 + [prior_cognition_code]
    else:
        predictors_m2 = [c for c in predictors_m2 if c != prior_cognition_code]

    return ManifestSelection(
        predictors_m1=predictors_m1,
        predictors_m2=predictors_m2,
        prior_cognition_code=prior_cognition_code if include_prior else None,
        outcome_code=outcome_code,
    )
