from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class LeakageViolation:
    code: str
    wave: str
    dataset: str
    reason: str


def validate_no_m3_predictors(manifest_df: pd.DataFrame, *, outcome_code: str) -> List[LeakageViolation]:
    """Hard rule: no M3 predictors allowed except the M3 outcome itself."""

    violations: List[LeakageViolation] = []

    required_cols = {"wave", "code", "dataset", "include_decision"}
    missing = required_cols - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    included = manifest_df[manifest_df["include_decision"].eq("include")].copy()

    m3 = included[included["wave"].eq("M3") & included["code"].ne(outcome_code)]
    for _, r in m3.iterrows():
        violations.append(
            LeakageViolation(
                code=str(r["code"]),
                wave=str(r["wave"]),
                dataset=str(r["dataset"]),
                reason="M3_non_target_forbidden",
            )
        )

    return violations
