from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from midus_pipeline.spss_io import load_sav


@dataclass(frozen=True)
class JoinReport:
    join_id: str
    m1_rows: int
    m2_rows: int
    m3_outcome_rows: int
    after_m1_m2_rows: int
    after_all_rows: int
    overlap_m1_m2: int
    overlap_m1_m3: int
    overlap_m2_m3: int
    one_to_one_ok: bool


def _assert_one_row_per_id(df: pd.DataFrame, join_id: str, name: str) -> None:
    if join_id not in df.columns:
        raise ValueError(f"{name}: join_id column not found: {join_id}")
    dup = df[join_id].duplicated(keep=False)
    if dup.any():
        bad = df.loc[dup, join_id].value_counts().head(20)
        raise ValueError(f"{name}: duplicate IDs found for join_id={join_id}. Sample counts: {bad.to_dict()}")


def _load_required_columns(path: Path, usecols: List[str]) -> pd.DataFrame:
    df, _ = load_sav(path, usecols=usecols)
    return df


def build_modeling_dataset(
    *,
    m1_survey_path: Path,
    m2_survey_path: Path,
    m2_prior_path: Path,
    m3_outcome_path: Path,
    join_id: str,
    predictors_m1: List[str],
    predictors_m2: List[str],
    outcome_code: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, JoinReport]:
    """Build leakage-safe modeling dataset.

    Returns:
        X, y, ids_df, join_report

    Notes:
        - Drops rows with missing outcome.
        - Enforces one-row-per-ID for each input table.
    """

    m1_use = sorted(set([join_id] + predictors_m1))
    m2_use = sorted(set([join_id] + predictors_m2))
    m2_prior_use = sorted(set([join_id] + ([c for c in predictors_m2 if c.startswith("B3T") or c == "B3TCOMPZ1"])))
    m3_use = sorted(set([join_id, outcome_code]))

    m1 = _load_required_columns(m1_survey_path, m1_use)
    m2 = _load_required_columns(m2_survey_path, m2_use)

    # If prior cognition is requested but not present in M2 survey, pull from BTACT file.
    need_prior = any(c == "B3TCOMPZ1" for c in predictors_m2)
    if need_prior and "B3TCOMPZ1" not in m2.columns:
        bt = _load_required_columns(m2_prior_path, [join_id, "B3TCOMPZ1"])
        _assert_one_row_per_id(bt, join_id, "M2_BTACT")
        m2 = m2.merge(bt, on=join_id, how="left", validate="one_to_one")

    m3 = _load_required_columns(m3_outcome_path, m3_use)

    _assert_one_row_per_id(m1, join_id, "M1_SURVEY")
    _assert_one_row_per_id(m2, join_id, "M2_SURVEY")
    _assert_one_row_per_id(m3, join_id, "M3_OUTCOME")

    overlap_m1_m2 = int(pd.Index(m1[join_id]).intersection(pd.Index(m2[join_id])).size)
    overlap_m1_m3 = int(pd.Index(m1[join_id]).intersection(pd.Index(m3[join_id])).size)
    overlap_m2_m3 = int(pd.Index(m2[join_id]).intersection(pd.Index(m3[join_id])).size)

    merged = m1.merge(m2, on=join_id, how="inner", validate="one_to_one", suffixes=("__M1", "__M2"))
    after_m1_m2_rows = int(len(merged))

    merged = merged.merge(m3, on=join_id, how="inner", validate="one_to_one")

    # Drop missing outcome
    merged = merged[merged[outcome_code].notna()].copy()

    y = merged[outcome_code]
    ids_df = merged[[join_id]].copy()

    X = merged.drop(columns=[outcome_code])

    # One-to-one already enforced by merge validate; if we got here, OK.
    report = JoinReport(
        join_id=join_id,
        m1_rows=int(len(m1)),
        m2_rows=int(len(m2)),
        m3_outcome_rows=int(len(m3)),
        after_m1_m2_rows=after_m1_m2_rows,
        after_all_rows=int(len(merged)),
        overlap_m1_m2=overlap_m1_m2,
        overlap_m1_m3=overlap_m1_m3,
        overlap_m2_m3=overlap_m2_m3,
        one_to_one_ok=True,
    )

    return X, y, ids_df, report
