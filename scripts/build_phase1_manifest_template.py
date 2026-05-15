from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


CATALOG_FILES = [
    "artifacts/column_catalog__MIDUS_1__M1_P1_SURVEY_N7108_20190116.sav.csv",
    "artifacts/column_catalog__MIDUS_2__M2_P1_SURVEY_N4963_20200720.sav.csv",
    "artifacts/column_catalog__MIDUS_2__M2_P3_BTACT_N4512_20211123.sav.csv",
    "artifacts/column_catalog__MIDUS_3__M3_P3_BTACT_N3291_20210922.sav.csv",
    "artifacts/column_catalog__MIDUS_2__M2_BIO_AGGREGATE_N1255_20240405.sav.csv",
    "artifacts/column_catalog__MIDUS_2__M2_P2_DAILY_DATA_N=2022_08-04-10 (1).sav.csv",
]


def _read_existing(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _leakage_status(wave: str, role: str) -> tuple[str, str]:
    # Hard rule for Phase 1: predicting M3 outcome => forbid all M3 predictors except the outcome itself.
    if wave == "M3":
        if role in {"primary_outcome"}:
            return "pass", ""
        return "fail", "M3_non_target_forbidden"
    # M1/M2 predictors are allowed
    return "pass", ""


def build_template(catalogs: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(catalogs, ignore_index=True)

    # Default role assignments (only for the explicitly agreed cognition pair)
    df["recommended_role"] = "candidate_predictor"
    df.loc[(df["wave"] == "M3") & (df["code"].str.upper() == "C3TCOMP"), "recommended_role"] = "primary_outcome"
    df.loc[(df["wave"] == "M2") & (df["code"].str.upper() == "B3TCOMPZ1"), "recommended_role"] = "prior_cognition"

    # Default tier based on dataset family; user will curate.
    df["tier"] = "review"
    df.loc[df["dataset_family"].eq("core_survey") & df["wave"].isin(["M1", "M2"]), "tier"] = "tier_1"
    df.loc[df["recommended_role"].eq("prior_cognition"), "tier"] = "tier_2a"
    df.loc[df["dataset_family"].eq("biomarkers") & df["wave"].eq("M2"), "tier"] = "tier_2b"
    df.loc[df["dataset_family"].eq("daily_diary") & df["wave"].eq("M2"), "tier"] = "tier_3"

    # Conservative: do not auto-assign these families; leave unassigned for manual review.
    df["feature_family"] = "unassigned"
    df["construct"] = ""
    df["justification_tag"] = ""

    # Missingness typing cannot be inferred reliably without study design metadata; leave unknown.
    df["missingness_type"] = "unknown"

    # Include decision left for manual curation
    df["include_decision"] = "review"
    df.loc[df["recommended_role"].eq("primary_outcome"), "include_decision"] = "include"
    df.loc[df["recommended_role"].eq("prior_cognition"), "include_decision"] = "include"

    # Leakage checks
    statuses = df.apply(lambda r: _leakage_status(str(r["wave"]), str(r["recommended_role"])), axis=1)
    df["leakage_check_status"] = [s[0] for s in statuses]
    df["leakage_reason"] = [s[1] for s in statuses]

    # Notes column for manual edits
    df["notes"] = ""

    # Keep only relevant columns for the manifest
    cols = [
        "wave",
        "dataset",
        "dataset_family",
        "code",
        "label",
        "missingness_rate",
        "n_unique",
        "dtype",
        "recommended_role",
        "tier",
        "feature_family",
        "construct",
        "justification_tag",
        "missingness_type",
        "include_decision",
        "leakage_check_status",
        "leakage_reason",
        "notes",
    ]
    out = df[cols].copy()

    # Sort to surface the key agreed variables first
    role_order = {
        "primary_outcome": 0,
        "prior_cognition": 1,
        "candidate_predictor": 2,
    }
    out["_role_order"] = out["recommended_role"].map(role_order).fillna(9)
    out = out.sort_values(["_role_order", "tier", "wave", "dataset_family", "code"], ascending=True).drop(columns=["_role_order"]).reset_index(drop=True)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Phase 1 feature manifest TEMPLATE for manual curation")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogs = []
    for f in CATALOG_FILES:
        df = _read_existing(f)
        if df.empty:
            continue
        catalogs.append(df)

    if not catalogs:
        raise SystemExit("No catalog CSVs found. Run scripts/generate_column_catalogs.py first.")

    tmpl = build_template(catalogs)

    tmpl.to_csv(out_dir / "phase1_feature_manifest__TEMPLATE.csv", index=False)
    tmpl.to_parquet(out_dir / "phase1_feature_manifest__TEMPLATE.parquet", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
