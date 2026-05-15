from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyreadstat


@dataclass(frozen=True)
class BtactFileSpec:
    wave: str
    dataset_path: Path


def _normalize_label(label: Optional[str]) -> str:
    if label is None:
        return ""
    return " ".join(str(label).replace("\r", " ").replace("\n", " ").split()).strip()


def _directionality_from_code_and_label(code: str, label: str) -> str:
    s = f"{code} {label}".lower()

    worse_tokens = [
        "intrusion",
        "intrusions",
        "repetition",
        "repetitions",
        "error",
        "errors",
    ]

    if any(t in s for t in worse_tokens):
        return "higher=worse"

    # Default convention for BTACT composites/totals
    return "higher=better"


def _sample_restriction_from_label(label: str) -> str:
    s = label.lower()
    if "milwaukee" in s:
        return "milwaukee"
    if "national" in s:
        return "national"
    if "complete sample" in s or "complete" in s:
        return "complete"
    return ""


def _scaling_note_from_code_and_label(code: str, label: str) -> str:
    s = f"{code} {label}".lower()
    if "zscore" in s or "z-score" in s or "z score" in s:
        return "z-score"
    if "standardized" in s:
        # keep the original wording for traceability
        return "standardized"
    return ""


def _scoring_description_from_label(label: str) -> str:
    # Keep it conservative and traceable.
    return label


def _recommended_role(wave: str, code: str, label: str) -> str:
    code_u = code.upper()
    if wave == "M3" and code_u == "C3TCOMP":
        return "primary_outcome"
    if wave == "M2" and code_u == "B3TCOMPZ1":
        return "prior_cognition"

    s = f"{code} {label}".lower()
    if "composite" in s or "zscore" in s or "z-score" in s:
        return "secondary_subtest"

    # Subtest totals (unique words, total correct, etc.) are candidates for later
    subtest_tokens = [
        "word list",
        "category fluency",
        "number series",
        "backward counting",
        "total correct",
        "total unique",
    ]
    if any(t in s for t in subtest_tokens):
        return "secondary_subtest"

    return "exclude"


def _missingness_rate(series: pd.Series) -> float:
    return float(series.isna().mean())


def generate_btact_harmonization(specs: List[BtactFileSpec]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for spec in specs:
        df, meta = pyreadstat.read_sav(spec.dataset_path)

        for code, raw_label in zip(meta.column_names, meta.column_labels):
            label = _normalize_label(raw_label)
            if code not in df.columns:
                continue

            miss = _missingness_rate(df[code])

            sample_restriction = _sample_restriction_from_label(label)
            scaling_note = _scaling_note_from_code_and_label(code, label)
            directionality = _directionality_from_code_and_label(code, label)
            recommended_role = _recommended_role(spec.wave, code, label)

            rows.append(
                {
                    "wave": spec.wave,
                    "dataset": spec.dataset_path.name,
                    "code": code,
                    "label": label,
                    "scoring_description": _scoring_description_from_label(label),
                    "sample_restriction": sample_restriction,
                    "scaling_note": scaling_note,
                    "missingness_rate": miss,
                    "directionality": directionality,
                    "recommended_role": recommended_role,
                    "notes": "",
                }
            )

    out = pd.DataFrame(rows)
    # Only keep likely BTACT performance metrics + composites
    keep_mask = out["code"].str.match(r"^[BC]3T", na=False)
    out = out.loc[keep_mask].copy()

    # Sort for readability
    out = out.sort_values(["wave", "recommended_role", "code"], ascending=[True, True, True]).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BTACT harmonization artifact and (later) feature manifest")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        BtactFileSpec(wave="M2", dataset_path=Path("data") / "MIDUS 2" / "M2_P3_BTACT_N4512_20211123.sav"),
        BtactFileSpec(wave="M3", dataset_path=Path("data") / "MIDUS 3" / "M3_P3_BTACT_N3291_20210922.sav"),
    ]

    df_h = generate_btact_harmonization(specs)
    df_h.to_parquet(out_dir / "btact_harmonization_m2_m3.parquet", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
