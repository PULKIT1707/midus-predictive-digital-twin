from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyreadstat


def _normalize_label(label: Optional[str]) -> str:
    if label is None:
        return ""
    return " ".join(str(label).replace("\r", " ").replace("\n", " ").split()).strip()


def _infer_wave_from_path(path: Path) -> str:
    s = str(path).lower()
    if "midus 1" in s or "m1_" in path.name.lower() or path.name.lower().startswith("m1"):
        return "M1"
    if "midus 2" in s or path.name.lower().startswith("m2"):
        return "M2"
    if "midus 3" in s or path.name.lower().startswith("m3"):
        return "M3"
    if "refresher 1" in s or path.name.lower().startswith("mr1"):
        return "MR1"
    if "refresher 2" in s or path.name.lower().startswith("mr2"):
        return "MR2"
    return ""


def _infer_dataset_family(path: Path) -> str:
    name = path.name.lower()

    if "btact" in name or "moca" in name:
        return "cognition"
    if "bio" in name:
        return "biomarkers"
    if "daily" in name or "diary" in name:
        return "daily_diary"
    if "twinscreener" in name or "twin" in name:
        return "twin_screener"

    if "survey" in name:
        return "core_survey"

    if "disposition" in name:
        return "disposition"
    if "codedtext" in name or "coded text" in name:
        return "coded_text"
    if "medication" in name:
        return "medications"

    return "other"


def _catalog_one_file(path: Path) -> pd.DataFrame:
    df, meta = pyreadstat.read_sav(path)

    wave = _infer_wave_from_path(path)
    dataset_family = _infer_dataset_family(path)

    rows: List[Dict[str, Any]] = []
    for code, raw_label in zip(meta.column_names, meta.column_labels):
        if code not in df.columns:
            continue
        label = _normalize_label(raw_label)
        s = df[code]
        missingness_rate = float(s.isna().mean())
        rows.append(
            {
                "wave": wave,
                "dataset": path.name,
                "dataset_family": dataset_family,
                "code": code,
                "label": label,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "missingness_rate": missingness_rate,
                "n_unique": int(s.nunique(dropna=True)),
                "feature_family_prelim": "unassigned",
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["wave", "dataset", "code"], ascending=[True, True, True]).reset_index(drop=True)
    return out


def _list_sav_files(input_dir: Path) -> List[Path]:
    savs = list(input_dir.rglob("*.sav")) + list(input_dir.rglob("*.SAV"))
    return sorted(set(savs))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full column catalogs (code+label+missingness) for all MIDUS .sav files")
    parser.add_argument("--input", type=str, default="data", help="Directory to scan for .sav files (default: data)")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory (default: artifacts)")
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Write one CSV per dataset in addition to the combined master catalog",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sav_files = _list_sav_files(input_dir)
    if not sav_files:
        raise SystemExit(f"No .sav files found under: {input_dir}")

    all_rows: List[pd.DataFrame] = []
    for p in sav_files:
        catalog = _catalog_one_file(p)
        all_rows.append(catalog)

        if args.per_file:
            stem = p.relative_to(input_dir).as_posix().replace("/", "__").replace(" ", "_")
            catalog.to_csv(out_dir / f"column_catalog__{stem}.csv", index=False)

    master = pd.concat(all_rows, ignore_index=True)
    master.to_csv(out_dir / "column_catalog__all_datasets.csv", index=False)
    master.to_parquet(out_dir / "column_catalog__all_datasets.parquet", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
