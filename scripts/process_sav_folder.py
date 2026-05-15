from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from midus_pipeline.spss_io import (
    apply_value_labels,
    build_data_dictionary,
    load_sav,
    make_code_label_columns,
)


def _list_sav_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.sav")] + [p for p in input_dir.rglob("*.SAV")])


def main() -> int:
    parser = argparse.ArgumentParser(description="Process MIDUS SPSS .sav files into renamed parquet/csv + data dictionaries")
    parser.add_argument("--input", type=str, default="data", help="Input directory containing .sav files (default: data)")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv"], help="Output dataset format")
    parser.add_argument("--apply-value-labels", action="store_true", help="Map SPSS value labels onto coded columns")
    parser.add_argument("--value-labels-as-category", action="store_true", help="When applying value labels, cast mapped columns to pandas 'category'")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sav_files = _list_sav_files(input_dir)
    if not sav_files:
        raise SystemExit(f"No .sav files found under: {input_dir}")

    dict_rows = []

    for sav_path in sav_files:
        rel = sav_path.relative_to(input_dir)
        stem = rel.as_posix().replace("/", "__").replace(" ", "_")

        df, meta = load_sav(sav_path)
        dd = build_data_dictionary(meta, file_path=sav_path)
        dict_rows.append(dd.columns)

        df2, rename_map = make_code_label_columns(df, meta)

        if args.apply_value_labels:
            # apply using original codes, before rename
            df_labeled = apply_value_labels(df, meta, as_categorical=args.value_labels_as_category)
            df2, _ = make_code_label_columns(df_labeled, meta)

        data_out = out_dir / f"{stem}.{ 'parquet' if args.format == 'parquet' else 'csv' }"
        dict_out = out_dir / f"{stem}__data_dictionary.parquet"

        if args.format == "parquet":
            df2.to_parquet(data_out, index=False)
        else:
            df2.to_csv(data_out, index=False)

        dd.columns.to_parquet(dict_out, index=False)

    full_dict = pd.concat(dict_rows, ignore_index=True)
    full_dict.to_parquet(out_dir / "data_dictionary__all.parquet", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
