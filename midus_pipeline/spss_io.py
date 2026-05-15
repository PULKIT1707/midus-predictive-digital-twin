from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pyreadstat


@dataclass(frozen=True)
class SpssMetadata:
    file_path: str
    row_count: int
    column_count: int
    columns: pd.DataFrame


def _normalize_label(label: Optional[str]) -> str:
    if label is None:
        return ""
    # Keep formatting stable: strip whitespace and collapse newlines
    return " ".join(str(label).replace("\r", " ").replace("\n", " ").split()).strip()


def load_sav(path: str | Path, *, usecols: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, Any]:
    """Load a .sav file and return (df, pyreadstat_meta)."""

    p = Path(path)
    df, meta = pyreadstat.read_sav(p, usecols=usecols)
    return df, meta


def build_data_dictionary(meta: Any, *, file_path: str | Path) -> SpssMetadata:
    """Build a tabular data dictionary from pyreadstat metadata.

    `meta` is the object returned by `pyreadstat.read_sav`.
    """

    file_path_str = str(Path(file_path))

    column_names: List[str] = list(getattr(meta, "column_names", []) or [])
    column_labels: List[str] = list(getattr(meta, "column_labels", []) or [])
    variable_measure: Mapping[str, Any] = getattr(meta, "variable_measure", {}) or {}
    variable_value_labels: Mapping[str, Dict[Any, Any]] = getattr(meta, "variable_value_labels", {}) or {}
    missing_ranges: Mapping[str, Any] = getattr(meta, "missing_ranges", {}) or {}

    # pyreadstat sometimes provides fewer labels than names
    if len(column_labels) < len(column_names):
        column_labels = column_labels + [""] * (len(column_names) - len(column_labels))

    rows: List[Dict[str, Any]] = []
    for name, label in zip(column_names, column_labels):
        vlabels = variable_value_labels.get(name)
        rows.append(
            {
                "file_path": file_path_str,
                "code": name,
                "label": _normalize_label(label),
                "measure": variable_measure.get(name),
                "has_value_labels": vlabels is not None,
                "value_labels": vlabels,
                "missing_ranges": missing_ranges.get(name),
            }
        )

    columns_df = pd.DataFrame(rows)

    row_count = int(getattr(meta, "number_rows", 0) or 0)
    column_count = int(getattr(meta, "number_columns", len(column_names)) or len(column_names))

    return SpssMetadata(
        file_path=file_path_str,
        row_count=row_count,
        column_count=column_count,
        columns=columns_df,
    )


def make_code_label_columns(
    df: pd.DataFrame,
    meta: Any,
    *,
    separator: str = " : ",
    prefer_code_when_no_label: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Rename columns to `CODE : Label`.

    Returns:
        (renamed_df, rename_map) where rename_map maps original_code -> new_name.

    Collision handling:
        If two columns would produce the same new name, suffix with ` [n]`.
    """

    column_names: List[str] = list(getattr(meta, "column_names", []) or list(df.columns))
    column_labels: List[str] = list(getattr(meta, "column_labels", []) or [])

    if len(column_labels) < len(column_names):
        column_labels = column_labels + [""] * (len(column_names) - len(column_labels))

    used: Dict[str, int] = {}
    rename_map: Dict[str, str] = {}

    for code, raw_label in zip(column_names, column_labels):
        label = _normalize_label(raw_label)
        if not label and prefer_code_when_no_label:
            new_name = code
        else:
            new_name = f"{code}{separator}{label}" if label else code

        if new_name in used:
            used[new_name] += 1
            new_name = f"{new_name} [{used[new_name]}]"
        else:
            used[new_name] = 1

        rename_map[code] = new_name

    out = df.rename(columns=rename_map)
    return out, rename_map


def apply_value_labels(
    df: pd.DataFrame,
    meta: Any,
    *,
    columns: Optional[Iterable[str]] = None,
    keep_unmapped: bool = True,
    as_categorical: bool = False,
) -> pd.DataFrame:
    """Map numeric codes to their SPSS value labels for specified columns.

    Notes:
        - This is intentionally conservative: only applies when value labels exist.
        - If `keep_unmapped` is True, values not in mapping remain as-is.
    """

    variable_value_labels: Mapping[str, Dict[Any, Any]] = getattr(meta, "variable_value_labels", {}) or {}

    if columns is None:
        target_cols = [c for c in df.columns if c in variable_value_labels]
    else:
        target_cols = [c for c in columns if c in variable_value_labels]

    out = df.copy()

    for c in target_cols:
        mapping = variable_value_labels.get(c)
        if not mapping:
            continue

        # Normalize mapping values to strings to avoid mixed dtypes
        mapping_norm = {k: str(v) if v is not None else "" for k, v in mapping.items()}

        if keep_unmapped:
            out[c] = out[c].map(mapping_norm).where(out[c].map(mapping_norm).notna(), out[c])
        else:
            out[c] = out[c].map(mapping_norm)

        if as_categorical:
            out[c] = out[c].astype("category")

    return out
