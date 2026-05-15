from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class GlobalTopFeature:
    feature: str
    score: float
    label: Optional[str] = None


def build_code_to_label_map_from_manifest(manifest_path: Path) -> Dict[str, str]:
    df = pd.read_csv(manifest_path)
    out: Dict[str, str] = {}

    if "code" not in df.columns:
        return out

    if "label" in df.columns:
        for _, r in df.iterrows():
            c = str(r["code"])
            lab = r.get("label")
            if pd.notna(lab) and str(lab).strip():
                out[c] = str(lab)

    return out


def read_global_top_features_from_shap(
    shap_values_path: Path,
    *,
    top_k: int = 15,
    code_to_label: Optional[Dict[str, str]] = None,
) -> List[GlobalTopFeature]:
    """Compute global top features by mean(|SHAP|) from a saved shap_values.parquet."""

    if not shap_values_path.exists():
        raise FileNotFoundError(f"SHAP values file not found: {shap_values_path}")

    df = pd.read_parquet(shap_values_path)
    if df.empty:
        return []

    scores = df.abs().mean(axis=0).sort_values(ascending=False).head(top_k)

    out: List[GlobalTopFeature] = []
    for feat, score in scores.items():
        label = None
        if code_to_label:
            # features may be one-hot expanded; try prefix match before exact
            if feat in code_to_label:
                label = code_to_label[feat]
            else:
                prefix = str(feat).split("_")[0]
                label = code_to_label.get(prefix)

        out.append(GlobalTopFeature(feature=str(feat), score=float(score), label=label))

    return out
