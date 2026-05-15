from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _get_feature_names(pipeline: Any) -> list[str]:
    # sklearn Pipeline: preprocess step is a ColumnTransformer
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is None:
        return []

    try:
        names = preprocess.get_feature_names_out().tolist()
    except Exception:
        names = []
    return [str(n) for n in names]


def explain_tree_model(
    *,
    pipeline: Any,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    out_dir: Path,
    max_background: int = 500,
) -> Tuple[pd.DataFrame, Path]:
    """Generate SHAP values for a fitted pipeline containing a tree model.

    Writes:
      - shap_values.parquet
      - shap_summary.png

    Returns:
      (shap_long_df, summary_png_path)
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # Transform to model input space
    X_train_t = preprocess.transform(X_train)
    X_explain_t = preprocess.transform(X_explain)

    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
    if hasattr(X_explain_t, "toarray"):
        X_explain_t = X_explain_t.toarray()

    X_train_t = np.asarray(X_train_t)
    X_explain_t = np.asarray(X_explain_t)

    bg = X_train_t
    if bg.shape[0] > max_background:
        bg = shap.sample(bg, max_background, random_state=0)

    explainer = shap.TreeExplainer(model, data=bg)
    shap_vals = explainer.shap_values(X_explain_t)

    feature_names = _get_feature_names(pipeline)
    if not feature_names:
        feature_names = [f"f{i}" for i in range(X_explain_t.shape[1])]

    # Save long-format SHAP values (row_id, feature, shap_value)
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df.to_parquet(out_dir / "shap_values.parquet", index=False)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, features=X_explain_t, feature_names=feature_names, show=False)
    summary_path = out_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=180)
    plt.close()

    return shap_df, summary_path
