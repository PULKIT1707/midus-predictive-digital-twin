from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from predictive_twin.preprocess import PreprocessSpec, build_preprocess_pipeline


def build_rf_pipeline(spec: PreprocessSpec, rf_params: Dict[str, Any]) -> Pipeline:
    preprocess = build_preprocess_pipeline(spec)
    model = RandomForestRegressor(**rf_params)

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
