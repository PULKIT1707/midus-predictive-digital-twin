from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class PreprocessSpec:
    numeric_features: List[str]
    categorical_features: List[str]


def infer_feature_types(X: pd.DataFrame, *, join_id: str) -> PreprocessSpec:
    cols = [c for c in X.columns if c != join_id]
    numeric = []
    categorical = []

    for c in cols:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        else:
            categorical.append(c)

    return PreprocessSpec(numeric_features=numeric, categorical_features=categorical)


def build_preprocess_pipeline(spec: PreprocessSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric_features),
            ("cat", categorical_pipe, spec.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
