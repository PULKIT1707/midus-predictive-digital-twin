from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error  # type: ignore
except Exception:  # pragma: no cover
    root_mean_squared_error = None


@dataclass(frozen=True)
class Metrics:
    r2: float
    mae: float
    rmse: float


def compute_metrics(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if root_mean_squared_error is not None:
        rmse = float(root_mean_squared_error(y_true, y_pred))
    else:
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))

    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=rmse,
    )


def metrics_to_dict(m: Metrics) -> Dict[str, float]:
    return {"r2": m.r2, "mae": m.mae, "rmse": m.rmse}
