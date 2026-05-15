from __future__ import annotations

import argparse
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from predictive_twin.evaluate import compute_metrics, metrics_to_dict
from predictive_twin.persist import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a persisted predictive twin model on a provided predictions.csv")
    parser.add_argument("--variant", type=str, default="primary", choices=["primary", "ablation"])
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions.csv; defaults to models/<variant>/predictions.csv")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    pred_path = Path(args.predictions) if args.predictions else models_dir / args.variant / "predictions.csv"

    df = pd.read_csv(pred_path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise SystemExit("predictions.csv must contain y_true and y_pred")

    m = compute_metrics(df["y_true"], df["y_pred"])
    save_json(metrics_to_dict(m), models_dir / args.variant / "metrics_recomputed.json")

    print(metrics_to_dict(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
