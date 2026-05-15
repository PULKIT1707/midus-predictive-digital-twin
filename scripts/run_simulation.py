from __future__ import annotations

import argparse
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from predictive_twin.config import load_config
from predictive_twin.dataset_builder import build_modeling_dataset
from predictive_twin.manifest import read_manifest, select_features
from predictive_twin.persist import load_model
from predictive_twin.simulate import simulate_what_if


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a what-if simulation for a single subject")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--variant", type=str, default="primary", choices=["primary", "ablation"])
    parser.add_argument("--id", type=str, required=True, help="Value of join_id for the subject")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Modification in the form CODE=VALUE. Can be provided multiple times.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    include_prior = args.variant == "primary"

    manifest = read_manifest(cfg.manifest_path)
    sel = select_features(
        manifest,
        m1_dataset_name=cfg.m1_survey_dataset.name,
        m2_dataset_name=cfg.m2_survey_dataset.name,
        outcome_code=cfg.outcome_code,
        prior_cognition_code=cfg.prior_code,
        include_prior=include_prior,
    )

    X, y, ids_df, report = build_modeling_dataset(
        m1_survey_path=cfg.m1_survey_dataset,
        m2_survey_path=cfg.m2_survey_dataset,
        m2_prior_path=cfg.prior_dataset,
        m3_outcome_path=cfg.outcome_dataset,
        join_id=cfg.join_id,
        predictors_m1=sel.predictors_m1,
        predictors_m2=sel.predictors_m2,
        outcome_code=cfg.outcome_code,
    )

    join_id = cfg.join_id
    subject_mask = ids_df[join_id].astype(str).eq(str(args.id))
    if not subject_mask.any():
        raise SystemExit(f"No subject found with {join_id}={args.id}")

    x_row = X.loc[subject_mask].drop(columns=[cfg.outcome_code], errors="ignore")
    if join_id in x_row.columns:
        x_row = x_row.drop(columns=[join_id])

    modifications = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"Bad --set value (expected CODE=VALUE): {item}")
        k, v = item.split("=", 1)
        # best-effort numeric casting
        try:
            vv = float(v)
            if vv.is_integer():
                vv = int(vv)
            modifications[k] = vv
        except Exception:
            modifications[k] = v

    model = load_model(cfg.models_dir / args.variant / "model.joblib")
    mod_row, res = simulate_what_if(pipeline=model, x_row=x_row, modifications=modifications)

    print("baseline_prediction", res.baseline_prediction)
    print("modified_prediction", res.modified_prediction)
    print("delta", res.delta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
