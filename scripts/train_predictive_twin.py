from __future__ import annotations

import argparse
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from predictive_twin.config import load_config
from predictive_twin.dataset_builder import build_modeling_dataset
from predictive_twin.evaluate import compute_metrics, metrics_to_dict
from predictive_twin.explain_shap import explain_tree_model
from predictive_twin.leakage import validate_no_m3_predictors
from predictive_twin.manifest import read_manifest, select_features
from predictive_twin.modeling import build_rf_pipeline
from predictive_twin.persist import save_json, save_model
from predictive_twin.preprocess import infer_feature_types


def _train_variant(*, variant: str, include_prior: bool, cfg_path: Path) -> None:
    cfg = load_config(cfg_path)

    manifest = read_manifest(cfg.manifest_path)
    violations = validate_no_m3_predictors(manifest, outcome_code=cfg.outcome_code)
    if violations:
        raise SystemExit(f"Leakage violations in manifest: {len(violations)}. Example: {violations[0]}")

    sel = select_features(
        manifest,
        m1_dataset_name=cfg.m1_survey_dataset.name,
        m2_dataset_name=cfg.m2_survey_dataset.name,
        outcome_code=cfg.outcome_code,
        prior_cognition_code=cfg.prior_code,
        include_prior=include_prior,
    )

    n_pred = len(sel.predictors_m1) + len(sel.predictors_m2)
    if n_pred == 0:
        raise SystemExit(
            "Manifest selection produced 0 predictors. "
            "Curate the manifest to include at least one M1/M2 predictor (Tier 1), "
            "then re-run training."
        )

    if not include_prior and (sel.prior_cognition_code is None) and (cfg.prior_code in (sel.predictors_m1 + sel.predictors_m2)):
        # Should not happen, but keep intent explicit.
        raise SystemExit("Ablation variant should exclude prior cognition, but it was still selected.")

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
    X_model = X.drop(columns=[join_id]) if join_id in X.columns else X

    if X_model.shape[1] == 0:
        raise SystemExit(
            f"Variant '{variant}' has 0 feature columns after dropping join_id. "
            "This usually means your manifest only includes the prior cognition feature and/or outcome. "
            "Add Tier 1 predictors to the manifest and re-run."
        )

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_model,
        y,
        ids_df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    spec = infer_feature_types(X_model, join_id="__unused__")
    pipeline = build_rf_pipeline(spec, cfg.rf_params)

    pipeline.fit(X_train, y_train)

    pred_test = pipeline.predict(X_test)
    m = compute_metrics(y_test, pred_test)

    out_dir = cfg.models_dir / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    save_model(pipeline, out_dir / "model.joblib")
    save_json(metrics_to_dict(m), out_dir / "metrics.json")

    # Save join report
    save_json(
        {
            "join_id": report.join_id,
            "m1_rows": report.m1_rows,
            "m2_rows": report.m2_rows,
            "m3_outcome_rows": report.m3_outcome_rows,
            "after_m1_m2_rows": report.after_m1_m2_rows,
            "after_all_rows": report.after_all_rows,
            "overlap_m1_m2": report.overlap_m1_m2,
            "overlap_m1_m3": report.overlap_m1_m3,
            "overlap_m2_m3": report.overlap_m2_m3,
            "one_to_one_ok": report.one_to_one_ok,
        },
        out_dir / "join_report.json",
    )

    # Save predictions for inspection
    pred_df = ids_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = pred_test
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # Optional SHAP
    if cfg.shap_enabled:
        X_bg = X_train
        if len(X_bg) > cfg.shap_max_background:
            X_bg = X_bg.sample(cfg.shap_max_background, random_state=0)

        X_exp = X_test
        if len(X_exp) > cfg.shap_max_explain:
            X_exp = X_exp.sample(cfg.shap_max_explain, random_state=0)

        shap_dir = out_dir / "shap"
        explain_tree_model(
            pipeline=pipeline,
            X_train=X_bg,
            X_explain=X_exp,
            out_dir=shap_dir,
            max_background=cfg.shap_max_background,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train predictive twin models (primary + ablation)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)

    _train_variant(variant="primary", include_prior=True, cfg_path=cfg_path)
    _train_variant(variant="ablation", include_prior=False, cfg_path=cfg_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
