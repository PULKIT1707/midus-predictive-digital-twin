from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class TwinConfig:
    data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    manifest_path: Path
    join_id: str
    family_id: Optional[str]

    outcome_dataset: Path
    outcome_code: str

    prior_dataset: Path
    prior_code: str

    m1_survey_dataset: Path
    m2_survey_dataset: Path

    test_size: float
    random_state: int

    rf_params: Dict[str, Any]

    shap_enabled: bool
    shap_max_background: int
    shap_max_explain: int


def load_config(path: str | Path) -> TwinConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())

    data_dir = Path(raw["data_dir"])
    artifacts_dir = Path(raw["artifacts_dir"])
    models_dir = Path(raw["models_dir"])
    manifest_path = Path(raw["manifest_path"])

    join_id = str(raw["join_id"])
    family_id = raw.get("family_id")

    outcome_dataset = Path(raw["outcome"]["dataset"])
    outcome_code = str(raw["outcome"]["code"])

    prior_dataset = Path(raw["prior_cognition"]["dataset"])
    prior_code = str(raw["prior_cognition"]["code"])

    m1_survey_dataset = Path(raw["sources"]["m1_survey"]["dataset"])
    m2_survey_dataset = Path(raw["sources"]["m2_survey"]["dataset"])

    test_size = float(raw["split"]["test_size"])
    random_state = int(raw["split"]["random_state"])

    rf_params = dict(raw.get("model", {}).get("random_forest", {}))

    shap_cfg = raw.get("shap", {}) or {}
    shap_enabled = bool(shap_cfg.get("enabled", True))
    shap_max_background = int(shap_cfg.get("max_background", 500))
    shap_max_explain = int(shap_cfg.get("max_explain", 2000))

    return TwinConfig(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        manifest_path=manifest_path,
        join_id=join_id,
        family_id=str(family_id) if family_id else None,
        outcome_dataset=outcome_dataset,
        outcome_code=outcome_code,
        prior_dataset=prior_dataset,
        prior_code=prior_code,
        m1_survey_dataset=m1_survey_dataset,
        m2_survey_dataset=m2_survey_dataset,
        test_size=test_size,
        random_state=random_state,
        rf_params=rf_params,
        shap_enabled=shap_enabled,
        shap_max_background=shap_max_background,
        shap_max_explain=shap_max_explain,
    )
