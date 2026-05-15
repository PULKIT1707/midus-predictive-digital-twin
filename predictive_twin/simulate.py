from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationResult:
    baseline_prediction: float
    modified_prediction: float
    delta: float


@dataclass(frozen=True)
class FeatureChange:
    feature: str
    old_value: Any
    new_value: Any
    rule: str


@dataclass(frozen=True)
class ScenarioResult:
    scenario_key: str
    scenario_label: str
    baseline_prediction: float
    new_prediction: float
    delta: float
    changes: Tuple[FeatureChange, ...]


@dataclass(frozen=True)
class FeatureRule:
    code: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def clamp(self, value: Any) -> Any:
        if value is None:
            return value
        try:
            v = float(value)
        except Exception:
            return value

        if self.min_value is not None:
            v = max(v, float(self.min_value))
        if self.max_value is not None:
            v = min(v, float(self.max_value))

        # preserve ints if the original is integer-like
        if float(v).is_integer():
            return int(v)
        return v


def default_feature_rules() -> Dict[str, FeatureRule]:
    # Rules are intentionally conservative and based on typical MIDUS codings seen in catalogs.
    # Values are clamped to prevent obviously invalid entries.
    return {
        # Alcohol (drinks when drank most)
        "A1PA55": FeatureRule("A1PA55", min_value=0, max_value=50),
        "B1PA55": FeatureRule("B1PA55", min_value=0, max_value=50),
        # Stress / worry / depressive-like duration
        "A1PA81": FeatureRule("A1PA81", min_value=0, max_value=52),
        "B1PA81": FeatureRule("B1PA81", min_value=0, max_value=52),
        # Self-rated health (often 1-5)
        "A1PA4": FeatureRule("A1PA4", min_value=1, max_value=5),
        "B1PA4": FeatureRule("B1PA4", min_value=0, max_value=30),
    }


def validate_features_exist(x_row: pd.DataFrame, required: Iterable[str], *, context: str) -> None:
    missing = [c for c in required if c not in x_row.columns]
    if missing:
        raise KeyError(f"Missing required features for {context}: {missing}")


def _safe_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def apply_percent_reduction(
    *,
    x_row: pd.DataFrame,
    feature: str,
    percent: float,
    rules: Mapping[str, FeatureRule],
    changes: list[FeatureChange],
    rule_name: str,
) -> None:
    old = x_row.iloc[0][feature]
    old_num = _safe_numeric(old)
    if old_num is None:
        return

    new = old_num * (1.0 - percent)
    if feature in rules:
        new = rules[feature].clamp(new)

    if new != old:
        x_row.iloc[0, x_row.columns.get_loc(feature)] = new
        changes.append(FeatureChange(feature=feature, old_value=old, new_value=new, rule=rule_name))


def apply_additive_change(
    *,
    x_row: pd.DataFrame,
    feature: str,
    delta: float,
    rules: Mapping[str, FeatureRule],
    changes: list[FeatureChange],
    rule_name: str,
) -> None:
    old = x_row.iloc[0][feature]
    old_num = _safe_numeric(old)
    if old_num is None:
        return

    new = old_num + delta
    if feature in rules:
        new = rules[feature].clamp(new)

    if new != old:
        x_row.iloc[0, x_row.columns.get_loc(feature)] = new
        changes.append(FeatureChange(feature=feature, old_value=old, new_value=new, rule=rule_name))


def apply_set_value(
    *,
    x_row: pd.DataFrame,
    feature: str,
    value: Any,
    rules: Mapping[str, FeatureRule],
    changes: list[FeatureChange],
    rule_name: str,
) -> None:
    old = x_row.iloc[0][feature]
    new = value
    if feature in rules:
        new = rules[feature].clamp(new)

    if new != old:
        x_row.iloc[0, x_row.columns.get_loc(feature)] = new
        changes.append(FeatureChange(feature=feature, old_value=old, new_value=new, rule=rule_name))


def scenario_library() -> Dict[str, Dict[str, Any]]:
    return {
        "reduce_alcohol": {
            "label": "Reduce alcohol consumption",
            "features": ["A1PA55", "B1PA55"],
        },
        "reduce_stress": {
            "label": "Reduce stress / worry",
            "features": ["A1PA81", "B1PA81"],
        },
        "improve_self_rated_health": {
            "label": "Improve self-rated health",
            "features": ["A1PA4"],
        },
        "combined_lifestyle": {
            "label": "Combined lifestyle improvement",
            "features": ["A1PA55", "B1PA55", "A1PA81", "B1PA81", "A1PA4"],
        },
    }


def run_scenario(
    *,
    pipeline: Any,
    x_row: pd.DataFrame,
    scenario_key: str,
    rules: Optional[Mapping[str, FeatureRule]] = None,
) -> ScenarioResult:
    if len(x_row) != 1:
        raise ValueError("x_row must be a single-row DataFrame")

    lib = scenario_library()
    if scenario_key not in lib:
        raise KeyError(f"Unknown scenario: {scenario_key}. Available: {sorted(lib.keys())}")

    rules_map = dict(default_feature_rules())
    if rules:
        rules_map.update(dict(rules))

    spec = lib[scenario_key]
    label = str(spec["label"])
    allowed_features = [f for f in spec["features"] if f in x_row.columns]

    if not allowed_features:
        raise KeyError(
            f"Scenario '{scenario_key}' has no applicable features in this dataset row. "
            f"Expected one of: {spec['features']}"
        )

    baseline = float(pipeline.predict(x_row)[0])

    mod = x_row.copy()
    changes: list[FeatureChange] = []

    # Apply scenario-specific safe rules
    if scenario_key == "reduce_alcohol":
        for f in allowed_features:
            apply_percent_reduction(
                x_row=mod,
                feature=f,
                percent=0.3,
                rules=rules_map,
                changes=changes,
                rule_name="reduce_by_30pct_and_clamp",
            )
    elif scenario_key == "reduce_stress":
        for f in allowed_features:
            apply_percent_reduction(
                x_row=mod,
                feature=f,
                percent=0.25,
                rules=rules_map,
                changes=changes,
                rule_name="reduce_by_25pct_and_clamp",
            )
    elif scenario_key == "improve_self_rated_health":
        # For MIDUS self-rated health, lower numeric often means better in some codings, but we can't assume.
        # We implement a conservative bounded shift toward the midpoint.
        f = allowed_features[0]
        old = mod.iloc[0][f]
        old_num = _safe_numeric(old)
        if old_num is not None:
            # Move 1 step toward "healthier" (lower value) but clamp to [1,5]
            apply_additive_change(
                x_row=mod,
                feature=f,
                delta=-1.0,
                rules=rules_map,
                changes=changes,
                rule_name="minus_1_step_and_clamp",
            )
    elif scenario_key == "combined_lifestyle":
        # Compose the above
        for f in ["A1PA55", "B1PA55"]:
            if f in mod.columns:
                apply_percent_reduction(
                    x_row=mod,
                    feature=f,
                    percent=0.3,
                    rules=rules_map,
                    changes=changes,
                    rule_name="reduce_by_30pct_and_clamp",
                )
        for f in ["A1PA81", "B1PA81"]:
            if f in mod.columns:
                apply_percent_reduction(
                    x_row=mod,
                    feature=f,
                    percent=0.25,
                    rules=rules_map,
                    changes=changes,
                    rule_name="reduce_by_25pct_and_clamp",
                )
        if "A1PA4" in mod.columns:
            apply_additive_change(
                x_row=mod,
                feature="A1PA4",
                delta=-1.0,
                rules=rules_map,
                changes=changes,
                rule_name="minus_1_step_and_clamp",
            )
    else:
        raise KeyError(f"Scenario handler not implemented: {scenario_key}")

    if not changes:
        raise ValueError(
            f"Scenario '{scenario_key}' did not apply any changes (values may be missing or non-numeric)."
        )

    for c in changes:
        logger.info("Scenario %s changed %s: %r -> %r (%s)", scenario_key, c.feature, c.old_value, c.new_value, c.rule)

    new_pred = float(pipeline.predict(mod)[0])
    delta = float(new_pred - baseline)

    return ScenarioResult(
        scenario_key=scenario_key,
        scenario_label=label,
        baseline_prediction=baseline,
        new_prediction=new_pred,
        delta=delta,
        changes=tuple(changes),
    )


def simulate_what_if(
    *,
    pipeline: Any,
    x_row: pd.DataFrame,
    modifications: Mapping[str, Any],
    rules: Optional[Mapping[str, FeatureRule]] = None,
) -> Tuple[pd.DataFrame, SimulationResult]:
    """Apply modifications to a single-row feature frame and compare predictions.

    Args:
        pipeline: trained sklearn Pipeline
        x_row: single-row DataFrame of predictors
        modifications: mapping of feature_name -> new_value (must match columns of x_row)

    Returns:
        (modified_row, result)
    """

    if len(x_row) != 1:
        raise ValueError("x_row must be a single-row DataFrame")

    baseline_pred = float(pipeline.predict(x_row)[0])

    mod = x_row.copy()
    rules_map = dict(default_feature_rules())
    if rules:
        rules_map.update(dict(rules))
    for k, v in modifications.items():
        if k not in mod.columns:
            raise KeyError(f"Modification key not in features: {k}")
        vv = rules_map[k].clamp(v) if k in rules_map else v
        mod.iloc[0, mod.columns.get_loc(k)] = vv

    modified_pred = float(pipeline.predict(mod)[0])

    res = SimulationResult(
        baseline_prediction=baseline_pred,
        modified_prediction=modified_pred,
        delta=float(modified_pred - baseline_pred),
    )

    return mod, res
