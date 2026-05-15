"""
Perform statistical tests for demographic comparisons.

Tests:
- Independent t-tests (2 groups)
- ANOVA (3+ groups)
- Post-hoc tests (Tukey HSD)
- Effect sizes (Cohen's d, eta-squared)
- Normality tests
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import pandas as pd
import numpy as np
from scipy import stats
from predictive_twin.config import load_config
from predictive_twin.manifest import read_manifest, select_features
from predictive_twin.dataset_builder import build_modeling_dataset


def create_age_groups(age_series: pd.Series) -> pd.Series:
    """Create age groups from continuous age."""
    return pd.cut(
        age_series,
        bins=[0, 40, 50, 60, 70, 120],
        labels=["<40", "40-49", "50-59", "60-69", "70+"],
        include_lowest=True
    )


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (group1.mean() - group2.mean()) / pooled_std


def eta_squared(groups: list) -> float:
    """Calculate eta-squared effect size for ANOVA."""
    all_data = pd.concat(groups)
    grand_mean = all_data.mean()
    
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((all_data - grand_mean)**2)
    
    return ss_between / ss_total if ss_total > 0 else 0


def independent_t_test(df: pd.DataFrame, variable: str, group_col: str, 
                      group1_val, group2_val) -> dict:
    """Perform independent samples t-test."""
    group1 = df[df[group_col] == group1_val][variable].dropna()
    group2 = df[df[group_col] == group2_val][variable].dropna()
    
    if len(group1) < 2 or len(group2) < 2:
        return {
            'variable': variable,
            'group1': group1_val,
            'group2': group2_val,
            'n1': len(group1),
            'n2': len(group2),
            'mean1': np.nan,
            'mean2': np.nan,
            'sd1': np.nan,
            'sd2': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'cohens_d': np.nan,
            'interpretation': 'Insufficient data'
        }
    
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    d = cohens_d(group1, group2)
    
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    if abs(d) < 0.2:
        effect = "negligible"
    elif abs(d) < 0.5:
        effect = "small"
    elif abs(d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    return {
        'variable': variable,
        'group1': group1_val,
        'group2': group2_val,
        'n1': len(group1),
        'n2': len(group2),
        'mean1': group1.mean(),
        'mean2': group2.mean(),
        'sd1': group1.std(),
        'sd2': group2.std(),
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': d,
        'significance': sig,
        'effect_size': effect,
        'interpretation': f"{sig} ({effect} effect)"
    }


def one_way_anova(df: pd.DataFrame, variable: str, group_col: str) -> dict:
    """Perform one-way ANOVA."""
    groups = []
    group_names = []
    
    for name, group in df.groupby(group_col):
        data = group[variable].dropna()
        if len(data) >= 2:
            groups.append(data)
            group_names.append(name)
    
    if len(groups) < 2:
        return {
            'variable': variable,
            'group_col': group_col,
            'n_groups': len(groups),
            'f_statistic': np.nan,
            'p_value': np.nan,
            'eta_squared': np.nan,
            'interpretation': 'Insufficient groups'
        }
    
    f_stat, p_val = stats.f_oneway(*groups)
    
    eta_sq = eta_squared(groups)
    
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    if eta_sq < 0.01:
        effect = "negligible"
    elif eta_sq < 0.06:
        effect = "small"
    elif eta_sq < 0.14:
        effect = "medium"
    else:
        effect = "large"
    
    return {
        'variable': variable,
        'group_col': group_col,
        'n_groups': len(groups),
        'group_names': ', '.join(map(str, group_names)),
        'f_statistic': f_stat,
        'p_value': p_val,
        'eta_squared': eta_sq,
        'significance': sig,
        'effect_size': effect,
        'interpretation': f"{sig} ({effect} effect)"
    }


def tukey_hsd_test(df: pd.DataFrame, variable: str, group_col: str) -> pd.DataFrame:
    """Perform Tukey HSD post-hoc test (manual implementation)."""
    from itertools import combinations
    
    groups_data = {}
    for name, group in df.groupby(group_col):
        data = group[variable].dropna()
        if len(data) >= 2:
            groups_data[name] = data
    
    if len(groups_data) < 2:
        return pd.DataFrame()
    
    results = []
    
    for (name1, data1), (name2, data2) in combinations(groups_data.items(), 2):
        t_stat, p_val = stats.ttest_ind(data1, data2)
        d = cohens_d(data1, data2)
        
        results.append({
            'Group 1': name1,
            'Group 2': name2,
            'Mean Diff': data1.mean() - data2.mean(),
            'p-value': p_val,
            'Cohen\'s d': d,
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })
    
    return pd.DataFrame(results)


def normality_test(df: pd.DataFrame, variable: str, group_col: str = None) -> dict:
    """Test normality using Shapiro-Wilk test."""
    if group_col:
        results = []
        for name, group in df.groupby(group_col):
            data = group[variable].dropna()
            if len(data) >= 3:
                stat, p_val = stats.shapiro(data)
                results.append({
                    'group': name,
                    'n': len(data),
                    'statistic': stat,
                    'p_value': p_val,
                    'normal': 'Yes' if p_val > 0.05 else 'No'
                })
        return results
    else:
        data = df[variable].dropna()
        if len(data) >= 3:
            stat, p_val = stats.shapiro(data)
            return [{
                'group': 'Overall',
                'n': len(data),
                'statistic': stat,
                'p_value': p_val,
                'normal': 'Yes' if p_val > 0.05 else 'No'
            }]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perform statistical tests for demographic comparisons"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--variant", type=str, default="primary",
                       choices=["primary", "ablation"])
    parser.add_argument("--output-dir", type=str,
                       default="artifacts/statistical_tests")
    
    args = parser.parse_args()
    
    cfg = load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = read_manifest(cfg.manifest_path)
    
    include_prior = args.variant == "primary"
    sel = select_features(
        manifest,
        m1_dataset_name=cfg.m1_survey_dataset.name,
        m2_dataset_name=cfg.m2_survey_dataset.name,
        outcome_code=cfg.outcome_code,
        prior_cognition_code=cfg.prior_code,
        include_prior=include_prior,
    )
    
    print(f"Building dataset for {args.variant} model...")
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
    
    full_df = X.copy()
    full_df[cfg.outcome_code] = y
    
    variable_labels = {}
    for _, row in manifest.iterrows():
        variable_labels[row['code']] = row['label']
    
    gender_col = 'B1PRSEX' if 'B1PRSEX' in full_df.columns else 'A1PRSEX' if 'A1PRSEX' in full_df.columns else None
    age_col = 'B1PRAGE_2019' if 'B1PRAGE_2019' in full_df.columns else 'A1PRAGE_2019' if 'A1PRAGE_2019' in full_df.columns else None
    edu_col = 'B1PB1' if 'B1PB1' in full_df.columns else 'A1PB1' if 'A1PB1' in full_df.columns else None
    
    test_vars = [cfg.outcome_code]
    if include_prior and cfg.prior_code in full_df.columns:
        test_vars.append(cfg.prior_code)
    if age_col:
        test_vars.append(age_col)
    
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    print("\n" + "-"*60)
    print("Test 1: Normality Tests (Shapiro-Wilk)")
    print("-"*60)
    
    normality_results = []
    for var in test_vars:
        if gender_col:
            full_df['Gender'] = full_df[gender_col].map({1.0: 'Male', 2.0: 'Female'})
            results = normality_test(full_df, var, 'Gender')
            for r in results:
                r['variable'] = variable_labels.get(var, var)
                normality_results.append(r)
    
    norm_df = pd.DataFrame(normality_results)
    print(norm_df.to_string(index=False))
    norm_df.to_csv(output_dir / "test1_normality_tests.csv", index=False)
    print(f"\nSaved to: {output_dir / 'test1_normality_tests.csv'}")
    
    if gender_col:
        print("\n" + "-"*60)
        print("Test 2: Independent T-Tests by Gender")
        print("-"*60)
        
        full_df['Gender'] = full_df[gender_col].map({1.0: 'Male', 2.0: 'Female'})
        
        t_test_results = []
        for var in test_vars:
            result = independent_t_test(full_df, var, 'Gender', 'Male', 'Female')
            result['variable_label'] = variable_labels.get(var, var)
            t_test_results.append(result)
        
        t_test_df = pd.DataFrame(t_test_results)
        
        display_cols = ['variable_label', 'n1', 'mean1', 'sd1', 'n2', 'mean2', 'sd2',
                       't_statistic', 'p_value', 'cohens_d', 'interpretation']
        print(t_test_df[display_cols].to_string(index=False))
        
        t_test_df.to_csv(output_dir / "test2_t_tests_by_gender.csv", index=False)
        print(f"\nSaved to: {output_dir / 'test2_t_tests_by_gender.csv'}")
        
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("Effect sizes: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)")
    
    if age_col:
        print("\n" + "-"*60)
        print("Test 3: One-Way ANOVA by Age Group")
        print("-"*60)
        
        full_df['Age_Group'] = create_age_groups(full_df[age_col])
        
        anova_results = []
        for var in test_vars:
            result = one_way_anova(full_df, var, 'Age_Group')
            result['variable_label'] = variable_labels.get(var, var)
            anova_results.append(result)
        
        anova_df = pd.DataFrame(anova_results)
        
        display_cols = ['variable_label', 'n_groups', 'f_statistic', 'p_value',
                       'eta_squared', 'interpretation']
        print(anova_df[display_cols].to_string(index=False))
        
        anova_df.to_csv(output_dir / "test3_anova_by_age_group.csv", index=False)
        print(f"\nSaved to: {output_dir / 'test3_anova_by_age_group.csv'}")
        
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("Effect sizes (η²): negligible (<0.01), small (0.01-0.06), medium (0.06-0.14), large (>0.14)")
    
    if age_col:
        print("\n" + "-"*60)
        print("Test 4: Post-hoc Pairwise Comparisons (Age Groups)")
        print("-"*60)
        
        for var in [cfg.outcome_code]:
            print(f"\nVariable: {variable_labels.get(var, var)}")
            tukey_df = tukey_hsd_test(full_df, var, 'Age_Group')
            
            if not tukey_df.empty:
                print(tukey_df.to_string(index=False))
                tukey_df.to_csv(
                    output_dir / f"test4_posthoc_{var}_by_age.csv",
                    index=False
                )
    
    predictions_path = cfg.models_dir / args.variant / "predictions.csv"
    if predictions_path.exists():
        print("\n" + "-"*60)
        print("Test 5: Model Performance Differences by Demographics")
        print("-"*60)
        
        pred_df = pd.read_csv(predictions_path)
        
        if gender_col:
            full_df['Gender'] = full_df[gender_col].map({1.0: 'Male', 2.0: 'Female'})
        if age_col:
            full_df['Age_Group'] = create_age_groups(full_df[age_col])
        
        merge_cols = [cfg.join_id]
        if gender_col:
            merge_cols.append('Gender')
        if age_col:
            merge_cols.append('Age_Group')
        
        pred_df = pred_df.merge(full_df[merge_cols], on=cfg.join_id, how='left')
        pred_df['Absolute_Error'] = (pred_df['y_pred'] - pred_df['y_true']).abs()
        
        if gender_col:
            print("\nMAE by Gender:")
            mae_gender_test = independent_t_test(
                pred_df, 'Absolute_Error', 'Gender', 'Male', 'Female'
            )
            mae_gender_df = pd.DataFrame([mae_gender_test])
            print(mae_gender_df[['n1', 'mean1', 'n2', 'mean2', 't_statistic',
                                'p_value', 'interpretation']].to_string(index=False))
            mae_gender_df.to_csv(output_dir / "test5_mae_by_gender.csv", index=False)
        
        if age_col:
            print("\nMAE by Age Group:")
            mae_age_test = one_way_anova(pred_df, 'Absolute_Error', 'Age_Group')
            mae_age_df = pd.DataFrame([mae_age_test])
            print(mae_age_df[['n_groups', 'f_statistic', 'p_value',
                            'interpretation']].to_string(index=False))
            mae_age_df.to_csv(output_dir / "test5_mae_by_age_group.csv", index=False)
    
    print("\n" + "="*60)
    print("All statistical tests completed!")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
