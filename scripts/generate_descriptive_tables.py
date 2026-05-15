"""
Generate descriptive statistics tables stratified by demographics.

Outputs:
- Overall statistics
- Statistics by gender (Male/Female)
- Statistics by age groups
- Statistics by education level
- Combined stratification tables
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import pandas as pd
import numpy as np
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


def compute_descriptive_stats(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """
    Compute mean and SD for numeric columns.
    
    Args:
        df: DataFrame with data
        group_col: Optional column to group by
    
    Returns:
        DataFrame with Mean, SD, and N for each variable
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if group_col and group_col in numeric_cols:
        numeric_cols.remove(group_col)
    
    if group_col:
        grouped = df.groupby(group_col)
        
        results = []
        for name, group in grouped:
            stats = {
                'Group': name,
                'N': len(group)
            }
            
            for col in numeric_cols:
                valid_data = group[col].dropna()
                if len(valid_data) > 0:
                    stats[f'{col}_Mean'] = valid_data.mean()
                    stats[f'{col}_SD'] = valid_data.std()
                    stats[f'{col}_N'] = len(valid_data)
                else:
                    stats[f'{col}_Mean'] = np.nan
                    stats[f'{col}_SD'] = np.nan
                    stats[f'{col}_N'] = 0
            
            results.append(stats)
        
        return pd.DataFrame(results)
    
    else:
        stats = {'Group': 'Overall', 'N': len(df)}
        
        for col in numeric_cols:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                stats[f'{col}_Mean'] = valid_data.mean()
                stats[f'{col}_SD'] = valid_data.std()
                stats[f'{col}_N'] = len(valid_data)
            else:
                stats[f'{col}_Mean'] = np.nan
                stats[f'{col}_SD'] = np.nan
                stats[f'{col}_N'] = 0
        
        return pd.DataFrame([stats])


def format_table_for_publication(stats_df: pd.DataFrame, variable_labels: dict) -> pd.DataFrame:
    """
    Format statistics table for publication (Mean ± SD format).
    
    Args:
        stats_df: Raw statistics DataFrame
        variable_labels: Mapping of variable codes to labels
    
    Returns:
        Formatted DataFrame
    """
    var_cols = [c for c in stats_df.columns if c.endswith('_Mean')]
    variables = [c.replace('_Mean', '') for c in var_cols]
    
    formatted_rows = []
    
    for var in variables:
        row = {
            'Variable': variable_labels.get(var, var),
            'Code': var
        }
        
        for _, group_row in stats_df.iterrows():
            group_name = group_row['Group']
            mean = group_row.get(f'{var}_Mean', np.nan)
            sd = group_row.get(f'{var}_SD', np.nan)
            n = group_row.get(f'{var}_N', 0)
            
            if pd.notna(mean) and pd.notna(sd):
                row[f'{group_name}'] = f"{mean:.2f} ± {sd:.2f} (n={int(n)})"
            else:
                row[f'{group_name}'] = "N/A"
        
        formatted_rows.append(row)
    
    return pd.DataFrame(formatted_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate descriptive statistics tables stratified by demographics"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="primary",
        choices=["primary", "ablation"],
        help="Model variant to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/descriptive_tables",
        help="Output directory for tables"
    )
    
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
    
    print(f"Dataset built: {len(full_df)} rows, {len(full_df.columns)} columns")
    
    variable_labels = {}
    for _, row in manifest.iterrows():
        variable_labels[row['code']] = row['label']
    
    gender_col = None
    if 'A1PRSEX' in full_df.columns:
        gender_col = 'A1PRSEX'
    elif 'B1PRSEX' in full_df.columns:
        gender_col = 'B1PRSEX'
    
    age_col = None
    if 'B1PRAGE_2019' in full_df.columns:
        age_col = 'B1PRAGE_2019'
    elif 'A1PRAGE_2019' in full_df.columns:
        age_col = 'A1PRAGE_2019'
    
    edu_col = None
    if 'B1PB1' in full_df.columns:
        edu_col = 'B1PB1'
    elif 'A1PB1' in full_df.columns:
        edu_col = 'A1PB1'
    
    print("\n" + "="*60)
    print("Table 1: Overall Descriptive Statistics")
    print("="*60)
    
    overall_stats = compute_descriptive_stats(full_df)
    overall_formatted = format_table_for_publication(overall_stats, variable_labels)
    
    print(overall_formatted.to_string(index=False))
    overall_formatted.to_csv(output_dir / "table1_overall_statistics.csv", index=False)
    print(f"\nSaved to: {output_dir / 'table1_overall_statistics.csv'}")
    
    if gender_col:
        print("\n" + "="*60)
        print("Table 2: Descriptive Statistics by Gender")
        print("="*60)
        
        full_df['Gender'] = full_df[gender_col].map({1.0: 'Male', 2.0: 'Female'})
        
        gender_stats = compute_descriptive_stats(full_df, group_col='Gender')
        gender_formatted = format_table_for_publication(gender_stats, variable_labels)
        
        print(gender_formatted.to_string(index=False))
        gender_formatted.to_csv(output_dir / "table2_by_gender.csv", index=False)
        print(f"\nSaved to: {output_dir / 'table2_by_gender.csv'}")
    else:
        print("\nWarning: Gender variable not found in dataset")
    
    if age_col:
        print("\n" + "="*60)
        print("Table 3: Descriptive Statistics by Age Group")
        print("="*60)
        
        full_df['Age_Group'] = create_age_groups(full_df[age_col])
        
        age_stats = compute_descriptive_stats(full_df, group_col='Age_Group')
        age_formatted = format_table_for_publication(age_stats, variable_labels)
        
        print(age_formatted.to_string(index=False))
        age_formatted.to_csv(output_dir / "table3_by_age_group.csv", index=False)
        print(f"\nSaved to: {output_dir / 'table3_by_age_group.csv'}")
    else:
        print("\nWarning: Age variable not found in dataset")
    
    if edu_col:
        print("\n" + "="*60)
        print("Table 4: Descriptive Statistics by Education Level")
        print("="*60)
        
        full_df['Education_Group'] = pd.cut(
            full_df[edu_col],
            bins=[0, 6, 9, 12],
            labels=["High School or Less", "Some College", "College Degree+"],
            include_lowest=True
        )
        
        edu_stats = compute_descriptive_stats(full_df, group_col='Education_Group')
        edu_formatted = format_table_for_publication(edu_stats, variable_labels)
        
        print(edu_formatted.to_string(index=False))
        edu_formatted.to_csv(output_dir / "table4_by_education.csv", index=False)
        print(f"\nSaved to: {output_dir / 'table4_by_education.csv'}")
    else:
        print("\nWarning: Education variable not found in dataset")
    
    print("\n" + "="*60)
    print("Table 5: Key Variables Summary (for Publication)")
    print("="*60)
    
    key_vars = [
        cfg.outcome_code,
        cfg.prior_code if include_prior else None,
        age_col,
        gender_col,
        edu_col,
    ]
    key_vars = [v for v in key_vars if v and v in full_df.columns]
    
    key_df = full_df[key_vars + (['Gender'] if gender_col else [])]
    
    if gender_col:
        key_gender_stats = compute_descriptive_stats(key_df, group_col='Gender')
        key_formatted = format_table_for_publication(key_gender_stats, variable_labels)
        
        print(key_formatted.to_string(index=False))
        key_formatted.to_csv(output_dir / "table5_key_variables_by_gender.csv", index=False)
        print(f"\nSaved to: {output_dir / 'table5_key_variables_by_gender.csv'}")
    
    print("\n" + "="*60)
    print("Table 6: Model Performance by Demographics")
    print("="*60)
    
    predictions_path = cfg.models_dir / args.variant / "predictions.csv"
    if predictions_path.exists():
        pred_df = pd.read_csv(predictions_path)
        
        pred_df = pred_df.merge(
            full_df[[cfg.join_id, 'Gender', 'Age_Group'] if gender_col and age_col else [cfg.join_id]],
            on=cfg.join_id,
            how='left'
        )
        
        pred_df['Prediction_Error'] = pred_df['y_pred'] - pred_df['y_true']
        pred_df['Absolute_Error'] = pred_df['Prediction_Error'].abs()
        
        overall_perf = {
            'Group': 'Overall',
            'N': len(pred_df),
            'MAE': pred_df['Absolute_Error'].mean(),
            'RMSE': np.sqrt((pred_df['Prediction_Error']**2).mean()),
            'Mean_True': pred_df['y_true'].mean(),
            'Mean_Pred': pred_df['y_pred'].mean(),
        }
        
        perf_rows = [overall_perf]
        
        if gender_col and 'Gender' in pred_df.columns:
            for gender in pred_df['Gender'].dropna().unique():
                subset = pred_df[pred_df['Gender'] == gender]
                perf_rows.append({
                    'Group': f'Gender: {gender}',
                    'N': len(subset),
                    'MAE': subset['Absolute_Error'].mean(),
                    'RMSE': np.sqrt((subset['Prediction_Error']**2).mean()),
                    'Mean_True': subset['y_true'].mean(),
                    'Mean_Pred': subset['y_pred'].mean(),
                })
        
        if age_col and 'Age_Group' in pred_df.columns:
            for age_grp in pred_df['Age_Group'].dropna().unique():
                subset = pred_df[pred_df['Age_Group'] == age_grp]
                perf_rows.append({
                    'Group': f'Age: {age_grp}',
                    'N': len(subset),
                    'MAE': subset['Absolute_Error'].mean(),
                    'RMSE': np.sqrt((subset['Prediction_Error']**2).mean()),
                    'Mean_True': subset['y_true'].mean(),
                    'Mean_Pred': subset['y_pred'].mean(),
                })
        
        perf_df = pd.DataFrame(perf_rows)
        print(perf_df.to_string(index=False))
        perf_df.to_csv(output_dir / "table6_model_performance_by_demographics.csv", index=False)
        print(f"\nSaved to: {output_dir / 'table6_model_performance_by_demographics.csv'}")
    
    print("\n" + "="*60)
    print("All tables generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
