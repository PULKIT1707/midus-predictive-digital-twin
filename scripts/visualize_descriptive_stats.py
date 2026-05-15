"""
Generate visualizations for descriptive statistics.

Creates:
- Bar charts for mean comparisons by demographics
- Box plots for distribution comparisons
- Violin plots for outcome distributions
- Heatmaps for correlation matrices
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predictive_twin.config import load_config
from predictive_twin.manifest import read_manifest, select_features
from predictive_twin.dataset_builder import build_modeling_dataset

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def create_age_groups(age_series: pd.Series) -> pd.Series:
    """Create age groups from continuous age."""
    return pd.cut(
        age_series,
        bins=[0, 40, 50, 60, 70, 120],
        labels=["<40", "40-49", "50-59", "60-69", "70+"],
        include_lowest=True
    )


def plot_outcome_by_gender(df: pd.DataFrame, outcome_col: str, gender_col: str, 
                           output_path: Path, outcome_label: str):
    """Box plot of outcome by gender."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    df_plot = df.copy()
    df_plot['Gender'] = df_plot[gender_col].map({1.0: 'Male', 2.0: 'Female'})
    df_plot = df_plot.dropna(subset=['Gender', outcome_col])
    
    positions = [1, 2]
    data_to_plot = [
        df_plot[df_plot['Gender'] == 'Male'][outcome_col].dropna(),
        df_plot[df_plot['Gender'] == 'Female'][outcome_col].dropna()
    ]
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Male', 'Female'])
    ax.set_ylabel(outcome_label, fontweight='bold')
    ax.set_xlabel('Gender', fontweight='bold')
    ax.set_title(f'{outcome_label} by Gender', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
        ax.text(pos, ax.get_ylim()[0] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'n={len(data)}', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_outcome_by_age_group(df: pd.DataFrame, outcome_col: str, age_col: str,
                              output_path: Path, outcome_label: str):
    """Box plot of outcome by age group."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_plot = df.copy()
    df_plot['Age_Group'] = create_age_groups(df_plot[age_col])
    df_plot = df_plot.dropna(subset=['Age_Group', outcome_col])
    
    age_groups = ["<40", "40-49", "50-59", "60-69", "70+"]
    data_to_plot = [df_plot[df_plot['Age_Group'] == ag][outcome_col].dropna() 
                    for ag in age_groups]
    
    positions = range(1, len(age_groups) + 1)
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(age_groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(age_groups)
    ax.set_ylabel(outcome_label, fontweight='bold')
    ax.set_xlabel('Age Group (years)', fontweight='bold')
    ax.set_title(f'{outcome_label} by Age Group', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
        ax.text(pos, ax.get_ylim()[0] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'n={len(data)}', ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_performance_by_demographics(predictions_df: pd.DataFrame, 
                                          output_path: Path):
    """Bar chart of model performance (MAE) by demographic groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'Gender' in predictions_df.columns:
        gender_perf = predictions_df.groupby('Gender').agg({
            'Absolute_Error': 'mean',
            'y_true': 'count'
        }).reset_index()
        gender_perf.columns = ['Gender', 'MAE', 'N']
        
        bars1 = ax1.bar(gender_perf['Gender'], gender_perf['MAE'], 
                       color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontweight='bold')
        ax1.set_xlabel('Gender', fontweight='bold')
        ax1.set_title('Model Performance by Gender', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, mae, n in zip(bars1, gender_perf['MAE'], gender_perf['N']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae:.3f}\n(n={int(n)})', ha='center', va='bottom', fontsize=9)
    
    if 'Age_Group' in predictions_df.columns:
        age_perf = predictions_df.groupby('Age_Group').agg({
            'Absolute_Error': 'mean',
            'y_true': 'count'
        }).reset_index()
        age_perf.columns = ['Age_Group', 'MAE', 'N']
        age_perf = age_perf.sort_values('Age_Group')
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(age_perf)))
        bars2 = ax2.bar(range(len(age_perf)), age_perf['MAE'], 
                       color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(age_perf)))
        ax2.set_xticklabels(age_perf['Age_Group'], rotation=45)
        ax2.set_ylabel('Mean Absolute Error (MAE)', fontweight='bold')
        ax2.set_xlabel('Age Group', fontweight='bold')
        ax2.set_title('Model Performance by Age Group', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, mae, n in zip(bars2, age_perf['MAE'], age_perf['N']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae:.3f}\n(n={int(n)})', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate visualizations for descriptive statistics"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--variant", type=str, default="primary", 
                       choices=["primary", "ablation"])
    parser.add_argument("--output-dir", type=str, 
                       default="artifacts/descriptive_visualizations")
    
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
    
    outcome_label = "M3 Cognition Score"
    
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    if gender_col:
        plot_outcome_by_gender(
            full_df, cfg.outcome_code, gender_col,
            output_dir / "fig1_outcome_by_gender_boxplot.png",
            outcome_label
        )
    
    if age_col:
        plot_outcome_by_age_group(
            full_df, cfg.outcome_code, age_col,
            output_dir / "fig2_outcome_by_age_group.png",
            outcome_label
        )
    
    predictions_path = cfg.models_dir / args.variant / "predictions.csv"
    if predictions_path.exists():
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
        
        plot_model_performance_by_demographics(
            pred_df,
            output_dir / "fig3_model_performance_by_demographics.png"
        )
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
