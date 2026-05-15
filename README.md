# MIDUS Predictive Digital Twin Pipeline

A research-grade predictive digital twin system for cognitive aging research using the MIDUS (Midlife in the United States) longitudinal dataset. The system predicts future cognitive outcomes based on historical health, lifestyle, and demographic data.

## Overview

**Research Question**: Can we predict cognitive function at MIDUS Wave 3 (M3) using data from Waves 1 and 2 (M1, M2)?

**Outcome**: M3 BTACT Composite Score (Brief Test of Adult Cognition by Telephone)

**Sample Size**: N = 2,735 participants with complete data across all three waves

**Algorithm**: Random Forest Regression with SHAP explainability

## Model Performance

### Primary Model (With Prior Cognition)
- **R² = 0.575** (57.5% of variance explained)
- **MAE = 0.344** (average error of 0.34 standard deviations)
- **RMSE = 0.434**
- Includes M2 BTACT composite z-score as predictor

### Ablation Model (Without Prior Cognition)
- **R² = 0.277**
- **MAE = 0.449**
- **RMSE = 0.566**
- Excludes prior cognition to test its importance

**Key Finding**: Prior cognition alone accounts for ~30% of predictive power, demonstrating cognitive trajectory stability over time.

## Key Features

### Leakage Prevention System
- **Manifest-driven feature selection**: Every feature explicitly labeled with wave (M1, M2, M3)
- **Automated validation**: Code ensures NO M3 variables (except outcome) are used as predictors
- **One-to-one joins**: Strict enforcement that each participant appears exactly once
- **Outcome isolation**: M3 cognition stored separately and merged only at final step

### SHAP Explainability
- Global feature importance rankings
- Individual prediction decompositions
- SHAP summary plots
- Top features with human-readable labels

### What-If Simulation Engine
Test hypothetical scenarios like "What if this person reduced alcohol by 30%?"
- Predefined scenarios: reduce_alcohol, reduce_stress, improve_self_rated_health, combined_lifestyle
- Value clamping to valid ranges
- Immutable features (age, sex) unchanged
- All changes logged and reported

**Note**: This is PREDICTIVE simulation, NOT causal inference. The model learned associations, not interventions.

### Model Fairness
- Tested across gender and age groups
- No significant bias found (p > 0.05 for all demographic subgroups)
- Model performs equally well across demographic subgroups

## Technical Stack

- **Python 3.12**
- **scikit-learn 1.5.2** (ML framework)
- **SHAP 0.46.0** (explainability)
- **Pandas, NumPy** (data processing)
- **Matplotlib, Seaborn, Altair** (visualization)
- **Streamlit** (dashboard)
- **pyreadstat** (SPSS .sav file reading)
- **PyYAML** (configuration)
- **joblib** (model persistence)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd windsurf-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
windsurf-project/
├── config.yaml                    # Central configuration file
├── demo_app.py                    # Streamlit dashboard (main UI)
├── requirements.txt               # Python dependencies
│
├── predictive_twin/               # Core modeling package
│   ├── config.py                  # Configuration loader
│   ├── manifest.py                # Feature selection with leakage checks
│   ├── dataset_builder.py         # Leakage-safe dataset construction
│   ├── preprocess.py              # Preprocessing pipeline
│   ├── modeling.py                # Random Forest model builder
│   ├── evaluate.py                # Performance metrics (R², MAE, RMSE)
│   ├── explain_shap.py            # SHAP explainability
│   ├── simulate.py                # What-if simulation engine
│   ├── interpretability.py        # Feature importance extraction
│   ├── persist.py                 # Model save/load utilities
│   └── leakage.py                 # Leakage validation
│
├── midus_pipeline/                # Data processing package
│   └── spss_io.py                 # SPSS file loader and metadata extractor
│
├── scripts/                       # Executable scripts
│   ├── train_predictive_twin.py           # Main training script
│   ├── generate_descriptive_tables.py     # Descriptive statistics
│   ├── statistical_tests.py               # Hypothesis testing
│   ├── visualize_descriptive_stats.py     # Visualization generation
│   ├── process_sav_folder.py              # Batch SPSS processing
│   ├── generate_column_catalogs.py        # Column metadata extraction
│   ├── generate_btact_artifacts.py        # BTACT harmonization
│   └── build_phase1_manifest_template.py  # Manifest template builder
│
├── artifacts/                     # Generated artifacts
│   ├── phase1_feature_manifest__BASELINE.csv  # Curated feature manifest
│   ├── descriptive_tables/        # Descriptive statistics CSVs
│   ├── descriptive_visualizations/ # Box plots, bar charts
│   └── statistical_tests/         # T-tests, ANOVA, post-hoc results
│
├── models/                        # Trained models
│   ├── primary/                   # Primary model artifacts
│   │   ├── model.joblib           # Trained Random Forest
│   │   ├── metrics.json           # R², MAE, RMSE
│   │   ├── predictions.csv        # Test set predictions
│   │   ├── join_report.json       # Dataset merge report
│   │   └── shap/                  # SHAP artifacts
│   └── ablation/                  # Ablation model artifacts
│
└── data/                          # MIDUS raw data (not in repo)
    ├── MIDUS 1/
    ├── MIDUS 2/
    └── MIDUS 3/
```

## Usage

### Training Pipeline

```bash
python scripts/train_predictive_twin.py
```

This will:
- Load config.yaml
- Read feature manifest
- Build leakage-safe dataset
- Train both models (primary + ablation)
- Compute metrics
- Generate SHAP explanations
- Save all artifacts to models/ directory

### Descriptive Statistics

```bash
# Generate stratified tables
python scripts/generate_descriptive_tables.py

# Create visualizations
python scripts/visualize_descriptive_stats.py

# Perform statistical tests
python scripts/statistical_tests.py
```

### Interactive Dashboard

```bash
streamlit run demo_app.py
```

Launches web dashboard at http://localhost:8501 with 4 main sections:
1. **SPSS Data Loader**: Browse and load SPSS files, preview data dictionaries
2. **Artifact Browser**: View column catalogs, BTACT harmonization artifacts
3. **Descriptive Statistics**: Stratified tables, visualizations, statistical tests
4. **Predictive Twin Simulator**: Model variant selection, cohort building, scenario simulation

## Configuration

Edit `config.yaml` to customize:

```yaml
# Data paths
data_dir: data
artifacts_dir: artifacts
models_dir: models
manifest_path: artifacts/phase1_feature_manifest__BASELINE.csv

# Join configuration
join_id: M2ID
family_id: M2FAMNUM

# Outcome configuration
outcome:
  wave: M3
  dataset: data/MIDUS 3/M3_P3_BTACT_N3291_20210922.sav
  code: C3TCOMP

# Prior cognition configuration
prior_cognition:
  wave: M2
  dataset: data/MIDUS 2/M2_P3_BTACT_N4512_20211123.sav
  code: B3TCOMPZ1

# Train/test split
split:
  test_size: 0.2
  random_state: 42

# Model hyperparameters
model:
  random_forest:
    n_estimators: 500
    random_state: 42
    n_jobs: -1
    max_depth: null
    min_samples_leaf: 2

# SHAP configuration
shap:
  enabled: true
  max_background: 500
  max_explain: 2000
```

## Feature Manifest

The manifest (`phase1_feature_manifest__BASELINE.csv`) is the single source of truth for feature selection:

**Columns**:
- `wave`: M1, M2, or M3
- `dataset_family`: Which MIDUS dataset
- `code`: Variable code (e.g., B1PAGE_M2)
- `label`: Human-readable description
- `include_decision`: "include" or "exclude"
- `recommended_role`: "predictor", "outcome", "prior_cognition", "id"
- `tier`: Priority level (1=highest)
- `leakage_check_status`: "SAFE" or "POTENTIAL_LEAK"

## Ethical Considerations

1. **Not for Clinical Use**: Research tool, not diagnostic instrument
2. **Predictive, Not Causal**: Associations do not imply causation
3. **Fairness Tested**: Model performs equally across demographic groups
4. **Transparency**: All methods documented and explainable
5. **Privacy**: No individual-level data displayed without aggregation

## Data Requirements

This project requires MIDUS dataset files in SPSS (.sav) format:
- MIDUS 1: M1_P1_SURVEY_N7108_20190116.sav
- MIDUS 2: M2_P1_SURVEY_N4963_20200720.sav, M2_P3_BTACT_N4512_20211123.sav
- MIDUS 3: M3_P3_BTACT_N3291_20210922.sav

Data files should be placed in the `data/` directory with appropriate subdirectories.

## License

This project is for research purposes. Please cite the MIDUS study if using this code or data.

## Citation

If you use this code in your research, please cite:

```
MIDUS Predictive Digital Twin Pipeline
https://github.com/[your-username]/windsurf-project
```

## Contact

For questions about this project, please open an issue on GitHub or contact [your-email].
