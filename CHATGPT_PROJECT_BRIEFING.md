# ChatGPT Project Briefing Prompt

Copy and paste this entire prompt to ChatGPT to brief it on the MIDUS Digital Twin project:

---

## PROJECT BRIEFING: MIDUS Predictive Digital Twin System

I'm working on a research-grade predictive digital twin system for cognitive aging research. Here's everything you need to know about the project:

### PROJECT OVERVIEW

**Name**: MIDUS Predictive Digital Twin Pipeline

**Purpose**: Predict future cognitive outcomes (M3 BTACT scores) using historical health, lifestyle, and demographic data from MIDUS longitudinal study

**Dataset**: Midlife in the United States (MIDUS) - a longitudinal study with three waves:
- Wave 1 (M1): 1995-1996 baseline
- Wave 2 (M2): 2004-2006 follow-up
- Wave 3 (M3): 2013-2014 follow-up

**Sample Size**: N = 2,735 participants with complete data across all three waves

**Research Question**: Can we predict cognitive function at M3 using only data from M1 and M2?

---

### MODEL ARCHITECTURE

**Algorithm**: Random Forest Regression (scikit-learn)

**Why Random Forest?**
- Handles non-linear relationships
- Robust to outliers and missing data
- Provides feature importance rankings
- No strong parametric assumptions
- Works with mixed data types

**Hyperparameters**:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 10
- min_samples_leaf: 4
- random_state: 42

**Preprocessing Pipeline**:
- Numeric features: Median imputation + StandardScaler
- Categorical features: Most frequent imputation + One-hot encoding
- Automatic type detection from manifest

---

### TWO-MODEL DESIGN

**1. Primary Model (With Prior Cognition)**
- Includes M2 BTACT composite z-score (B3TCOMPZ1) as predictor
- Performance: R² = 0.575, MAE = 0.344, RMSE = 0.434
- Use case: Best predictive accuracy when baseline cognition available

**2. Ablation Model (Without Prior Cognition)**
- Excludes M2 prior cognition variable
- Performance: R² = 0.277, MAE = 0.449, RMSE = 0.566
- Use case: Tests importance of baseline cognition

**Key Finding**: Prior cognition alone accounts for ~30% of predictive power, demonstrating cognitive trajectory stability over time.

---

### LEAKAGE PREVENTION (CRITICAL FEATURE)

**The Problem**: Temporal leakage occurs when future information "leaks" into training data, artificially inflating performance.

**Our Solution**:
1. **Manifest-Driven Feature Selection**: Every feature explicitly labeled with wave (M1, M2, M3)
2. **Automated Validation**: Code checks that NO M3 variables (except outcome) are used as predictors
3. **One-to-One Joins**: Strict enforcement that each participant appears exactly once
4. **Outcome Isolation**: M3 cognition stored separately and merged only at final step

**Validation**:
- All predictors verified to be from M1 or M2 only
- Join reports confirm one-to-one matching across waves
- Leakage check status: "SAFE" for all included features

---

### MODEL PERFORMANCE & FAIRNESS

**Primary Model Metrics**:
- R² = 0.575 (57.5% of variance explained - good for behavioral data)
- MAE = 0.344 (average error of 0.34 standard deviations)
- RMSE = 0.434

**Model Fairness** (tested across demographics):
- Gender: No significant difference in MAE (p = 0.63)
- Age Groups: No significant difference in MAE (p = 0.18)
- Conclusion: Model performs equally well across demographic subgroups

---

### INTERPRETABILITY (SHAP)

**What is SHAP?**
SHapley Additive exPlanations - a game-theoretic approach to explain model predictions.

**How It Works**:
1. For each prediction, SHAP calculates how much each feature contributed
2. Positive SHAP = feature pushed prediction higher
3. Negative SHAP = feature pushed prediction lower
4. Sum of all SHAP values = deviation from average prediction

**Outputs**:
- Global feature importance rankings
- Individual prediction decompositions
- SHAP summary plots
- Top features with labels from manifest

---

### WHAT-IF SIMULATION ENGINE

**Purpose**: Test hypothetical scenarios like "What if this person reduced alcohol by 30%?"

**How It Works**:
1. Select a person from the cohort
2. Get baseline prediction using their actual features
3. Apply scenario (e.g., reduce alcohol, improve health)
4. Get new prediction with modified features
5. Calculate delta (change in predicted cognition)

**Predefined Scenarios**:
- `reduce_alcohol`: 30% reduction in alcohol consumption
- `reduce_stress`: 25% reduction in worry/stress frequency
- `improve_self_rated_health`: 1-step improvement in health rating
- `combined_lifestyle`: All of the above simultaneously

**Safety Features**:
- Value clamping to valid ranges
- Immutable features (age, sex) unchanged
- All changes logged and reported

**CRITICAL LIMITATION**: This is PREDICTIVE simulation, NOT causal inference. The model learned associations, not interventions. Real-world changes may have different effects.

---

### DESCRIPTIVE STATISTICS & STATISTICAL TESTS

**Generated Tables**:
1. Table 2: Statistics by Gender (Mean ± SD for all variables)
2. Table 3: Statistics by Age Group (5 groups: <40, 40-49, 50-59, 60-69, 70+)
3. Table 5: Key Variables by Gender (publication-ready)
4. Table 6: Model Performance by Demographics (MAE, RMSE by groups)

**Statistical Tests Performed**:
1. Independent t-tests by gender
2. One-way ANOVA by age group
3. Post-hoc pairwise comparisons (Tukey HSD)
4. Normality tests (Shapiro-Wilk)
5. Model fairness tests (ANOVA on MAE)

**Visualizations Generated**:
1. Box plots: Cognition by gender and age group
2. Bar charts: Model performance by demographics
3. SHAP summary plots

**Key Findings**:
- Age Effect: Strong negative correlation (r = -0.45, p < 0.001), F = 177.5, η² = 0.206
- Gender Effect: No significant difference (p = 0.257)
- Education Effect: Positive association with cognition

---

### TECHNICAL STACK

**Language**: Python 3.12

**Core Libraries**:
- scikit-learn 1.5.2 (ML framework)
- SHAP 0.46.0 (explainability)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn, Altair (visualization)
- Streamlit (dashboard)
- pyreadstat (SPSS .sav file reading)
- PyYAML (configuration)
- joblib (model persistence)

**Data Format**: SPSS (.sav) files from MIDUS study

---

### PROJECT STRUCTURE

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
│   │       ├── shap_values.parquet
│   │       ├── shap_summary.png
│   │       └── top_features.csv
│   └── ablation/                  # Ablation model artifacts
│       └── (same structure as primary)
│
└── data/                          # MIDUS raw data (not in repo)
    ├── MIDUS 1/
    ├── MIDUS 2/
    └── MIDUS 3/
```

---

### KEY WORKFLOWS

**1. Training Pipeline**:
```bash
python scripts/train_predictive_twin.py
```
- Loads config.yaml
- Reads feature manifest
- Builds leakage-safe dataset
- Trains both models (primary + ablation)
- Computes metrics
- Generates SHAP explanations
- Saves all artifacts

**2. Descriptive Statistics**:
```bash
python scripts/generate_descriptive_tables.py
python scripts/visualize_descriptive_stats.py
python scripts/statistical_tests.py
```
- Generates stratified tables (gender, age, education)
- Creates visualizations (box plots, bar charts)
- Performs statistical tests (t-tests, ANOVA, post-hoc)

**3. Interactive Demo**:
```bash
streamlit run demo_app.py
```
- Launches web dashboard at http://localhost:8501
- 4 main sections:
  1. SPSS data loader
  2. Artifact browser
  3. Descriptive statistics
  4. Predictive twin simulator

---

### STREAMLIT DASHBOARD FEATURES

**Section 1: Load .sav file + preview metadata**
- Browse and load SPSS files
- Preview data dictionaries
- View renamed columns (CODE : Label format)

**Section 2: Browse generated artifacts**
- Column catalogs
- BTACT harmonization artifacts
- Feature manifest template

**Section 3: Descriptive Statistics & Demographics**
- **Tables Tab**: Stratified descriptive statistics
- **Visualizations Tab**: Box plots, bar charts
- **Statistical Tests Tab**: T-tests, ANOVA, post-hoc, fairness tests

**Section 4: Predictive Twin (model + what-if simulation)**
- Model variant selection (primary vs ablation)
- Cohort building
- Subject selection
- Scenario simulation
- Delta visualization
- Changed features display
- SHAP global interpretability

**Information System**:
- Comprehensive "About This Project" expander at top (300+ lines of documentation)
- Section-level expanders explaining methodology
- Component-level popovers with quick help
- All text-based (no emojis)

---

### CONFIGURATION (config.yaml)

```yaml
# Data paths
m1_survey_dataset:
  name: "M1_SURVEY"
  path: "data/MIDUS 1/M1_P1_SURVEY_N7108_20200720.sav"

m2_survey_dataset:
  name: "M2_SURVEY"
  path: "data/MIDUS 2/M2_P1_SURVEY_N4963_20200720.sav"

prior_dataset:
  name: "M2_BTACT"
  path: "data/MIDUS 2/M2_P4_BTACT_N4512_20200720.sav"

outcome_dataset:
  name: "M3_BTACT"
  path: "data/MIDUS 3/M3_P4_BTACT_N3294_20200720.sav"

# Feature manifest
manifest_path: "artifacts/phase1_feature_manifest__BASELINE.csv"

# Model outputs
models_dir: "models"

# Join configuration
join_id: "M2ID"
outcome_code: "C3TCOMP"
prior_code: "B3TCOMPZ1"

# Train/test split
test_size: 0.2
random_state: 42

# Model hyperparameters
rf_params:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 10
  min_samples_leaf: 4
  random_state: 42

# SHAP configuration
shap_params:
  max_samples: 100
  top_n_features: 20
```

---

### FEATURE MANIFEST STRUCTURE

The manifest (`phase1_feature_manifest__BASELINE.csv`) is the **single source of truth** for feature selection:

**Columns**:
- `wave`: M1, M2, or M3
- `dataset_family`: Which MIDUS dataset
- `code`: Variable code (e.g., B1PAGE_M2)
- `label`: Human-readable description
- `include_decision`: "include" or "exclude"
- `recommended_role`: "predictor", "outcome", "prior_cognition", "id"
- `tier`: Priority level (1=highest)
- `leakage_check_status`: "SAFE" or "POTENTIAL_LEAK"

**Example Rows**:
```csv
wave,dataset_family,code,label,include_decision,recommended_role,tier,leakage_check_status
M3,M3_BTACT,C3TCOMP,M3 BTACT Composite Score,include,outcome,1,SAFE
M2,M2_BTACT,B3TCOMPZ1,M2 BTACT Composite Z-score,include,prior_cognition,1,SAFE
M1,M1_SURVEY,A1PAGE_M2,Age at M2,include,predictor,1,SAFE
M2,M2_SURVEY,B1PRSEX,Sex (1=Male 2=Female),include,predictor,1,SAFE
```

---

### RECENT DEVELOPMENT WORK

**What We've Built**:

1. **Core Modeling System** (completed):
   - Leakage-safe dataset builder
   - Two-model training pipeline
   - SHAP explainability
   - What-if simulation engine

2. **Descriptive Statistics** (completed):
   - Stratified tables by demographics
   - Statistical tests (t-tests, ANOVA, post-hoc)
   - Visualizations (box plots, bar charts)
   - Model fairness testing

3. **Streamlit Dashboard** (completed):
   - 4-section interactive demo
   - Comprehensive documentation system
   - Info buttons on all components
   - Clean, consistent UI (no emojis)

**Recent UI Improvements**:
- Added comprehensive "About This Project" expander at top
- Added info expanders for each section
- Added info popovers for all tables, visualizations, and tests
- Simplified CSS for consistent spacing
- Removed all emojis for professional appearance
- Changed popover buttons from emoji to "Info" text

---

### ETHICAL CONSIDERATIONS

1. **Not for Clinical Use**: Research tool, not diagnostic instrument
2. **Predictive, Not Causal**: Associations do not imply causation
3. **Fairness Tested**: Model performs equally across demographic groups
4. **Transparency**: All methods documented and explainable
5. **Privacy**: No individual-level data displayed without aggregation

---

### CURRENT STATUS

**Completed**:
- ✅ Core modeling pipeline
- ✅ Leakage prevention system
- ✅ Two-model training (primary + ablation)
- ✅ SHAP explainability
- ✅ What-if simulation engine
- ✅ Descriptive statistics generation
- ✅ Statistical testing suite
- ✅ Visualization generation
- ✅ Streamlit dashboard
- ✅ Comprehensive documentation
- ✅ Info button system
- ✅ Clean UI (emoji-free)

**Performance**:
- ✅ Primary model: R² = 0.575 (good for behavioral data)
- ✅ Model fairness: No bias across demographics
- ✅ Leakage validation: All features verified safe
- ✅ One-to-one joins: All participants matched correctly

**Ready For**:
- Research presentations
- Manuscript preparation
- Stakeholder demos
- Further model refinement
- Additional feature engineering

---

### HOW TO USE THIS INFORMATION

When asking me questions about this project, you can reference:
- Specific components (e.g., "the SHAP explainability module")
- Specific files (e.g., "the simulate.py file")
- Specific features (e.g., "the leakage prevention system")
- Specific results (e.g., "the model fairness tests")

I understand the full context of:
- Why we built it this way
- What each component does
- How the pieces fit together
- What the results mean
- What the limitations are

---

**END OF BRIEFING**

You now have complete context on the MIDUS Predictive Digital Twin project. Feel free to ask me anything about the implementation, methodology, results, or next steps!
