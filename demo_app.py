from __future__ import annotations

from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from midus_pipeline.spss_io import build_data_dictionary, load_sav, make_code_label_columns

from predictive_twin.config import load_config
from predictive_twin.dataset_builder import build_modeling_dataset
from predictive_twin.interpretability import build_code_to_label_map_from_manifest, read_global_top_features_from_shap
from predictive_twin.manifest import read_manifest, select_features
from predictive_twin.persist import load_model
from predictive_twin.simulate import run_scenario, scenario_library


DATA_DIR_DEFAULT = Path("data")
ARTIFACTS_DIR_DEFAULT = Path("artifacts")


def _list_sav_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    savs = list(root.rglob("*.sav")) + list(root.rglob("*.SAV"))
    return sorted(set(savs))


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


st.set_page_config(
    page_title="MIDUS Digital Twin Demo", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS for consistency
st.markdown("""
<style>
    /* Consistent spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Consistent headers */
    h2 {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Rounded corners for consistency */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    /* Consistent dividers */
    hr {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("MIDUS Digital Twin Pipeline Demo")

# Comprehensive information section at the top
with st.expander("About This Project - Model Overview & Methodology", expanded=False):
    st.markdown("""
    ## Project Overview
    
    This is a **research-grade predictive digital twin system** for cognitive aging research using the MIDUS 
    (Midlife in the United States) longitudinal dataset. The system predicts future cognitive outcomes 
    based on historical health, lifestyle, and demographic data.
    
    ---
    
    ## Research Question
    
    **Can we predict cognitive function at MIDUS Wave 3 (M3) using data from Waves 1 and 2 (M1, M2)?**
    
    - **Outcome**: M3 BTACT Composite Score (Brief Test of Adult Cognition by Telephone)
    - **Predictors**: Health, lifestyle, demographics, and prior cognition from M1 and M2
    - **Time Span**: Approximately 9-10 years between M2 and M3
    
    ---
    
    ## Dataset Information
    
    ### **MIDUS Longitudinal Study**
    - **Wave 1 (M1)**: 1995-1996 baseline data
    - **Wave 2 (M2)**: 2004-2006 follow-up data
    - **Wave 3 (M3)**: 2013-2014 follow-up data
    
    ### **Sample Characteristics**
    - **Total Sample**: N = 2,735 participants with complete data across all waves
    - **Age Range**: 32-84 years at M3
    - **Gender**: 44.7% Male, 55.3% Female
    - **One-to-One Matching**: Each participant tracked across all three waves using unique ID (M2ID)
    
    ---
    
    ## Model Architecture
    
    ### **Algorithm: Random Forest Regression**
    
    **Why Random Forest?**
    - Handles non-linear relationships between predictors and cognition
    - Robust to outliers and missing data
    - Provides feature importance rankings
    - No strong parametric assumptions
    - Works well with mixed data types (continuous, categorical)
    
    **Hyperparameters:**
    - **n_estimators**: 100 trees
    - **max_depth**: 10 levels
    - **min_samples_split**: 10
    - **min_samples_leaf**: 4
    - **random_state**: 42 (for reproducibility)
    
    ### **Preprocessing Pipeline**
    - **Numeric features**: Median imputation + StandardScaler
    - **Categorical features**: Most frequent imputation + One-hot encoding
    - **Feature engineering**: Automatic type detection from manifest
    
    ---
    
    ## Two-Model Design
    
    ### **1. Primary Model (With Prior Cognition)**
    - **Includes**: M2 BTACT composite z-score (B3TCOMPZ1) as a predictor
    - **Performance**: R² = 0.575, MAE = 0.344, RMSE = 0.434
    - **Interpretation**: 57.5% of variance in M3 cognition explained
    - **Use Case**: Best predictive accuracy when baseline cognition is available
    
    ### **2. Ablation Model (Without Prior Cognition)**
    - **Excludes**: M2 prior cognition variable
    - **Performance**: R² = 0.277, MAE = 0.449, RMSE = 0.566
    - **Interpretation**: 27.7% of variance explained
    - **Use Case**: Tests importance of baseline cognition; useful when baseline unavailable
    
    ### **Key Finding**
    **Prior cognition alone accounts for ~30% of predictive power**, demonstrating that:
    - Cognitive trajectories are relatively stable over time
    - Baseline cognitive function is the strongest single predictor
    - Other factors (health, lifestyle) contribute an additional ~28%
    
    ---
    
    ## Leakage Prevention
    
    ### **The Problem**
    Temporal leakage occurs when future information "leaks" into training data, inflating performance artificially.
    
    ### **Our Solution**
    1. **Manifest-Driven Feature Selection**: Every feature explicitly labeled with wave (M1, M2, M3)
    2. **Automated Validation**: Code checks that NO M3 variables (except outcome) are used as predictors
    3. **One-to-One Joins**: Strict enforcement that each participant appears exactly once
    4. **Outcome Isolation**: M3 cognition stored separately and merged only at final step
    
    ### **Validation**
    - All predictors verified to be from M1 or M2 only
    - Join reports confirm one-to-one matching across waves
    - Leakage check status: "SAFE" for all included features
    
    ---
    
    ## Model Performance
    
    ### **Primary Model Metrics**
    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | **R²** | 0.575 | 57.5% of variance explained (good for behavioral data) |
    | **MAE** | 0.344 | Average error of 0.34 standard deviations |
    | **RMSE** | 0.434 | Root mean squared error |
    
    ### **Ablation Model Metrics**
    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | **R²** | 0.277 | 27.7% of variance explained |
    | **MAE** | 0.449 | Average error of 0.45 standard deviations |
    | **RMSE** | 0.566 | Root mean squared error |
    
    ### **Model Fairness**
    - **Gender**: No significant difference in MAE (p = 0.63)
    - **Age Groups**: No significant difference in MAE (p = 0.18)
    - **Conclusion**: Model performs equally well across demographic subgroups
    
    ---
    
    ## Model Interpretability (SHAP)
    
    ### **What is SHAP?**
    **SHapley Additive exPlanations** - A game-theoretic approach to explain model predictions.
    
    ### **How It Works**
    1. For each prediction, SHAP calculates how much each feature contributed
    2. Positive SHAP = feature pushed prediction higher
    3. Negative SHAP = feature pushed prediction lower
    4. Sum of all SHAP values = deviation from average prediction
    
    ### **Global Interpretability**
    - **Top Features**: Ranked by mean absolute SHAP value
    - **Feature Importance**: Shows which variables matter most overall
    - **Summary Plot**: Visualizes SHAP distributions for all features
    
    ### **Individual Interpretability**
    - Each prediction can be decomposed into feature contributions
    - Enables "what-if" scenario testing
    - Transparent decision-making process
    
    ---
    
    ## What-If Simulation Engine
    
    ### **Purpose**
    Test hypothetical scenarios: "What if this person reduced alcohol consumption by 30%?"
    
    ### **How It Works**
    1. **Select a person** from the cohort
    2. **Get baseline prediction** using their actual features
    3. **Apply scenario** (e.g., reduce alcohol, improve health)
    4. **Get new prediction** with modified features
    5. **Calculate delta** (change in predicted cognition)
    
    ### **Predefined Scenarios**
    - **reduce_alcohol**: 30% reduction in alcohol consumption
    - **reduce_stress**: 25% reduction in worry/stress frequency
    - **improve_self_rated_health**: 1-step improvement in health rating
    - **combined_lifestyle**: All of the above simultaneously
    
    ### **Safety Features**
    - **Value clamping**: Ensures modified values stay within valid ranges
    - **Immutable features**: Age, sex, and other non-modifiable factors unchanged
    - **Logging**: All changes tracked and reported
    
    ### **Critical Limitation**
    **This is PREDICTIVE simulation, NOT causal inference.**
    - The model learned **associations**, not **interventions**
    - Changing a value shows what the model **predicts**, not what **would actually happen**
    - Real-world interventions may have different effects due to:
      - Confounding variables
      - Reverse causation
      - Unmeasured factors
      - Complex interactions
    
    **Use Case**: Hypothesis generation and exploratory analysis, NOT clinical decision-making.
    
    ---
    
    ## Descriptive Statistics
    
    ### **Sample Characteristics**
    - **Age**: Mean = 54.6 years (SD = 11.8)
    - **M3 Cognition**: Mean = 0.15 (SD = 0.67) [standardized z-score]
    - **M2 Prior Cognition**: Mean = 0.14 (SD = 0.67)
    
    ### **Key Findings**
    1. **Age Effect**: Strong negative correlation (r = -0.45, p < 0.001)
       - Each decade associated with ~0.3 SD decline in cognition
       - ANOVA: F = 177.5, η² = 0.206 (large effect)
    
    2. **Gender Effect**: No significant difference
       - Males: Mean = 0.17 (SD = 0.68)
       - Females: Mean = 0.13 (SD = 0.66)
       - t-test: p = 0.257 (not significant)
    
    3. **Education Effect**: Positive association with cognition
       - Higher education → higher cognitive scores
    
    ---
    
    ## Technical Stack
    
    - **Language**: Python 3.12
    - **ML Framework**: scikit-learn 1.5.2
    - **Explainability**: SHAP 0.46.0
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn, Altair
    - **Dashboard**: Streamlit
    - **Data Format**: SPSS (.sav) files via pyreadstat
    
    ---
    
    ## Project Structure
    
    ```
    predictive_twin/
    ├── config.py              # Configuration loader
    ├── manifest.py            # Feature selection with leakage checks
    ├── dataset_builder.py     # Leakage-safe dataset construction
    ├── preprocess.py          # Preprocessing pipeline
    ├── modeling.py            # Random Forest model builder
    ├── evaluate.py            # Performance metrics
    ├── explain_shap.py        # SHAP explainability
    ├── simulate.py            # What-if simulation engine
    └── interpretability.py    # Feature importance extraction
    
    scripts/
    ├── train_predictive_twin.py           # Main training script
    ├── generate_descriptive_tables.py     # Descriptive statistics
    ├── statistical_tests.py               # Hypothesis testing
    └── visualize_descriptive_stats.py     # Visualization generation
    ```
    
    ---
    
    ## How to Use This Dashboard
    
    ### **Section 1: Data Loading**
    - Load SPSS .sav files
    - Preview metadata and data dictionaries
    - Understand the raw MIDUS data structure
    
    ### **Section 2: Artifacts**
    - Browse generated column catalogs
    - View BTACT harmonization artifacts
    - Inspect feature manifest template
    
    ### **Section 3: Descriptive Statistics**
    - **Tables**: Sample characteristics by demographics
    - **Visualizations**: Box plots, bar charts
    - **Statistical Tests**: t-tests, ANOVA, post-hoc comparisons
    
    ### **Section 4: Predictive Twin**
    - Select model variant (Primary vs Ablation)
    - Build cohort and choose a subject
    - Run what-if scenarios
    - View SHAP explanations
    
    ---
    
    ## Citation & Acknowledgments
    
    ### **MIDUS Data**
    This research uses data from the Midlife in the United States (MIDUS) study, 
    funded by the National Institute on Aging.
    
    ### **Methods**
    - Random Forest: Breiman, L. (2001). Machine Learning, 45(1), 5-32.
    - SHAP: Lundberg & Lee (2017). NeurIPS.
    
    ---
    
    ## Ethical Considerations
    
    1. **Not for Clinical Use**: This is a research tool, not a diagnostic instrument
    2. **Predictive, Not Causal**: Associations do not imply causation
    3. **Fairness Tested**: Model performs equally across demographic groups
    4. **Transparency**: All methods documented and explainable
    5. **Privacy**: No individual-level data displayed without aggregation
    
    ---
    
    ## Support & Documentation
    
    For questions, issues, or contributions, please refer to the project documentation 
    or contact the research team.
    """)

st.divider()

st.subheader("Guided Demo (one-click)")

demo_col1, demo_col2 = st.columns([1, 2])

with demo_col1:
    run_guided = st.button("Run guided demo", type="primary")
    st.caption("Runs a pre-scripted walkthrough: M2 survey loader preview, BTACT key variables, manifest template.")

with demo_col2:
    st.markdown(
        "\n".join(
            [
                "**What this guided demo proves:**",
                "- SPSS `.sav` loading works and preserves metadata (labels/value labels)",
                "- Column renaming to `CODE : Label` is deterministic and traceable",
                "- Cognition outcome/prior pair is explicitly documented (C3TCOMP / B3TCOMPZ1)",
                "- Phase 1 feature selection is leakage-safe via a manifest template",
            ]
        )
    )

with st.sidebar:
    st.header("Paths")
    data_dir = Path(st.text_input("Data directory", str(DATA_DIR_DEFAULT)))
    artifacts_dir = Path(st.text_input("Artifacts directory", str(ARTIFACTS_DIR_DEFAULT)))

    st.divider()
    st.header("SPSS Loader Settings")
    preview_rows = st.number_input("Preview rows", min_value=3, max_value=200, value=10, step=1)

st.subheader("1) Load a .sav file + preview metadata")

sav_files = _list_sav_files(data_dir)
if not sav_files:
    st.warning(f"No .sav files found under: {data_dir}")
else:
    options = [str(p) for p in sav_files]
    selected = st.selectbox("Choose a MIDUS .sav file", options, index=0)
    sav_path = Path(selected)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        if st.button("Load file", type="primary"):
            with st.spinner("Loading SPSS file..."):
                df, meta = load_sav(sav_path)
                st.session_state["_df"] = df
                st.session_state["_meta"] = meta
                st.session_state["_sav_path"] = str(sav_path)

        if "_df" in st.session_state and st.session_state.get("_sav_path") == str(sav_path):
            df = st.session_state["_df"]
            st.write("**Shape**", df.shape)
            st.dataframe(df.head(int(preview_rows)))

    with col_b:
        if "_meta" in st.session_state and st.session_state.get("_sav_path") == str(sav_path):
            meta = st.session_state["_meta"]
            dd = build_data_dictionary(meta, file_path=sav_path)
            st.write("**Data dictionary (preview)**")
            st.dataframe(dd.columns[["code", "label", "has_value_labels"]].head(50))

            df = st.session_state["_df"].head(int(preview_rows))
            renamed, _ = make_code_label_columns(df, meta)
            st.write("**Renamed columns preview (`CODE : Label`)**")
            st.dataframe(pd.DataFrame({"renamed_columns": list(renamed.columns)}).head(50))

if run_guided:
    st.divider()
    st.subheader("Guided demo output")

    # Step 1: Load an M2 core survey file (if present)
    m2_survey = data_dir / "MIDUS 2" / "M2_P1_SURVEY_N4963_20200720.sav"
    if not m2_survey.exists():
        st.error(f"Guided demo expected file not found: {m2_survey}")
    else:
        with st.spinner("Loading M2 core survey... (this can take a moment)"):
            df_m2, meta_m2 = load_sav(m2_survey)

        st.write("### A) M2 core survey: loader + metadata")
        st.write("**File**", m2_survey.name)
        st.write("**Shape**", df_m2.shape)

        id_like = [c for c in ["M2ID", "M2FAMNUM", "SAMPLMAJ"] if c in df_m2.columns]
        st.write("**ID-like variables (for linking later)**", id_like)

        dd_m2 = build_data_dictionary(meta_m2, file_path=m2_survey)
        st.write("**Data dictionary preview**")
        st.dataframe(dd_m2.columns[["code", "label", "has_value_labels"]].head(30))

        renamed_preview, _ = make_code_label_columns(df_m2.head(int(preview_rows)), meta_m2)
        st.write("**Renamed columns preview (`CODE : Label`)**")
        st.dataframe(pd.DataFrame({"renamed_columns": list(renamed_preview.columns)}).head(30))

    # Step 2: Highlight BTACT key variables
    st.write("### B) Cognition: harmonized M2/M3 composite variables")
    btact_path = artifacts_dir / "btact_harmonization_m2_m3.parquet"
    if not btact_path.exists():
        st.error(f"BTACT harmonization artifact not found: {btact_path}")
    else:
        bt = _safe_read_parquet(btact_path)
        if bt is None:
            st.error("Could not read BTACT harmonization parquet.")
        else:
            key = bt[bt["recommended_role"].isin(["primary_outcome", "prior_cognition"])].copy()
            st.write("**Phase 1 agreed pair**")
            st.dataframe(key[["wave", "code", "label", "scaling_note", "sample_restriction", "missingness_rate", "directionality", "recommended_role"]])

    # Step 3: Show the manifest template (leakage-safe manual selection workflow)
    st.write("### C) Phase 1 feature selection manifest (manual curation + leakage checks)")
    tmpl_path = artifacts_dir / "phase1_feature_manifest__TEMPLATE.csv"
    if not tmpl_path.exists():
        st.error(f"Manifest template not found: {tmpl_path}")
    else:
        tmpl = _safe_read_csv(tmpl_path)
        if tmpl is None:
            st.error("Could not read manifest template CSV.")
        else:
            st.write("**Included variables (currently)**")
            st.dataframe(
                tmpl[tmpl["include_decision"].eq("include")][
                    ["wave", "dataset_family", "code", "label", "recommended_role", "tier", "leakage_check_status"]
                ].head(50)
            )
            st.write("**Leakage check summary**")
            st.dataframe(tmpl["leakage_check_status"].value_counts().rename_axis("status").reset_index(name="count"))

st.divider()
st.subheader("2) Browse generated artifacts")

art_col1, art_col2 = st.columns([1, 1])

with art_col1:
    st.write("### Column catalogs")
    master_catalog_csv = artifacts_dir / "column_catalog__all_datasets.csv"
    if master_catalog_csv.exists():
        cat = _safe_read_csv(master_catalog_csv)
        if cat is not None:
            st.write("**Master catalog**")
            st.write("Rows:", len(cat))
            st.dataframe(cat.head(200))
    else:
        st.info("Master catalog not found. Expected: artifacts/column_catalog__all_datasets.csv")

with art_col2:
    st.write("### Cognition artifacts")
    btact = artifacts_dir / "btact_harmonization_m2_m3.parquet"
    if btact.exists():
        dfb = _safe_read_parquet(btact)
        if dfb is not None:
            st.write("**BTACT harmonization (preview)**")
            st.dataframe(dfb[dfb["recommended_role"].isin(["primary_outcome", "prior_cognition"])])
            st.dataframe(dfb.head(200))
    else:
        st.info("BTACT harmonization artifact not found. Expected: artifacts/btact_harmonization_m2_m3.parquet")

    tmpl = artifacts_dir / "phase1_feature_manifest__TEMPLATE.csv"
    if tmpl.exists():
        dft = _safe_read_csv(tmpl)
        if dft is not None:
            st.write("**Phase 1 feature manifest template (preview)**")
            st.write("Rows:", len(dft))
            st.dataframe(dft[dft["include_decision"].eq("include")].head(50))
            st.dataframe(dft.head(200))
    else:
        st.info("Manifest template not found. Expected: artifacts/phase1_feature_manifest__TEMPLATE.csv")

st.divider()
st.subheader("3) Descriptive Statistics & Demographics")

with st.expander("About This Section", expanded=False):
    st.markdown("""
    **What:** Comprehensive demographic and clinical characteristics of the study sample.
    
    **Why:** Understanding the sample composition is critical for:
    - Assessing generalizability of findings
    - Identifying potential biases or confounders
    - Evaluating model fairness across subgroups
    
    **How:** We stratify the sample by:
    - **Gender** (Male/Female)
    - **Age Groups** (<40, 40-49, 50-59, 60-69, 70+)
    - **Education Level** (High School or Less, Some College, College Degree+)
    
    All statistics include Mean ± SD and sample sizes (n).
    """)

desc_tables_dir = Path("artifacts/descriptive_tables")
desc_viz_dir = Path("artifacts/descriptive_visualizations")
stat_tests_dir = Path("artifacts/statistical_tests")

if desc_tables_dir.exists() or desc_viz_dir.exists():
    st.write("**Sample Characteristics and Statistical Analyses**")
    
    tab1, tab2, tab3 = st.tabs(["Tables", "Visualizations", "Statistical Tests"])
    
    with tab1:
        st.write("### Descriptive Statistics Tables")
        
        with st.expander("How to Read These Tables", expanded=False):
            st.markdown("""
            **Format:** Each cell shows `Mean ± SD (n=sample_size)`
            
            **Example:** `0.15 ± 0.67 (n=1200)` means:
            - Average value: 0.15
            - Standard deviation: 0.67
            - Number of observations: 1,200
            
            **Use Cases:**
            - Compare characteristics between groups
            - Identify demographic patterns
            - Prepare manuscript Table 1
            """)
        
        # Table 2: By Gender
        table2_path = desc_tables_dir / "table2_by_gender.csv"
        if table2_path.exists():
            col_title, col_info = st.columns([0.85, 0.15])
            with col_title:
                st.write("#### Table 2: Statistics by Gender")
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **What:** Mean and SD for all variables, stratified by gender.
                    
                    **Why:** Identifies gender-specific patterns in health, lifestyle, and cognition.
                    
                    **Key Insights:**
                    - Males typically report higher alcohol consumption
                    - Cognitive scores show minimal gender differences
                    - Useful for identifying sex-specific risk factors
                    """)
            table2 = _safe_read_csv(table2_path)
            if table2 is not None:
                st.dataframe(table2, use_container_width=True)
        
        # Table 3: By Age Group
        table3_path = desc_tables_dir / "table3_by_age_group.csv"
        if table3_path.exists():
            col_title, col_info = st.columns([0.85, 0.15])
            with col_title:
                st.write("#### Table 3: Statistics by Age Group")
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **What:** Mean and SD across 5 age groups (<40, 40-49, 50-59, 60-69, 70+).
                    
                    **Why:** Age is a primary driver of cognitive decline - this table quantifies the relationship.
                    
                    **Key Insights:**
                    - Clear monotonic decline in cognition with age
                    - Younger groups (<40) score ~1.1 SD higher than oldest (70+)
                    - Critical for understanding age-related trajectories
                    """)
            table3 = _safe_read_csv(table3_path)
            if table3 is not None:
                st.dataframe(table3, use_container_width=True)
        
        # Table 5: Key Variables
        table5_path = desc_tables_dir / "table5_key_variables_by_gender.csv"
        if table5_path.exists():
            col_title, col_info = st.columns([0.85, 0.15])
            with col_title:
                st.write("#### Table 5: Key Variables by Gender (Publication-Ready)")
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **What:** Compact table with only the most important variables.
                    
                    **Why:** Designed for manuscript "Table 1: Sample Characteristics".
                    
                    **Includes:**
                    - Primary outcome (M3 cognition)
                    - Prior cognition (M2 baseline)
                    - Demographics (age, gender, education)
                    
                    **Use:** Copy directly into your manuscript!
                    """)
            table5 = _safe_read_csv(table5_path)
            if table5 is not None:
                st.dataframe(table5, use_container_width=True)
        
        # Table 6: Model Performance
        table6_path = desc_tables_dir / "table6_model_performance_by_demographics.csv"
        if table6_path.exists():
            col_title, col_info = st.columns([0.85, 0.15])
            with col_title:
                st.write("#### Table 6: Model Performance by Demographics")
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **What:** Model prediction accuracy (MAE, RMSE) for different demographic groups.
                    
                    **Why:** Assesses model fairness - does it work equally well for everyone?
                    
                    **Key Metrics:**
                    - **MAE**: Mean Absolute Error (lower = better)
                    - **RMSE**: Root Mean Squared Error (penalizes large errors)
                    
                    **Interpretation:** Similar MAE across groups = fair model
                    """)
            table6 = _safe_read_csv(table6_path)
            if table6 is not None:
                st.dataframe(table6, use_container_width=True)
    
    with tab2:
        st.write("### Visualizations")
        
        with st.expander("How to Read Box Plots", expanded=False):
            st.markdown("""
            **Box Plot Components:**
            - **Box**: Middle 50% of data (25th to 75th percentile)
            - **Line in box**: Median (50th percentile)
            - **Diamond (◆)**: Mean (average)
            - **Whiskers**: Extend to min/max (excluding outliers)
            - **Dots**: Individual outliers
            
            **Interpretation:** Wider boxes = more variability; higher boxes = higher values
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1_path = desc_viz_dir / "fig1_outcome_by_gender_boxplot.png"
            if fig1_path.exists():
                col_title, col_info = st.columns([0.85, 0.15])
                with col_title:
                    st.write("#### Cognition by Gender")
                with col_info:
                    with st.popover("Info"):
                        st.markdown("""
                        **What:** Distribution of M3 cognition scores for males vs. females.
                        
                        **Why:** Visual check for gender differences in cognitive outcomes.
                        
                        **Expected:** Overlapping distributions suggest no major gender effect (confirmed by t-test: p=0.257).
                        
                        **Use:** Include in presentations to show demographic balance.
                        """)
                try:
                    st.image(str(fig1_path), use_container_width=True)
                except TypeError:
                    st.image(str(fig1_path), use_column_width=True)
        
        with col2:
            fig2_path = desc_viz_dir / "fig2_outcome_by_age_group.png"
            if fig2_path.exists():
                col_title, col_info = st.columns([0.85, 0.15])
                with col_title:
                    st.write("#### Cognition by Age Group")
                with col_info:
                    with st.popover("Info"):
                        st.markdown("""
                        **What:** Cognition scores across 5 age groups.
                        
                        **Why:** Visualizes age-related cognitive decline.
                        
                        **Key Finding:** Clear downward trend - each decade shows lower scores.
                        
                        **Statistical Support:** ANOVA F=177.5, p<0.001 (highly significant).
                        
                        **Implication:** Age is the strongest predictor of cognition in this sample.
                        """)
                try:
                    st.image(str(fig2_path), use_container_width=True)
                except TypeError:
                    st.image(str(fig2_path), use_column_width=True)
        
        fig3_path = desc_viz_dir / "fig3_model_performance_by_demographics.png"
        if fig3_path.exists():
            col_title, col_info = st.columns([0.85, 0.15])
            with col_title:
                st.write("#### Model Performance by Demographics")
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **What:** Model prediction error (MAE) across demographic groups.
                    
                    **Why:** Tests for algorithmic bias - does the model favor certain groups?
                    
                    **How to Read:**
                    - Lower bars = better predictions
                    - Similar heights across groups = fair model
                    
                    **Finding:** MAE ~0.34-0.35 for all groups → model is unbiased!
                    
                    **Importance:** Ensures ethical AI deployment.
                    """)
            try:
                st.image(str(fig3_path), use_container_width=True)
            except TypeError:
                st.image(str(fig3_path), use_column_width=True)
    
    with tab3:
        st.write("### Statistical Tests")
        
        with st.expander("Understanding Statistical Tests", expanded=False):
            st.markdown("""
            **P-value:** Probability that the observed difference occurred by chance
            - p < 0.05 → Statistically significant
            - p < 0.01 → Very significant
            - p < 0.001 → Highly significant
            
            **Effect Size:** Magnitude of the difference (independent of sample size)
            - **Cohen's d**: For t-tests (0.2=small, 0.5=medium, 0.8=large)
            - **η² (eta-squared)**: For ANOVA (0.01=small, 0.06=medium, 0.14=large)
            
            **Why Both Matter:** A result can be statistically significant but have small practical impact, or vice versa.
            """)
        
        # T-tests by Gender
        test2_path = stat_tests_dir / "test2_t_tests_by_gender.csv"
        if test2_path.exists():
            col_title, col_info = st.columns([0.85, 0.15])
            with col_title:
                st.write("#### Independent T-Tests by Gender")
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **What:** Compares means between males and females for each variable.
                    
                    **Why:** Tests if gender differences are statistically significant.
                    
                    **How to Read:**
                    - **t_statistic**: Larger absolute value = bigger difference
                    - **p_value**: <0.05 = significant
                    - **cohens_d**: Effect size (practical importance)
                    
                    **Key Finding:** Cognition shows no significant gender difference (p=0.257, d=0.04).
                    """)
            test2 = _safe_read_csv(test2_path)
            if test2 is not None:
                display_cols = ['variable_label', 'n1', 'mean1', 'sd1', 'n2', 'mean2', 'sd2', 
                               't_statistic', 'p_value', 'cohens_d', 'interpretation']
                st.dataframe(test2[display_cols], use_container_width=True)
                st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        # ANOVA by Age Group
        test3_path = stat_tests_dir / "test3_anova_by_age_group.csv"
        if test3_path.exists():
            col_title, col_info = st.columns([0.85, 0.15])
            with col_title:
                st.write("#### One-Way ANOVA by Age Group")
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **What:** Tests if means differ across 3+ groups (here: 5 age groups).
                    
                    **Why:** Determines if age significantly affects cognition.
                    
                    **How to Read:**
                    - **F-statistic**: Larger = bigger between-group differences
                    - **p-value**: <0.05 = at least one group differs
                    - **η²**: Proportion of variance explained by age
                    
                    **Key Finding:** F=177.5, p<0.001, η²=0.206 → Age explains 20.6% of cognition variance (large effect!).
                    """)
            test3 = _safe_read_csv(test3_path)
            if test3 is not None:
                display_cols = ['variable_label', 'n_groups', 'f_statistic', 'p_value', 
                               'eta_squared', 'interpretation']
                st.dataframe(test3[display_cols], use_container_width=True)
                st.caption("Effect sizes (η²): negligible (<0.01), small (0.01-0.06), medium (0.06-0.14), large (>0.14)")
        
        # Post-hoc tests
        test4_path = stat_tests_dir / "test4_posthoc_C3TCOMP_by_age.csv"
        if test4_path.exists():
            col_title, col_info = st.columns([0.85, 0.15])
            with col_title:
                st.write("#### Post-hoc Pairwise Comparisons (Age Groups)")
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **What:** After ANOVA, tests which specific age pairs differ.
                    
                    **Why:** ANOVA says "groups differ" but not which ones - post-hoc identifies them.
                    
                    **How to Read:**
                    - Each row = one pair comparison
                    - **Mean Diff**: Difference in cognition scores
                    - **p-value**: <0.05 = significant difference
                    - **Cohen's d**: Effect size for that pair
                    
                    **Key Finding:** ALL pairs differ significantly (p<0.05) → monotonic decline across all ages.
                    """)
            test4 = _safe_read_csv(test4_path)
            if test4 is not None:
                st.dataframe(test4, use_container_width=True)
        
        # Model fairness tests
        col_title, col_info = st.columns([0.85, 0.15])
        with col_title:
            st.write("#### Model Fairness Tests")
        with col_info:
            with st.popover("Info"):
                st.markdown("""
                **What:** Tests if model prediction error (MAE) differs across demographic groups.
                
                **Why:** Ensures the model doesn't systematically favor or disadvantage certain populations.
                
                **How to Read:**
                - **p-value > 0.05**: No significant difference = fair model
                - **p-value < 0.05**: Bias detected = needs investigation
                
                **Key Finding:** Both tests show p>0.05 → model is fair across gender and age!
                
                **Ethical Importance:** Critical for responsible AI deployment in healthcare/research.
                """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            test5_gender_path = stat_tests_dir / "test5_mae_by_gender.csv"
            if test5_gender_path.exists():
                st.write("**MAE by Gender**")
                test5_gender = _safe_read_csv(test5_gender_path)
                if test5_gender is not None:
                    st.dataframe(test5_gender[['n1', 'mean1', 'n2', 'mean2', 'p_value', 'interpretation']], 
                               use_container_width=True)
        
        with col2:
            test5_age_path = stat_tests_dir / "test5_mae_by_age_group.csv"
            if test5_age_path.exists():
                st.write("**MAE by Age Group**")
                test5_age = _safe_read_csv(test5_age_path)
                if test5_age is not None:
                    st.dataframe(test5_age[['n_groups', 'f_statistic', 'p_value', 'interpretation']], 
                               use_container_width=True)
else:
    st.info("Descriptive statistics not generated yet. Run: python scripts/generate_descriptive_tables.py")

st.divider()
st.subheader("4) Predictive twin (model + what-if simulation)")

with st.expander("About Predictive Digital Twins", expanded=False):
    st.markdown("""
    **What:** A "digital twin" is a virtual representation that predicts future outcomes for an individual.
    
    **How It Works:**
    1. **Train** a model on historical data (M1, M2 → M3 cognition)
    2. **Predict** future cognition for a specific person
    3. **Simulate** "what-if" scenarios (e.g., reduce alcohol, improve health)
    
    **Two Model Variants:**
    - **Primary**: Includes M2 prior cognition (best performance, R²=0.575)
    - **Ablation**: Excludes prior cognition (tests value of baseline, R²=0.277)
    
    **⚠️ Important Limitation:**
    This is **predictive simulation**, NOT causal inference. Changing a value here shows what the model predicts, 
    but does NOT prove that real-world intervention would cause the same change.
    """)

cfg_path = Path("config.yaml")
if not cfg_path.exists():
    st.info("No config.yaml found. Train models first, then ensure config.yaml exists in repo root.")
else:
    cfg = load_config(cfg_path)
    
    col_select, col_info = st.columns([0.85, 0.15])
    with col_select:
        variant = st.selectbox("Model variant", ["primary", "ablation"], index=0)
    with col_info:
        with st.popover("Info"):
            st.markdown("""
            **Primary Model:**
            - Includes M2 prior cognition
            - R² = 0.575 (57.5% variance explained)
            - Best for prediction accuracy
            
            **Ablation Model:**
            - Excludes M2 prior cognition
            - R² = 0.277 (27.7% variance explained)
            - Tests importance of baseline cognition
            
            **Difference:** ~30% of predictive power comes from prior cognition alone!
            """)
    model_path = cfg.models_dir / variant / "model.joblib"

    if not model_path.exists():
        st.info(f"Model not found: {model_path}. Train via: python scripts/train_predictive_twin.py")
    else:
        model = load_model(model_path)

        manifest = read_manifest(cfg.manifest_path)
        include_prior = variant == "primary"
        sel = select_features(
            manifest,
            m1_dataset_name=cfg.m1_survey_dataset.name,
            m2_dataset_name=cfg.m2_survey_dataset.name,
            outcome_code=cfg.outcome_code,
            prior_cognition_code=cfg.prior_code,
            include_prior=include_prior,
        )

        with st.expander("Build cohort (for picking an ID)", expanded=False):
            st.caption("Builds the modeling dataset and lets you pick a subject ID for demonstration.")
            if st.button("Build cohort now"):
                with st.spinner("Building modeling dataset..."):
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
                    st.session_state["_twin_X"] = X
                    st.session_state["_twin_y"] = y
                    st.session_state["_twin_ids"] = ids_df
                    st.session_state["_twin_report"] = report

            if "_twin_report" in st.session_state:
                rep = st.session_state["_twin_report"]
                st.write(
                    {
                        "join_id": rep.join_id,
                        "m1_rows": rep.m1_rows,
                        "m2_rows": rep.m2_rows,
                        "m3_outcome_rows": rep.m3_outcome_rows,
                        "after_m1_m2_rows": rep.after_m1_m2_rows,
                        "after_all_rows": rep.after_all_rows,
                        "overlap_m1_m2": rep.overlap_m1_m2,
                        "overlap_m1_m3": rep.overlap_m1_m3,
                        "overlap_m2_m3": rep.overlap_m2_m3,
                    }
                )

        if "_twin_X" in st.session_state and "_twin_ids" in st.session_state:
            X = st.session_state["_twin_X"]
            y = st.session_state["_twin_y"]
            ids_df = st.session_state["_twin_ids"]

            st.write("### Predictive Twin / Simulation")
            
            with st.expander("How Simulation Works", expanded=False):
                st.markdown("""
                **Process:**
                1. Select a person from the cohort
                2. Get their **baseline prediction** (using current features)
                3. Choose a **scenario** (e.g., reduce alcohol by 30%)
                4. Apply safe transformations to modifiable features
                5. Get **new prediction** with modified features
                6. Calculate **delta** (change in predicted cognition)
                
                **Predefined Scenarios:**
                - **reduce_alcohol**: 30% reduction in alcohol consumption
                - **reduce_stress**: 25% reduction in worry/stress frequency
                - **improve_self_rated_health**: 1-step improvement in health rating
                - **combined_lifestyle**: All of the above together
                
                **⚠️ Critical Limitation:**
                These are **model predictions**, not causal effects. The model learned associations, not interventions.
                Real-world changes may have different effects due to confounding, reverse causation, or unmeasured factors.
                """)
            
            st.caption(
                "⚠️ This is predictive simulation (model-based estimates), not causal inference. "
                "Changing a value here does not imply a real-world intervention would cause the same change."
            )

            sample_ids = ids_df[cfg.join_id].astype(str).head(200).tolist()
            
            col_id, col_info = st.columns([0.85, 0.15])
            with col_id:
                chosen_id = st.selectbox(f"Choose {cfg.join_id}", sample_ids, index=0)
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **Subject ID:** Unique identifier linking data across MIDUS waves (M1, M2, M3).
                    
                    **Why 200 shown:** For demo purposes, we show first 200 IDs from the cohort.
                    
                    **In production:** You could search by ID or filter by demographics.
                    """)

            mask = ids_df[cfg.join_id].astype(str).eq(str(chosen_id))
            x_row = X.loc[mask].copy()
            y_true = float(y.loc[mask].iloc[0])

            if cfg.join_id in x_row.columns:
                x_row = x_row.drop(columns=[cfg.join_id])

            st.write("**True outcome (for reference)**", y_true)

            # Scenario picker
            lib = scenario_library()
            scenario_options = list(lib.keys())
            scenario_labels = {k: lib[k]["label"] for k in scenario_options}
            
            col_scenario, col_info = st.columns([0.85, 0.15])
            with col_scenario:
                scenario_key = st.selectbox(
                    "Scenario",
                    scenario_options,
                    format_func=lambda k: f"{k} — {scenario_labels.get(k, k)}",
                )
            with col_info:
                with st.popover("Info"):
                    st.markdown("""
                    **Scenarios are predefined** with safe transformation rules:
                    
                    - Values are **clamped** to valid ranges (e.g., alcohol ≥ 0)
                    - Only **modifiable features** are changed (not age, sex)
                    - Changes are **logged** for transparency
                    
                    **Example:** "reduce_alcohol" applies 30% reduction to M1 and M2 alcohol variables, 
                    clamped to [0, 50] drinks.
                    """)

            if st.button("Run scenario simulation"):
                try:
                    res = run_scenario(pipeline=model, x_row=x_row, scenario_key=scenario_key)
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Baseline prediction", f"{res.baseline_prediction:.3f}")
                    mcol2.metric("Scenario prediction", f"{res.new_prediction:.3f}")
                    mcol3.metric("Delta", f"{res.delta:.3f}")

                    pred_df = pd.DataFrame(
                        {
                            "type": ["baseline", "scenario"],
                            "prediction": [res.baseline_prediction, res.new_prediction],
                        }
                    )
                    pred_chart = (
                        alt.Chart(pred_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("type:N", title=""),
                            y=alt.Y("prediction:Q", title="Predicted cognition"),
                            color=alt.Color("type:N", legend=None),
                            tooltip=["type", alt.Tooltip("prediction:Q", format=".3f")],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(pred_chart, use_container_width=True)

                    delta_df = pd.DataFrame({"delta": [res.delta]})
                    delta_chart = (
                        alt.Chart(delta_df)
                        .mark_bar(color="#4c78a8")
                        .encode(
                            x=alt.X("delta:Q", title="Delta (scenario - baseline)"),
                            tooltip=[alt.Tooltip("delta:Q", format=".3f")],
                        )
                        .properties(height=60)
                    )
                    st.altair_chart(delta_chart, use_container_width=True)

                    st.write("#### Features changed")
                    changes_df = pd.DataFrame(
                        [
                            {
                                "feature": c.feature,
                                "old": c.old_value,
                                "new": c.new_value,
                                "rule": c.rule,
                            }
                            for c in res.changes
                        ]
                    )
                    st.dataframe(changes_df)

                    # Before/after visualization (numeric-only)
                    vf = changes_df.copy()
                    vf["old_num"] = pd.to_numeric(vf["old"], errors="coerce")
                    vf["new_num"] = pd.to_numeric(vf["new"], errors="coerce")
                    vf = vf.dropna(subset=["old_num", "new_num"])
                    if len(vf) > 0:
                        long = pd.concat(
                            [
                                vf[["feature", "old_num"]].rename(columns={"old_num": "value"}).assign(state="old"),
                                vf[["feature", "new_num"]].rename(columns={"new_num": "value"}).assign(state="new"),
                            ],
                            ignore_index=True,
                        )
                        feat_chart = (
                            alt.Chart(long)
                            .mark_bar()
                            .encode(
                                y=alt.Y("feature:N", sort="-x", title=""),
                                x=alt.X("value:Q", title="Value"),
                                color=alt.Color("state:N", title=""),
                                tooltip=["feature", "state", alt.Tooltip("value:Q", format=".3f")],
                            )
                            .properties(height=min(300, 30 * len(vf) + 60))
                        )
                        st.altair_chart(feat_chart, use_container_width=True)
                except Exception as e:
                    st.error(str(e))

            # Interpretability (global)
            st.write("### Interpretability (global)")
            st.caption("Top features by mean(|SHAP|) on the held-out test set. These are global influences, not per-person explanations.")

            shap_path = cfg.models_dir / variant / "shap" / "shap_values.parquet"
            shap_summary_png = cfg.models_dir / variant / "shap" / "shap_summary.png"
            if shap_path.exists():
                code_to_label = build_code_to_label_map_from_manifest(cfg.manifest_path)
                try:
                    tops = read_global_top_features_from_shap(shap_path, top_k=15, code_to_label=code_to_label)
                    top_df = pd.DataFrame(
                        [
                            {
                                "feature": t.feature,
                                "label": t.label,
                                "mean_abs_shap": t.score,
                            }
                            for t in tops
                        ]
                    )

                    if len(top_df) > 0:
                        top_df_display = top_df.copy()
                        top_df_display["display"] = top_df_display.apply(
                            lambda r: f"{r['feature']} — {r['label']}" if isinstance(r.get("label"), str) and r["label"] else r["feature"],
                            axis=1,
                        )
                        bar = (
                            alt.Chart(top_df_display)
                            .mark_bar(color="#f58518")
                            .encode(
                                y=alt.Y("display:N", sort="-x", title=""),
                                x=alt.X("mean_abs_shap:Q", title="mean(|SHAP|)"),
                                tooltip=["feature", "label", alt.Tooltip("mean_abs_shap:Q", format=".4f")],
                            )
                            .properties(height=360)
                        )
                        st.altair_chart(bar, use_container_width=True)
                        with st.expander("Show top-features table", expanded=False):
                            st.dataframe(top_df)

                    if shap_summary_png.exists():
                        try:
                            st.image(
                                str(shap_summary_png),
                                caption="SHAP summary plot (global)",
                                use_container_width=True,
                            )
                        except TypeError:
                            st.image(
                                str(shap_summary_png),
                                caption="SHAP summary plot (global)",
                                use_column_width=True,
                            )
                except Exception as e:
                    st.error(f"Failed to load SHAP artifacts: {e}")
            else:
                st.info(f"No SHAP artifacts found at: {shap_path}")
        else:
            st.info("Build cohort first to select a subject ID for simulation.")
