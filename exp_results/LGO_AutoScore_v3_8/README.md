# LGO vs AutoScore Comparison Framework

Version 3.8 — December 2025

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive experimental framework for comparing **Logistic-Gated Operators (LGO)** with **AutoScore** on clinical risk prediction tasks. This repository contains the complete codebase for reproducible experiments, statistical analysis, and visualization.

## Highlights

- **LGO v2.2**: DEAP-based symbolic regression engine with interpretable threshold gates and external calibration support
- **AutoScore v2.0**: Pure Python implementation (no R dependency), fully compatible with the original R version
- **Multi-dataset support**: ICU (MIMIC-IV), eICU, and NHANES datasets
- **Rigorous comparison**: Multi-seed experiments across multiple computational budgets (30k–500k evaluations)
- **Complete statistical analysis**: Wilcoxon tests, effect sizes, bootstrap CIs, multiple comparison corrections

---

## Table of Contents

- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Running Comparison Experiments](#running-comparison-experiments)
  - [Statistical Analysis](#statistical-analysis)
  - [Visualization](#visualization)
  - [Using LGO Standalone](#using-lgo-standalone)
  - [Using AutoScore Standalone](#using-autoscore-standalone)
- [Methodology](#methodology)
- [Output Files](#output-files)
- [References](#references)

---

## Key Results

Performance comparison on ICU Composite Risk Score dataset (N=2,939; 10 random seeds):

| Budget | LGO AUROC | AutoScore AUROC | ΔAUROC | 95% CI | p-value | Cohen's d | Win Rate |
|--------|-----------|-----------------|--------|--------|---------|-----------|----------|
| 30k | 0.927±0.064 | 0.864±0.009 | **+0.063** | [0.022, 0.098] | 0.027* | 1.31 (large) | 7/10 |
| 100k | 0.910±0.062 | 0.864±0.009 | **+0.045** | [0.006, 0.083] | 0.084 | 0.96 (large) | 6/10 |
| 200k | 0.908±0.062 | 0.864±0.009 | **+0.044** | [0.005, 0.081] | 0.084 | 0.95 (large) | 6/10 |
| 300k | 0.859±0.137 | 0.864±0.009 | −0.006 | [−0.100, 0.066] | 0.846 | −0.06 (negligible) | 4/10 |
| 500k | 0.884±0.142 | 0.864±0.009 | +0.019 | [−0.079, 0.091] | 0.275 | 0.18 (negligible) | 6/10 |

*Note: \* p<0.05 (Wilcoxon signed-rank test).*

---

## Project Structure

```
LGO_AutoScore_v3_8/
├── LGO_v2_2.py                    # LGO core engine (DEAP v2.1 + external calibration)
├── AutoScore_v2.py                # AutoScore implementation (R-compatible, pure Python)
│
├── run_comparison_v2_5.py         # Main comparison script (multi-seed experiments)
├── run_comparison_plot_v1.py      # Single-seed detailed visualization
├── plot_scaling_analysis_v3.py    # Cross-budget scaling analysis
├── statistical_analysis_v2_1.py   # Comprehensive statistical tests
│
├── stats_output/                  # Statistical analysis outputs
│   ├── ICU/                       # ICU dataset results
│   ├── eICU/                      # eICU dataset results
│   └── NHANES/                    # NHANES dataset results
│
├── requirements.txt               # Python dependencies
└── README.md                      # This document
```

---

## Installation

### Requirements

- Python ≥ 3.8
- NumPy, Pandas, Scikit-learn, Matplotlib
- DEAP (for LGO symbolic regression)
- Seaborn (optional, for visualization)

### Setup

```bash
# Clone or download the repository
cd LGO_AutoScore_v3_8

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Run comparison with ICU dataset
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --output_dir results_quick
```

---

## Usage

### Running Comparison Experiments

The main comparison script `run_comparison_v2_5.py` runs both LGO and AutoScore with multiple random seeds.

#### ICU Dataset Examples

```bash
# 30k evaluations (100 gen × 300 pop)
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_30k_full

# 100k evaluations (100 gen × 1000 pop)
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_100k_full

# 200k evaluations (200 gen × 1000 pop)
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 200 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_200k_full

# 500k evaluations (500 gen × 1000 pop)
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 500 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_500k_full
```

#### eICU and NHANES Dataset Examples

```bash
# eICU dataset
python run_comparison_v2_5.py \
  --data_path ../data/eICU/eICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir eICU_results_30k_full

# NHANES dataset
python run_comparison_v2_5.py \
  --data_path ../data/NHANES/NHANES_metabolic_score.csv \
  --outcome metabolic_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir NHANES_results_30k_full
```

#### Fair Comparison Mode

To restrict LGO to the same top-n features as AutoScore:

```bash
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --fair_features \
  --calibration_method platt \
  --output_dir ICU_results_30k_fair
```

#### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to CSV data file | Required |
| `--outcome` | Target variable column name | `composite_risk_score` |
| `--binarize_threshold` | Threshold for binary classification | `5` |
| `--gate_type` | LGO gate type: `hard` or `soft` | `hard` |
| `--n_generations` | Number of evolutionary generations | `100` |
| `--population_size` | Population size per generation | `300` |
| `--seeds` | Comma-separated random seeds | `1,2,3,5,8,13,21,34,55,89` |
| `--n_variables` | Number of variables for AutoScore | `6` |
| `--fair_features` | Restrict LGO to same features as AutoScore | `False` |
| `--max_height` | Maximum LGO expression tree height | `10` |
| `--calibration_method` | LGO calibration: `platt`, `isotonic`, or `none` | `platt` |
| `--output_dir` | Output directory for results | `results_*` |

### Statistical Analysis

Run comprehensive statistical tests on experiment results:

```bash
# ICU dataset analysis
python statistical_analysis_v2_1.py \
  --results_dirs ICU_results_30k_full,ICU_results_100k_full,ICU_results_200k_full,ICU_results_300k_full,ICU_results_500k_full \
  --output_dir stats_output/ICU

# eICU dataset analysis
python statistical_analysis_v2_1.py \
  --results_dirs eICU_results_30k_full,eICU_results_100k_full,eICU_results_200k_full,eICU_results_300k_full \
  --output_dir stats_output/eICU

# NHANES dataset analysis
python statistical_analysis_v2_1.py \
  --results_dirs NHANES_results_30k_full,NHANES_results_100k_full,NHANES_results_200k_full,NHANES_results_300k_full \
  --output_dir stats_output/NHANES
```

**Output files:**
- `summary_table_*.csv` — Summary statistics table
- `table_*.tex` — LaTeX table (copy-paste ready for papers)
- `detailed_stats_*.json` — Complete statistical results (JSON)

**Statistical tests included:**
- Wilcoxon signed-rank test (non-parametric pairwise comparison)
- Paired t-test (parametric)
- Effect sizes: Cohen's d, Cliff's δ, CLES
- Bootstrap confidence intervals (percentile and BCa methods)
- Trend analysis: Spearman correlation, Jonckheere-Terpstra test
- Multiple comparison corrections: Bonferroni, Holm-Bonferroni, FDR

### Visualization

#### Scaling Analysis (Cross-Budget)

```bash
python plot_scaling_analysis_v3.py \
  --results_dirs ICU_results_30k_full,ICU_results_100k_full,ICU_results_200k_full,ICU_results_500k_full \
  --output scaling_analysis.png
```

#### Single-Seed Detailed Plots

```bash
python run_comparison_plot_v1.py \
  --results_csv ICU_results_30k_full/multiseed_results_*.csv \
  --detailed_pkl ICU_results_30k_full/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_30k_full
```

### Using LGO Standalone

```python
from LGO_v2_2 import run_lgo_sr_v3, ZScaler
import numpy as np
import pandas as pd

# Prepare data
X = df[feature_columns].values
y = df[outcome_column].values

# Z-score normalization (recommended)
scaler = ZScaler.fit(X)
X_scaled = scaler.transform(X)

# Run LGO symbolic regression
result_df = run_lgo_sr_v3(
    X=X_scaled,
    y=y,
    feature_names=feature_columns,
    experiment='lgo_hard',      # 'lgo_hard' or 'lgo_soft'
    typed_mode='light',
    pop_size=300,               # Population size
    ngen=100,                   # Number of generations
    max_height=10,              # Max tree depth
    random_state=42,
)

# Get the best expression
best_expr = result_df.iloc[0]['expr']
print(f"Best formula: {best_expr}")
```

### Using AutoScore Standalone

```python
from AutoScore_v2 import AutoScore, AutoScoreConfig
import pandas as pd

# Configuration
config = AutoScoreConfig(
    n_trees=100,                # Random forest trees for variable ranking
    quantiles=[0, 0.05, 0.2, 0.8, 0.95, 1.0],  # Binning quantiles (R-compatible)
    max_score=100,              # Maximum total score
    verbose=True
)

# Initialize
autoscore = AutoScore(config)

# Step 1: Variable ranking
ranking = autoscore.rank_variables(train_data, outcome_col='label')
print("Variable ranking:")
print(ranking)

# Step 2: Build score table with top-n variables
n_variables = 6
autoscore.build_score_table(
    train_data=train_data,
    selected_vars=list(ranking.index[:n_variables]),
    outcome_col='label'
)
print("Score table:")
print(autoscore.score_table)

# Step 3: Compute scores and evaluate
scores = autoscore.compute_score(test_data)
auroc = autoscore.evaluate(test_data, outcome_col='label')['auroc']
print(f"Test AUROC: {auroc:.4f}")
```

---

## Methodology

### LGO: Logistic-Gated Operators

LGO uses genetic programming (DEAP) to evolve symbolic expressions with interpretable threshold gates:

**Soft Gate** (amplitude-preserving):
```
lgo(x, a, b) = x · σ(a · (x − b))
```

**Hard Gate** (binary threshold):
```
lgo_thre(x, a, b) = σ(a · (x − b))
```

Where:
- `x`: Input feature (z-score normalized)
- `a`: Transition steepness (controls sharpness)
- `b`: Threshold position (in z-score space)
- `σ`: Sigmoid function

Additional operators include `lgo_pair`, `lgo_and2`, `lgo_or2`, `lgo_and3`, and `gate_expr` for complex logical combinations.

### AutoScore

AutoScore v2.0 follows a 6-module pipeline (R-compatible):

1. **Variable Ranking** — Random Forest feature importance
2. **Parsimony Analysis** — Evaluate AUC vs. number of variables
3. **Variable Binning** — Non-uniform quantiles: `[0, 0.05, 0.2, 0.8, 0.95, 1.0]`
4. **Score Derivation** — Two-step logistic regression → integer scores
5. **Fine-tuning** — Optional manual adjustment of cut points
6. **Performance Evaluation** — AUROC, AUPRC, Brier score, etc.

### Fairness Considerations

| Aspect | LGO | AutoScore | Treatment |
|--------|-----|-----------|-----------|
| **Training Data** | train+val combined | train+val combined | ✅ Consistent |
| **Test Data** | test set | test set | ✅ Consistent |
| **Data Split** | Same random seed | Same random seed | ✅ Consistent |
| **Features** | All or top-n (fair mode) | Top-n (RF ranking) | ✅ Configurable |
| **Calibration** | Platt/Isotonic (optional) | Score-to-prob mapping | ⚠️ Different |
| **Output** | Symbolic expression | Integer score card | ⚠️ Different |

---

## Output Files

### Per-Experiment Outputs (`*_results_*/`)

| File | Description |
|------|-------------|
| `multiseed_results_*.csv` | Per-seed metrics for both methods |
| `multiseed_summary_*.csv` | Aggregated statistics across seeds |
| `multiseed_detailed_*.pkl` | Detailed results (predictions, formulas, etc.) |

### Statistical Analysis Outputs (`stats_output/`)

| File | Description |
|------|-------------|
| `summary_table_*.csv` | Summary statistics table |
| `table_*.tex` | LaTeX table for papers |
| `detailed_stats_*.json` | Complete statistical results |

---

## Changelog

### v3.8 (December 2025)
- LGO engine v2.2: DEAP v2.1 core with external calibration support
- AutoScore v2.0: R-compatible pure Python implementation
- run_comparison_v2.5: Improved binarization handling, explicit argument passing
- Multi-dataset support: ICU, eICU, NHANES
- Comprehensive statistical analysis with multiple comparison corrections

---

## References

- **LGO**: [https://github.com/oudeng/LGO](https://github.com/oudeng/LGO)
- **AutoScore**: Xie F, et al. *AutoScore: A Machine Learning-Based Automatic Clinical Score Generator and Its Application to Mortality Prediction Using Electronic Health Records.* JMIR Medical Informatics. 2020. [GitHub](https://github.com/nliulab/AutoScore)
- **DEAP**: Fortin FA, et al. *DEAP: Evolutionary Algorithms Made Easy.* Journal of Machine Learning Research. 2012. [GitHub](https://github.com/DEAP/deap)

---

