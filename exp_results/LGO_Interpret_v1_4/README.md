# LGO vs InterpretML (EBM) Comparison Framework

Version 1.4 — December 2025

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive experimental framework for comparing **Logistic-Gated Operators (LGO)** with **Explainable Boosting Machine (EBM)** from InterpretML on clinical risk prediction tasks.

## Overview

This repository provides tools for fair comparison between two highly interpretable machine learning methods:

| Method | Type | Interpretability | Output |
|--------|------|------------------|--------|
| **LGO** | Symbolic Regression | Unit-aware thresholds in mathematical formulas | Explicit equations with gating functions |
| **EBM** | Generalized Additive Model + Boosting | Shape functions per feature | Additive contributions with interactions |

Both methods prioritize interpretability while maintaining competitive predictive performance.

---

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Running Comparison Experiments](#running-comparison-experiments)
  - [Visualization](#visualization)
  - [Statistical Analysis](#statistical-analysis)
- [Methodology](#methodology)
- [Output Files](#output-files)
- [Using as Standalone](#using-as-standalone)
- [References](#references)

---

## Key Features

- **LGO v2.2**: DEAP-based symbolic regression with interpretable threshold gates
  - Hard/soft gating mechanisms
  - Unit-aware threshold recovery
  - Probability calibration (Platt/Isotonic)

- **EBM v1.0**: InterpretML's Explainable Boosting Machine wrapper
  - Shape functions for each feature
  - Automatic interaction detection
  - Global and local explanations

- **Fair Comparison Framework**
  - Same train/val/test splits
  - Multi-seed experiments (10 Fibonacci seeds)
  - Bootstrap confidence intervals
  - Comprehensive statistical tests

- **Multi-Dataset Support**
  - ICU (MIMIC-IV composite risk score)
  - eICU (composite risk score)
  - NHANES (metabolic score)

---

## Project Structure

```
LGO_Interpret_v1_4/
├── LGO_v2_2.py                      # LGO core engine (DEAP-based, v2.2)
├── InterpretML_v1.py                # EBM wrapper for InterpretML
├── run_lgo_interpret_comparison.py  # Main comparison script (multi-seed)
├── run_lgo_interpret_plot.py        # Visualization script
├── statistical_analysis.py          # Comprehensive statistical tests
├── test_setup.py                    # Dependency verification script
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This document
```

**Generated directories (created by scripts):**
```
├── *_results_*/                     # Experiment results
│   ├── multiseed_results_*.csv      # Per-seed metrics
│   ├── multiseed_summary_*.csv      # Aggregated statistics
│   ├── multiseed_detailed_*.pkl     # Detailed data for plotting
│   └── comparison_seed*_*.png       # Comparison figures
│
└── stats_output/                    # Statistical analysis outputs
    ├── summary_table_*.csv
    ├── table_*.tex                  # LaTeX table
    └── detailed_stats_*.json
```

---

## Installation

### Requirements

- Python ≥ 3.8
- NumPy, Pandas, SciPy, Scikit-learn
- Matplotlib, Seaborn (visualization)
- DEAP (for LGO symbolic regression)
- InterpretML (for EBM)

### Setup

```bash
# Navigate to the project directory
cd LGO_Interpret_v1_4

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy pandas scipy scikit-learn matplotlib seaborn deap interpret
```

### Verify Installation

Run the test script to check all dependencies:
```bash
python test_setup.py
```

Or verify manually:
```python
# Check LGO
from LGO_v2_2 import run_lgo_sr_v3
print("LGO OK")

# Check EBM
from interpret.glassbox import ExplainableBoostingClassifier
print("EBM OK")
```

---

## Quick Start

```bash
# Run comparison with synthetic data (no external data required)
python run_lgo_interpret_comparison.py \
  --seeds 1,2,3 \
  --output_dir results_quick

# Generate visualizations
python run_lgo_interpret_plot.py \
  --detailed_pkl results_quick/multiseed_detailed_*.pkl \
  --plot_summary \
  --output_dir results_quick
```

---

## Usage

### Running Comparison Experiments

#### Single Seed Experiment

```bash
python run_lgo_interpret_comparison.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --seed 42 \
  --output_dir results_single
```

#### Multi-Seed Experiments (Recommended)

**ICU Dataset:**
```bash
python run_lgo_interpret_comparison.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --calibration_method platt \
  --output_dir ICU_results_30k
```

**eICU Dataset:**
```bash
python run_lgo_interpret_comparison.py \
  --data_path ../data/eICU/eICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --calibration_method platt \
  --output_dir eICU_results_30k
```

**NHANES Dataset:**
```bash
python run_lgo_interpret_comparison.py \
  --data_path ../data/NHANES/NHANES_metabolic_score.csv \
  --outcome metabolic_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --calibration_method platt \
  --output_dir NHANES_results_30k
```

**Synthetic Data (no external files needed):**
```bash
python run_lgo_interpret_comparison.py \
  --seeds 1,2,3 \
  --output_dir results_synthetic
```

#### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to CSV data file | Auto-generates synthetic data |
| `--outcome` | Target variable column name | `composite_risk_score` |
| `--binarize_threshold` | Threshold for binary classification | None |
| `--gate_type` | LGO gate type: `hard` or `soft` | `hard` |
| `--n_generations` | Evolutionary generations (LGO) | `100` |
| `--population_size` | Population size per generation (LGO) | `300` |
| `--ebm_interactions` | Number of automatic interactions (EBM) | `10` |
| `--ebm_learning_rate` | Learning rate (EBM) | `0.01` |
| `--calibration_method` | LGO calibration: `platt`, `isotonic`, `none` | `platt` |
| `--seed` | Single random seed | — |
| `--seeds` | Comma-separated random seeds (multi-seed mode) | — |
| `--output_dir` | Output directory | `./results` |

### Visualization

#### Single Seed Detailed Plot

```bash
# Generate 6-panel comparison for a specific seed
python run_lgo_interpret_plot.py \
  --detailed_pkl ICU_results_30k/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --output_dir ICU_results_30k
```

#### Generate All Seed Plots

```bash
python run_lgo_interpret_plot.py \
  --detailed_pkl ICU_results_30k/multiseed_detailed_*.pkl \
  --plot_all \
  --output_dir ICU_results_30k
```

#### Multi-Seed Summary

```bash
python run_lgo_interpret_plot.py \
  --results_csv ICU_results_30k/multiseed_results_*.csv \
  --plot_summary \
  --output_dir ICU_results_30k
```

#### Scaling Analysis (Multiple Budgets)

```bash
python run_lgo_interpret_plot.py \
  --results_dirs results_30k,results_100k,results_200k \
  --budgets 30000,100000,200000 \
  --plot_scaling \
  --output_dir results
```

### Statistical Analysis

```bash
# Analyze single result directory
python statistical_analysis.py \
  --results_csv ICU_results_30k/multiseed_results_*.csv \
  --output_dir stats_output

# Analyze multiple result directories
python statistical_analysis.py \
  --results_dirs results_30k,results_100k,results_200k,results_500k \
  --output_dir stats_output
```

**Output files:**
- `summary_table_*.csv` — Summary statistics
- `table_*.tex` — LaTeX table (copy-paste ready)
- `detailed_stats_*.json` — Complete statistical results

**Statistical tests included:**
- Wilcoxon signed-rank test (non-parametric)
- Paired t-test (parametric)
- Effect sizes: Cohen's d, Cliff's δ, CLES
- Bootstrap confidence intervals (percentile and BCa)
- Multiple comparison corrections: Bonferroni, Holm-Bonferroni, FDR

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

Key features:
- Thresholds are explicit parameters in the formula
- Unit-aware: thresholds can be mapped back to physical units
- Parsimonious: gates are pruned when not needed

### EBM: Explainable Boosting Machine

EBM is a Generalized Additive Model (GAM) trained with boosting:

```
f(x) = β₀ + Σᵢ fᵢ(xᵢ) + Σᵢⱼ fᵢⱼ(xᵢ, xⱼ)
```

Key features:
- Shape functions: visualize each feature's contribution
- Automatic interaction detection
- Global and local explanations
- Competitive accuracy with tree ensembles

### Fairness Considerations

| Aspect | LGO | EBM | Treatment |
|--------|-----|-----|-----------|
| **Training Data** | train+val combined | train+val combined | ✅ Consistent |
| **Test Data** | test set | test set | ✅ Consistent |
| **Data Split** | Same random seed | Same random seed | ✅ Consistent |
| **Calibration** | Platt scaling | Built-in | ⚠️ Different methods |
| **Output** | Symbolic expression | Shape functions | ⚠️ Different paradigms |

---

## Output Files

### Per-Experiment Outputs (`*_results_*/`)

| File | Description |
|------|-------------|
| `multiseed_results_*.csv` | Per-seed metrics for both methods |
| `multiseed_summary_*.csv` | Aggregated statistics across seeds |
| `multiseed_detailed_*.pkl` | Detailed results for plotting |
| `comparison_seed*_*.png` | 6-panel comparison figure |
| `multiseed_summary_*.png` | Multi-seed summary figure |

### Statistical Analysis Outputs (`stats_output/`)

| File | Description |
|------|-------------|
| `summary_table_*.csv` | Summary statistics table |
| `table_*.tex` | LaTeX table for papers |
| `detailed_stats_*.json` | Complete statistical results |

---

## Using as Standalone

### LGO Standalone

```python
from LGO_v2_2 import run_lgo_sr_v3, ZScaler
import pandas as pd

# Prepare data
X_train = pd.DataFrame(...)  # Your features
y_train = ...                 # Your labels

# Optional: Z-score normalization
scaler = ZScaler.fit(X_train.values)
X_scaled = scaler.transform(X_train.values)

# Run LGO symbolic regression
results = run_lgo_sr_v3(
    X=X_scaled,
    y=y_train,
    feature_names=list(X_train.columns),
    experiment='lgo_hard',    # or 'lgo_soft'
    pop_size=300,
    ngen=100,
    max_height=10,
    random_state=42,
)

# Get best formula
print(results.iloc[0]['expr'])
```

### EBM Standalone

```python
from InterpretML_v1 import get_ebm, EBMConfig

# Configure EBM
config = EBMConfig(
    interactions=10,
    learning_rate=0.01,
    random_state=42,
    verbose=True
)

# Train EBM
ebm = get_ebm(config=config)
ebm.fit(X_train, y_train, task='classification')

# Predict
y_prob = ebm.predict_proba(X_test)

# Get feature importances
importances = ebm.get_feature_importances()
print(importances.head(10))

# Get model summary
summary = ebm.get_model_summary()
print(f"Number of terms: {summary['n_terms']}")
print(f"Number of interactions: {summary['n_interactions']}")
```

---

## Changelog

### v1.4 (December 2025)
- LGO engine v2.2 with external calibration support
- InterpretML wrapper v1.0 with compatibility for multiple interpret versions
- Multi-dataset support: ICU, eICU, NHANES
- Comprehensive statistical analysis with effect sizes and multiple comparison corrections
- Enhanced visualization with EBM shape function plots

---

## References

- **LGO**: Deng et al., "Logistic-Gated Operators Enable Auditable Unit-Aware Thresholds in Symbolic Regression"
  - [GitHub](https://github.com/oudeng/LGO)

- **InterpretML**: Nori et al., "InterpretML: A Unified Framework for Machine Learning Interpretability"
  - [Paper](https://arxiv.org/abs/1909.09223)
  - [GitHub](https://github.com/interpretml/interpret)

- **EBM**: Lou et al., "Accurate Intelligible Models with Pairwise Interactions"
  - [Paper](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf)

- **DEAP**: Fortin et al., "DEAP: Evolutionary Algorithms Made Easy"
  - [GitHub](https://github.com/DEAP/deap)

---