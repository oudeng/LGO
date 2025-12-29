# LGO Utility Analysis Scripts

Version 1.0 — December 2025

A collection of post-processing scripts for analyzing LGO (Logistic-Gated Operators) experimental results. These scripts generate complexity statistics, Pareto fronts, gating usage analysis, threshold auditing, and stability summaries.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Pipeline Workflow](#pipeline-workflow)
- [Script Reference](#script-reference)
  - [01 - Collect Hyperparameters](#01_collect_hyperparamspy)
  - [02 - Collect Runtime Profile](#02_collect_runtime_profilepy)
  - [03 - Export Expressions](#03_export_expressionspy)
  - [04 - Complexity & Stability](#04_gen_complexity_and_stabilitypy)
  - [05 - Pareto Front](#05_gen_pareto_frontpy)
  - [06 - Gating Usage](#06_gen_gating_usagepy)
  - [07 - Thresholds & Units](#07_gen_thresholds_unitspy)
  - [08 - Ground Truth Conversion](#08_convert_ground_truth_to_guidelinespy)
  - [08 - Threshold Audit](#08_threshold_auditpy)
  - [09 - Feature Threshold Sanity](#09_feature_threshold_sanitypy)
  - [10 - Ablation Matrix](#10_build_ablation_matrixpy)
  - [11 - Stability Summary](#11_stability_summarypy)
- [Optional Utilities](#optional-utilities)
- [Output Files](#output-files)

---

## Overview

These scripts form a post-processing pipeline for LGO experiments:

```
Raw Results → Expression Export → Complexity Analysis → Threshold Audit → Summary Statistics
```

The pipeline supports:
- Multi-seed experiments (Fibonacci seeds: 1,2,3,5,8,13,21,34,55,89)
- Multiple datasets (ICU, eICU, NHANES, UCI)
- Multiple experiment types (base, lgo_soft, lgo_hard)

---

## Directory Structure

### Input Structure

```
overall_{dataset_name}/
├── candidates/
│   ├── candidates_{method}_{experiment}_seed{N}.csv
│   └── ...
├── runs/
│   ├── {method}_{experiment}_seed{N}/
│   │   └── hparams.json
│   └── ...
├── predictions/                     # If --save_predictions used
│   └── test_predictions_{method}_{experiment}_seed{N}.csv
└── config/
    ├── scaler.json
    └── units.yaml
```

### Output Structure

```
overall_{dataset_name}/
├── aggregated/
│   ├── hyperparams.csv              # From 01
│   ├── runtime_profile.csv          # From 02
│   ├── complexity_by_model.csv      # From 04
│   ├── complexity_stats.csv         # From 04
│   ├── pareto_front.csv             # From 05
│   ├── gating_usage.csv             # From 06
│   ├── thresholds_units.csv         # From 07
│   ├── threshold_audit.csv          # From 08
│   ├── threshold_audit_summary.csv  # From 08
│   ├── ablation_table.csv           # From 10
│   ├── stability_summary.csv        # From 11
│   ├── calibration_bins.csv         # From opt_gen_calibration
│   └── calibration_ece.csv          # From opt_gen_calibration
├── expressions/
│   ├── topk_expressions.csv         # From 03
│   └── top1_expressions.txt         # From 03
└── config/
    ├── guidelines.yaml              # From 08_convert
    └── ground_truth_snapshot.json   # From 08_convert
```

---

## Pipeline Workflow

### Recommended Execution Order

```bash
# Step 1: Collect metadata
python 01_collect_hyperparams.py --dataset_dir overall_ICU
python 02_collect_runtime_profile.py --dataset_dir overall_ICU

# Step 2: Export and analyze expressions
python 03_export_expressions.py --dataset_dir overall_ICU --method all --topk 10
python 04_gen_complexity_and_stability.py --dataset_dir overall_ICU

# Step 3: Generate analysis files
python 05_gen_pareto_front.py --dataset_dir overall_ICU
python 06_gen_gating_usage.py --dataset_dir overall_ICU --topk 100

# Step 4: Threshold analysis (requires ground truth)
python 08_convert_ground_truth_to_guidelines.py \
  --ground_truth data/ICU/ground_truth.json \
  --output_dir overall_ICU/config \
  --dataset ICU
python 07_gen_thresholds_units.py --dataset_dir overall_ICU --method lgo --topk 10
python 08_threshold_audit.py --dataset_dir overall_ICU --guidelines overall_ICU/config/guidelines.yaml

# Step 5: Summary statistics
python 10_build_ablation_matrix.py --dataset_dir overall_ICU
python 11_stability_summary.py --dataset_dir overall_ICU
```

---

## Script Reference

### 01_collect_hyperparams.py

Collects hyperparameters from all experiment runs.

```bash
python 01_collect_hyperparams.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | `CUSTOM` | Dataset name for labeling |

**Output:** `aggregated/hyperparams.csv`

---

### 02_collect_runtime_profile.py

Collects runtime profiling information from experiment runs.

```bash
python 02_collect_runtime_profile.py \
  --dataset_dir overall_ICU_composite_risk_score
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |

**Output:** `aggregated/runtime_profile.csv`

---

### 03_export_expressions.py

Exports top-k expressions from all methods and seeds.

```bash
# Export all methods
python 03_export_expressions.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --method all \
  --topk 10 \
  --dataset ICU

# Export specific method
python 03_export_expressions.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --method lgo \
  --topk 10

# Export multiple methods
python 03_export_expressions.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --method lgo,pysr,operon \
  --topk 10
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--method` | Yes | — | Method name, `all`, or comma-separated list |
| `--topk` | No | `10` | Number of top expressions to export |
| `--dataset` | No | `CUSTOM` | Dataset name for labeling |

**Output:** `expressions/topk_expressions.csv`, `expressions/top1_expressions.txt`

---

### 04_gen_complexity_and_stability.py

Computes complexity metrics for expressions (token count, depth, gate counts).

```bash
python 04_gen_complexity_and_stability.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU \
  --method lgo \
  --experiments lgo_hard,lgo_soft
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | Auto | Dataset name |
| `--method` | No | All | Filter by method |
| `--experiments` | No | All | Comma-separated experiment filter |

**Output:** `aggregated/complexity_by_model.csv`, `aggregated/complexity_stats.csv`

**Metrics computed:**
- `complexity`: Token count (functions, variables, constants)
- `depth`: Maximum parenthesis nesting depth
- `expr_length`: Character length of expression
- `num_gates`: Total gate count
- `num_AND/OR/soft/pair`: Gate type breakdown

---

### 05_gen_pareto_front.py

Extracts Pareto-optimal solutions (complexity vs. loss trade-off).

```bash
python 05_gen_pareto_front.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU \
  --method lgo \
  --experiments lgo_hard
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | Auto | Dataset name |
| `--method` | No | All | Filter by method |
| `--experiments` | No | All | Comma-separated experiment filter |

**Output:** `aggregated/pareto_front.csv`

---

### 06_gen_gating_usage.py

Analyzes gating mechanism usage in top-k expressions.

```bash
python 06_gen_gating_usage.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU \
  --topk 100 \
  --method lgo \
  --experiments lgo_hard,lgo_soft
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | Auto | Dataset name |
| `--topk` | No | `100` | Top-k expressions to analyze |
| `--method` | No | All | Filter by method |
| `--experiments` | No | All | Comma-separated experiment filter |

**Output:** `aggregated/gating_usage.csv`

**Statistics:**
- `n_with_gates`: Count of expressions using gates
- `prop_with_gates`: Proportion using gates
- `gates_median/q1/q3`: Gate count distribution

---

### 07_gen_thresholds_units.py

Extracts threshold values from LGO expressions and converts to physical units.

```bash
python 07_gen_thresholds_units.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU_composite_risk_score \
  --method lgo \
  --topk 10 \
  --experiments lgo_hard

# Multiple experiments
python 07_gen_thresholds_units.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU_composite_risk_score \
  --method lgo \
  --topk 10 \
  --experiments lgo_hard,lgo_soft
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | Auto | Dataset name |
| `--method` | No | `lgo` | Method to analyze |
| `--topk` | No | `10` | Top-k expressions per seed |
| `--experiments` | No | All | Comma-separated experiment filter |

**Output:** `aggregated/thresholds_units.csv`

---

### 08_convert_ground_truth_to_guidelines.py

Converts `ground_truth.json` to `guidelines.yaml` format for threshold auditing.

```bash
python 08_convert_ground_truth_to_guidelines.py \
  --ground_truth data/ICU/ground_truth.json \
  --output_dir overall_ICU/config \
  --dataset ICU \
  --verbose

# Batch conversion for multiple datasets
python 08_convert_ground_truth_to_guidelines.py \
  --ground_truth data/NHANES/ground_truth.json \
  --output_dir overall_NHANES/config \
  --dataset NHANES
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--ground_truth` | Yes | — | Path to ground_truth.json |
| `--output_dir` | Yes | — | Directory for guidelines.yaml |
| `--dataset` | Yes | — | Dataset name (used as key in YAML) |
| `--verbose` | No | False | Print detailed output |

**Output:** `config/guidelines.yaml`, `config/ground_truth_snapshot.json`

---

### 08_threshold_audit.py

Compares LGO-discovered thresholds against clinical guidelines.

```bash
python 08_threshold_audit.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU \
  --guidelines overall_ICU/config/guidelines.yaml \
  --experiments lgo_hard
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | Auto | Dataset name |
| `--guidelines` | Yes | — | Path to guidelines.yaml |
| `--experiments` | No | All | Comma-separated experiment filter |

**Output:** `aggregated/threshold_audit.csv`, `aggregated/threshold_audit_summary.csv`

**Metrics:**
- `rel_error`: Relative error vs. guideline
- `hit_10pct`: Within 10% of guideline (0/1)
- `hit_20pct`: Within 20% of guideline (0/1)

---

### 09_feature_threshold_sanity.py

Generates sanity check plots for threshold comparisons (distribution + ROC).

```bash
# ICU: Lactate (higher is riskier)
python 09_feature_threshold_sanity.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 5 \
  --feature_col lactate_mmol_l \
  --guideline 2.0 \
  --lgo_threshold 4.72 \
  --higher_is_riskier \
  --outdir sanity_plots/ICU/lactate

# eICU: GCS (lower is riskier - no --higher_is_riskier flag)
python 09_feature_threshold_sanity.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 5 \
  --feature_col gcs \
  --guideline 8 \
  --lgo_threshold 13.0 \
  --outdir sanity_plots/eICU/gcs

# NHANES: BMI (higher is riskier)
python 09_feature_threshold_sanity.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --label_col metabolic_score \
  --label_threshold 3 \
  --feature_col bmi \
  --guideline 25.0 \
  --lgo_threshold 27.5 \
  --higher_is_riskier \
  --outdir sanity_plots/NHANES/bmi
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | — | Path to data CSV |
| `--label_col` | Yes | — | Label column name |
| `--feature_col` | Yes | — | Feature column name |
| `--guideline` | Yes | — | Clinical guideline threshold |
| `--lgo_threshold` | Yes | — | LGO-discovered threshold |
| `--label_threshold` | No | None | Threshold to binarize continuous labels |
| `--label_positive_below` | No | False | If set, positive = label ≤ threshold |
| `--higher_is_riskier` | No | False | If set, higher feature values = risk |
| `--outdir` | No | `sanity_plots` | Output directory |

**Output:** `{feature}_dist.png`, `{feature}_roc.png`

---

### 10_build_ablation_matrix.py

Builds ablation study comparison table (base vs. lgo_soft vs. lgo_hard).

```bash
python 10_build_ablation_matrix.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | Auto | Dataset name |

**Output:** `aggregated/ablation_table.csv`

**Columns:**
- `base_median/IQR`: Baseline (no gates) performance
- `lgo_soft_median/IQR`: Soft gate performance
- `lgo_hard_median/IQR`: Hard gate performance

---

### 11_stability_summary.py

Generates stability summary statistics across seeds.

```bash
python 11_stability_summary.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | Auto | Dataset name |

**Output:** `aggregated/stability_summary.csv`

**Statistics per metric:** median, IQR, mean, std, n

---

## Optional Utilities

### opt_dedup_csv.py

Removes duplicate rows from CSV files.

```bash
python opt_dedup_csv.py \
  --csv aggregated/overall_metrics.csv \
  --keys method,experiment,seed \
  --keep last \
  --backup
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | — | Input CSV file |
| `--keys` | Yes | — | Comma-separated key columns |
| `--out` | No | Overwrite | Output file path |
| `--keep` | No | `last` | Keep first or last duplicate |
| `--backup` | No | False | Create .bak backup |

---

### opt_gen_calibration.py

Calibration analysis for binary classification predictions.

```bash
python opt_gen_calibration.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU \
  --method lgo \
  --experiment lgo_hard \
  --calibrator platt \
  --cv_folds 3 \
  --n_bins 10
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dir` | Yes | — | Path to dataset results directory |
| `--dataset` | No | `CUSTOM` | Dataset name |
| `--method` | No | All | Filter by method |
| `--experiment` | No | All | Filter by experiment |
| `--calibrator` | No | `none` | `none`, `platt`, `isotonic`, `both` |
| `--cv_folds` | No | `3` | CV folds for calibration |
| `--n_bins` | No | `10` | Number of reliability bins |

**Output:** `aggregated/calibration_bins.csv`, `aggregated/calibration_ece.csv`

---

## Output Files

### Summary Table

| File | Script | Description |
|------|--------|-------------|
| `hyperparams.csv` | 01 | Hyperparameters from all runs |
| `runtime_profile.csv` | 02 | Runtime statistics |
| `topk_expressions.csv` | 03 | Top-k expressions per method/seed |
| `top1_expressions.txt` | 03 | Best expression per method |
| `complexity_by_model.csv` | 04 | Per-expression complexity metrics |
| `complexity_stats.csv` | 04 | Aggregated complexity statistics |
| `pareto_front.csv` | 05 | Pareto-optimal solutions |
| `gating_usage.csv` | 06 | Gate usage statistics |
| `thresholds_units.csv` | 07 | Extracted thresholds in physical units |
| `guidelines.yaml` | 08_convert | Clinical guideline thresholds |
| `threshold_audit.csv` | 08_audit | Per-feature threshold comparison |
| `threshold_audit_summary.csv` | 08_audit | Hit rate summary |
| `ablation_table.csv` | 10 | Ablation study comparison |
| `stability_summary.csv` | 11 | Cross-seed stability statistics |
| `calibration_bins.csv` | opt_cal | Reliability diagram data |
| `calibration_ece.csv` | opt_cal | Expected Calibration Error |

---
