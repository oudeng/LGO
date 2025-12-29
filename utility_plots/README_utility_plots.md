# LGO Utility Plots

Version 1.0 — December 2025

Publication-quality visualization scripts for LGO (Logistic-Gated Operators) experimental results.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Script Reference](#script-reference)
  - [01 - Performance Violin](#01_median_performance_violinpy)
  - [02 - Pareto Front](#02_pareto_frontpy)
  - [03 - Gating Usage](#03_gating_usagepy)
  - [04 - Complexity Distribution](#04_complexity_distributionpy)
  - [05 - Threshold Analysis Pipeline](#05-threshold-analysis-pipeline)
  - [06 - Stability Comparison](#06_stability_comparisonpy)
  - [07 - Ablation Heatmap](#07_ablation_heatmappy)
  - [08 - Runtime Efficiency](#08_runtime_efficiencypy)
  - [09 - Clinical Baselines](#09_plot_clinical_baselines_icu_eicupy)
- [Configuration](#configuration)
- [Paper Figure Mapping](#paper-figure-mapping)
- [Dependencies](#dependencies)

---

## Overview

This package contains visualization scripts for LGO experiments, organized into:

| Category | Scripts | Description |
|----------|---------|-------------|
| Performance & Comparison | 01-04, 06-08 | Performance, Pareto, gating, complexity plots |
| Threshold Analysis | 05_* | Clinical threshold discovery analysis pipeline |
| Clinical Baselines | 09 | ICU/eICU clinical baseline comparison |

All scripts generate both PNG (300 DPI) and PDF outputs for publication.

---

## Directory Structure

### Input Requirements

```
overall_{dataset_name}/
├── aggregated/
│   ├── overall_metrics.csv      # Performance metrics (01, 06)
│   ├── pareto_front.csv         # Pareto solutions (02)
│   ├── complexity_stats.csv     # Complexity statistics (04)
│   ├── gating_usage.csv         # Gate usage stats (03)
│   ├── stability_summary.csv    # Stability metrics (06)
│   ├── ablation_table.csv       # Ablation results (07)
│   ├── runtime_profile.csv      # Runtime data (08)
│   └── thresholds_units.csv     # Threshold values (05_*)
└── config/
    └── guidelines.yaml          # Clinical guidelines (05_*)
```

### Output Structure

```
figs/
├── performance/
│   ├── 01_violin_performance_2x3_with_box.png
│   ├── 01_violin_performance_2x3_with_box.pdf
│   └── plots/                   # Individual dataset plots
│       ├── violin_icu.png
│       ├── violin_eicu.png
│       └── ...
├── pareto/
│   ├── 02_pareto_front_2x3.png
│   ├── 02_pareto_front_2x3.pdf
│   └── plots/
├── gating/
│   ├── 03_gating_usage_2x3.png
│   └── plots/
├── complexity/
│   ├── 04_complexity_distribution_2x3.png
│   └── plots/
├── 05threshold_analysis/
│   ├── all_thresholds_summary.csv
│   ├── heatmaps/
│   ├── distributions/
│   ├── gating/
│   ├── tables/
│   └── publication/
├── stability/
│   └── 06_stability_comparison.png
├── ablation/
│   └── 07_ablation_heatmap.png
└── runtime/
    └── 08_runtime_efficiency.png
```

---

## Quick Start

### Run All Main Plots

```bash
# Define dataset roots
ROOTS="overall_ICU_composite_risk_score \
       overall_eICU_composite_risk_score \
       overall_NHANES_metabolic_score \
       overall_UCI_CTG_NSPbin \
       overall_UCI_Heart_Cleveland_num \
       overall_UCI_HydraulicSys_fault_score"

# Generate all main plots
python utility_plots/01_median_performance_violin.py --roots $ROOTS --outdir figs/performance --show_box
python utility_plots/02_pareto_front.py --roots $ROOTS --outdir figs/pareto
python utility_plots/03_gating_usage.py --roots $ROOTS --outdir figs/gating
python utility_plots/04_complexity_distribution.py --roots $ROOTS --outdir figs/complexity
python utility_plots/06_stability_comparison.py --roots $ROOTS --outdir figs/stability
python utility_plots/07_ablation_heatmap.py --roots $ROOTS --outdir figs/ablation
python utility_plots/08_runtime_efficiency.py --roots $ROOTS --outdir figs/runtime
```

### Run Threshold Analysis Pipeline

```bash
# Method 1: Direct dataset directories (recommended)
bash utility_plots/05_00_run_all_analysis.sh \
  overall_ICU_composite_risk_score \
  overall_eICU_composite_risk_score \
  overall_NHANES_metabolic_score \
  overall_UCI_CTG_NSPbin \
  overall_UCI_Heart_Cleveland_num \
  overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/05threshold_analysis

# Method 2: Using --roots parameter
bash utility_plots/05_00_run_all_analysis.sh \
  --roots overall_ICU_* overall_eICU_* overall_NHANES_* overall_UCI_* \
  --outdir utility_plots/figs/05threshold_analysis

# Method 3: Auto-discovery from current directory
bash utility_plots/05_00_run_all_analysis.sh \
  --data_root . \
  --outdir utility_plots/figs/05threshold_analysis
```

---

## Script Reference

### 01_median_performance_violin.py

Generates violin plots with box overlay showing R²/AUROC distribution across methods.

```bash
python utility_plots/01_median_performance_violin.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/performance \
  --show_box \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--roots` | Yes | — | Dataset directories (space-separated) |
| `--outdir` | No | `figs/performance` | Output directory |
| `--show_box` | No | True | Show box plot overlay |
| `--no_individual` | No | False | Skip individual plot export |
| `--dpi` | No | `300` | Output resolution |

**Output:** `01_violin_performance_2x3_with_box.png`, individual plots in `plots/`

---

### 02_pareto_front.py

Generates Pareto front scatter plots (complexity vs. CV loss trade-offs).

```bash
python utility_plots/02_pareto_front.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/pareto \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--roots` | Yes | — | Dataset directories |
| `--outdir` | No | `figs/pareto` | Output directory |
| `--no_individual` | No | False | Skip individual plot export |
| `--dpi` | No | `300` | Output resolution |

**Output:** `02_pareto_front_2x3.png`, individual plots in `plots/`

---

### 03_gating_usage.py

Visualizes LGO soft and hard gating usage patterns.

```bash
python utility_plots/03_gating_usage.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/gating \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--roots` | Yes | — | Dataset directories |
| `--outdir` | No | `figs/gating` | Output directory |
| `--no_individual` | No | False | Skip individual plot export |
| `--dpi` | No | `300` | Output resolution |

**Output:** `03_gating_usage_2x3.png`, individual plots in `plots/`

---

### 04_complexity_distribution.py

Visualizes model complexity distributions across methods.

```bash
python utility_plots/04_complexity_distribution.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/complexity \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--roots` | Yes | — | Dataset directories |
| `--outdir` | No | `figs/complexity` | Output directory |
| `--no_individual` | No | False | Skip individual plot export |
| `--dpi` | No | `300` | Output resolution |

**Output:** `04_complexity_distribution_2x3.png`, individual plots in `plots/`

---

### 05 - Threshold Analysis Pipeline

A complete pipeline for clinical threshold analysis, orchestrated by `05_00_run_all_analysis.sh`.

#### 05_00_run_all_analysis.sh (Master Script)

```bash
bash utility_plots/05_00_run_all_analysis.sh \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/05threshold_analysis
```

| Option | Description |
|--------|-------------|
| `--data_root DIR` | Root directory containing `overall_*` directories |
| `--roots DIR...` | Dataset directories to process |
| `--outdir DIR` | Output directory (default: `threshold_analysis`) |
| `--skip_aggregate` | Skip aggregation step (use existing CSV) |

#### 05_01_aggregate_thresholds.py

Aggregates threshold data from multiple datasets.

```bash
python utility_plots/05_01_aggregate_thresholds.py \
  --dataset_dirs overall_ICU_* overall_eICU_* overall_NHANES_* \
  --outdir aggregated_analysis \
  --only_anchored \
  --min_count 1
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dirs` | Yes | — | Dataset directories |
| `--outdir` | No | `aggregated_analysis` | Output directory |
| `--only_anchored` | No | False | Only features with guidelines |
| `--min_count` | No | `1` | Minimum count for inclusion |

**Output:** `all_thresholds_summary.csv`

#### 05_02_agreement_heatmap.py

Generates threshold agreement heatmaps.

```bash
python utility_plots/05_02_agreement_heatmap.py \
  --csv all_thresholds_summary.csv \
  --outdir figs/heatmaps \
  --combined \
  --annotate \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | — | Path to aggregated threshold CSV |
| `--outdir` | No | `figs/heatmaps` | Output directory |
| `--datasets` | No | All | Specific datasets to plot |
| `--combined` | No | False | Create combined multi-dataset figure |
| `--layout_2x3` | No | True | Use 2x3 layout for combined figure |
| `--annotate` | No | True | Annotate cells with values |
| `--dpi` | No | `300` | Output resolution |

**Output:** `heatmap_combined.png`, `heatmap_combined_2x3.png`, per-dataset heatmaps

#### 05_03_distribution_plot.py

Generates IQR distribution plots.

```bash
python utility_plots/05_03_distribution_plot.py \
  --csv all_thresholds_summary.csv \
  --outdir figs/distributions \
  --combined \
  --layout_2x3 \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | — | Path to aggregated threshold CSV |
| `--outdir` | No | `figs/distributions` | Output directory |
| `--datasets` | No | All | Specific datasets to plot |
| `--combined` | No | False | Create combined multi-dataset figure |
| `--layout_2x3` | No | True | Use 2x3 layout |
| `--dpi` | No | `300` | Output resolution |

**Output:** `distribution_combined.png`, `distribution_combined_2x3.png`

#### 05_04_gating_parsimony.py

Analyzes gate parsimony (soft vs. hard gating).

```bash
python utility_plots/05_04_gating_parsimony.py \
  --dataset_dirs overall_ICU_* overall_eICU_* overall_NHANES_* \
  --outdir figs/gating \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset_dirs` | Yes | — | Dataset directories |
| `--outdir` | No | `figs/gating` | Output directory |
| `--dpi` | No | `300` | Output resolution |

**Output:** `gate_usage_comparison.png`, `gate_usage_table.csv`, `gate_usage_table.tex`

#### 05_05_cross_dataset_table.py

Creates summary tables (CSV and LaTeX).

```bash
python utility_plots/05_05_cross_dataset_table.py \
  --csv all_thresholds_summary.csv \
  --outdir figs/tables
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | — | Path to aggregated threshold CSV |
| `--outdir` | No | `figs/tables` | Output directory |

**Output:** `threshold_table.csv`, `threshold_table.tex`, `summary_table.csv`, `summary_table.tex`

#### 05_06_publication_figure.py

Generates combined publication-ready Figure 2.

```bash
python utility_plots/05_06_publication_figure.py \
  --csv all_thresholds_summary.csv \
  --outdir figs/publication \
  --dpi 600 \
  --layout_2x3
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | — | Path to aggregated threshold CSV |
| `--outdir` | No | `figs/publication` | Output directory |
| `--dpi` | No | `300` | Output resolution |
| `--layout_2x3` | No | True | Generate 2x3 layout for all 6 datasets |

**Output:** `figure2_thresholds.pdf`, `figure2_thresholds_2x3.pdf`, high-res PNG versions

---

### 06_stability_comparison.py

Generates stability analysis plots (IQR across seeds).

```bash
python utility_plots/06_stability_comparison.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/stability \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--roots` | Yes | — | Dataset directories |
| `--outdir` | No | `figs/stability` | Output directory |
| `--dpi` | No | `300` | Output resolution |

**Output:** `06_stability_comparison.png`

---

### 07_ablation_heatmap.py

Generates ablation study heatmaps (base vs. soft vs. hard).

```bash
python utility_plots/07_ablation_heatmap.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/ablation \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--roots` | Yes | — | Dataset directories |
| `--outdir` | No | `figs/ablation` | Output directory |
| `--dpi` | No | `300` | Output resolution |

**Output:** `07_ablation_heatmap.png`

---

### 08_runtime_efficiency.py

Generates runtime comparison plots.

```bash
python utility_plots/08_runtime_efficiency.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/runtime \
  --dpi 300
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--roots` | Yes | — | Dataset directories |
| `--outdir` | No | `figs/runtime` | Output directory |
| `--dpi` | No | `300` | Output resolution |

**Output:** `08_runtime_efficiency.png`

---

### 09_plot_clinical_baselines_icu_eicu.py

Compares LGO, AutoScore, and EBM on ICU/eICU datasets.

```bash
python utility_plots/09_plot_clinical_baselines_icu_eicu.py
```

This script auto-discovers result files from:
- `LGO_AutoScore_v3_8/ICU_results_*/`
- `LGO_AutoScore_v3_8/eICU_results_*/`
- `LGO_Interpret_v1_4/*/`

**Output:** `figs/clinical_baselines_icu_eicu.png`

---

## Configuration

### Method Display Names

```python
METHOD_DISPLAY = {
    "pysr": "PySR",
    "pstree": "PSTree", 
    "rils_rols": "RILS-ROLS",
    "operon": "Operon",
    "lgo_base": r"LGO$_\mathrm{base}$",
    "lgo_soft": r"LGO$_\mathrm{soft}$",
    "lgo_hard": r"LGO$_\mathrm{hard}$"
}
```

### Color Palette

```python
METHOD_COLORS = {
    "pysr": "#2166ac",      # blue
    "pstree": "#762a83",    # purple
    "rils_rols": "#92c5de", # light blue
    "operon": "#5aae61",    # green
    "lgo_base": "#f4a582",  # light orange
    "lgo_soft": "#d6604d",  # red-orange
    "lgo_hard": "#b2182b"   # dark red
}
```

### Dataset Titles

```python
TITLES = {
    "ICU": "MIMIC-IV ICU",
    "eICU": "eICU",
    "NHANES": "NHANES",
    "CTG": "UCI CTG",
    "Cleveland": "UCI Cleveland",
    "Hydraulic": "UCI Hydraulic"
}
```

---

## Paper Figure Mapping

| Paper Element | Script(s) | Output File |
|--------------|-----------|-------------|
| Figure 1 (Performance) | 01 | `violin_performance_2x3_with_box.pdf` |
| Figure 2 (Thresholds) | 05_06 | `figure2_thresholds.pdf` |
| Figure 3 (Pareto) | 02 | `pareto_front_2x3.pdf` |
| Figure 4 (Gating) | 03, 05_04 | `gating_usage_2x3.pdf` |
| Table 3 (Complexity) | 04 | `complexity_distribution_2x3.pdf` |
| Table 4 (Ablation) | 07 | `ablation_heatmap.pdf` |
| SI (Runtime) | 08 | `runtime_efficiency.pdf` |
| SI (Stability) | 06 | `stability_comparison.pdf` |
| SI (Clinical) | 09 | `clinical_baselines_icu_eicu.pdf` |
| Threshold Table | 05_05 | `threshold_table.tex` |
| Summary Table | 05_05 | `summary_table.tex` |

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
```

---
