# LGO Smoke Test

Quick reproducibility verification for reviewers and readers.

Version 1.2 — January 2026

---

## Table of Contents

- [Overview](#overview)
- [Quick Start (3 Commands)](#quick-start-3-commands)
- [What Gets Tested](#what-gets-tested)
- [Expected Results](#expected-results)
- [Detailed Steps](#detailed-steps)
- [Alternative: NHANES Dataset](#alternative-nhanes-dataset)
- [Troubleshooting](#troubleshooting)

---

## Overview

This smoke test provides a **minimal end-to-end verification** of the LGO framework:

| Component | What's Tested | Time |
|-----------|---------------|------|
| Environment | Conda setup, dependencies | ~3 min |
| LGO Engine | Symbolic regression with soft/hard gating | ~5 min |
| Threshold Analysis | Clinical threshold extraction & audit | ~1 min |
| Visualization | Publication-quality figures | ~1 min |

**Total time: ~10 minutes** on a standard laptop (no GPU required).

---

## Quick Start (3 Commands)

From any directory with conda available:

```bash
# 1. Clone repository
git clone https://github.com/oudeng/LGO.git && cd LGO

# 2. Make script executable
chmod +x smoke_test/run_smoke_test.sh

# 3. Run smoke test
bash smoke_test/run_smoke_test.sh
```

That's it! Results will appear in `smoke_test/results/`.

---

## What Gets Tested

### 1. LGO Symbolic Regression

Runs `run_v3_8_2.py` on NHANES metabolic syndrome dataset:
- **Task**: Regression (predicting metabolic_score)
- **Experiments**: `base`, `lgo_soft`, `lgo_hard`
- **Seeds**: 1, 2, 3 (for reproducibility check)
- **Output**: Candidate expressions, predictions, metrics

### 2. Threshold Extraction

Extracts clinical thresholds from LGO-discovered expressions:
- Converts normalized thresholds to physical units (mg/dL, mmHg, etc.)
- Compares against established clinical guidelines

### 3. Threshold Audit

Validates discovered thresholds against medical literature (uses existing `exp_results/.../config/guidelines.yaml`):
- **Fasting glucose**: 100 mg/dL (ADA prediabetes cutoff)
- **Triglycerides**: 150 mg/dL (ATP-III metabolic syndrome criterion)
- **Systolic BP**: 130 mmHg (hypertension stage 1)
- **HDL cholesterol**: 40/50 mg/dL (M/F, metabolic syndrome criterion)
- **Waist circumference**: 102/88 cm (M/F, central obesity)

### 4. Visualization

Generates publication-quality figures showing:
- LGO-discovered thresholds vs. clinical guidelines
- Threshold agreement heatmaps
- Distribution of discovered values across seeds

---

## Expected Results

After successful completion, check `smoke_test/results/`:

```
smoke_test/results/
├── NHANES_metabolic_score/
│   ├── aggregated/
│   │   ├── overall_metrics.csv      # R², RMSE per experiment
│   │   ├── thresholds_units.csv     # Thresholds in physical units
│   │   └── threshold_audit.csv      # Comparison vs. guidelines
│   ├── candidates/
│   │   └── candidates_lgo_*.csv     # Top expressions per seed
│   └── predictions/
│       └── test_predictions_*.csv   # Test set predictions
├── figs/
│   ├── threshold_comparison.png     # Visual comparison plot
│   └── *.pdf                        # Publication-quality figures
└── smoke_test.log                   # Full execution log
```

### Key Metrics to Verify

| Metric | Expected Range | File |
|--------|---------------|------|
| R² (lgo_hard) | 0.50 - 0.75 | `overall_metrics.csv` |
| Threshold hit rate (±20%) | ≥ 60% | `threshold_audit.csv` |
| Expressions discovered | ≥ 10 per seed | `candidates/` |

### Sample Output

From `threshold_audit.csv` (example):

| Feature | LGO Threshold | Guideline | Rel. Error | Hit 20% |
|---------|---------------|-----------|------------|---------|
| fasting_glucose | 98.5 | 100.0 | 1.5% | ✓ |
| triglycerides | 142.3 | 150.0 | 5.1% | ✓ |
| systolic_bp | 128.7 | 130.0 | 1.0% | ✓ |

---

## Detailed Steps

If you prefer to run steps manually:

### Step 1: Environment Setup

```bash
# Create conda environment
conda env create -f smoke_test/env_py310_smoke.yml

# Activate environment
conda activate py310_smoke

# Set PYTHONPATH (important! - engines are in exp_engins/)
export PYTHONPATH="$(pwd)/exp_engins:$(pwd):$PYTHONPATH"
```

### Step 2: Run LGO

```bash
python run_v3_8_2.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --task regression \
  --experiments base,lgo_soft,lgo_hard \
  --seeds 1,2,3 \
  --test_size 0.2 \
  --outdir smoke_test/results/NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --save_predictions \
  --hparams_json '{
    "gate_expr_enable": true,
    "pop_size": 1000,
    "ngen": 100,
    "local_opt_steps": 150,
    "micro_mutation_prob": 0.2,
    "cv_proxy_weight": 0.15,
    "cv_proxy_weight_final": 0.3,
    "cv_proxy_warmup_frac": 0.7,
    "cv_proxy_subsample": 0.3,
    "cv_proxy_folds": 2,
    "typed_mode": "light",
    "typed_grouping": "none",
    "include_lgo_multi": true,
    "include_lgo_and3": true,
    "include_lgo_pair": false
  }' \
  --unit_map_json '{
    "systolic_bp": "mmHg",
    "triglycerides": "mg/dL",
    "waist_circumference": "cm",
    "fasting_glucose": "mg/dL",
    "hdl_cholesterol": "mg/dL",
    "age": "years",
    "bmi": "kg/m²"
  }'
```

### Step 3: Extract Thresholds

```bash
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir smoke_test/results/NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --method lgo \
  --topk 10 \
  --experiments lgo_hard
```

### Step 4: Audit Against Guidelines

```bash
python utility_analysis/08_threshold_audit.py \
  --dataset_dir smoke_test/results/NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --guidelines exp_results/overall_NHANES_metabolic_score/config/guidelines.yaml
```

### Step 5: Generate Visualizations

```bash
# Aggregate thresholds
python utility_plots/05_01_aggregate_thresholds.py \
  --dataset_dirs smoke_test/results/NHANES_metabolic_score \
  --outdir smoke_test/results/figs

# Generate heatmap
python utility_plots/05_02_agreement_heatmap.py \
  --csv smoke_test/results/figs/all_thresholds_summary.csv \
  --outdir smoke_test/results/figs \
  --annotate
```

---

## Alternative: eICU Dataset

For ICU mortality risk prediction (larger dataset, ~20 min runtime):

```bash
bash smoke_test/run_smoke_test.sh --dataset eICU
```

Or manually:

```bash
python run_v3_8_2.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments base,lgo_soft,lgo_hard \
  --seeds 1,2,3 \
  --test_size 0.2 \
  --outdir smoke_test/results/eICU_composite_risk_score \
  --dataset eICU_composite_risk_score \
  --save_predictions \
  --hparams_json '{
    "gate_expr_enable": true,
    "pop_size": 1000,
    "ngen": 100,
    "local_opt_steps": 150,
    "micro_mutation_prob": 0.2,
    "cv_proxy_weight": 0.15,
    "cv_proxy_weight_final": 0.3,
    "cv_proxy_warmup_frac": 0.7,
    "cv_proxy_subsample": 0.3,
    "cv_proxy_folds": 2,
    "typed_mode": "light",
    "typed_grouping": "none",
    "include_lgo_multi": true,
    "include_lgo_and3": true,
    "include_lgo_pair": false
  }' \
  --unit_map_json '{
    "map_mmhg": "mmHg",
    "sbp_min": "mmHg",
    "dbp_min": "mmHg",
    "lactate_mmol_l": "mmol/L",
    "creatinine_mg_dl": "mg/dL",
    "hemoglobin_min": "g/dL",
    "sodium_min": "mmol/L",
    "age_years": "years",
    "hr_max": "bpm",
    "resprate_max": "/min",
    "spo2_min": "%",
    "gcs": "",
    "urine_output_min": "mL"
  }'
```

**eICU Expected Thresholds:**

| Feature | LGO (expected) | Clinical Guideline |
|---------|----------------|-------------------|
| Lactate | 1.0 - 2.5 mmol/L | 2.0 mmol/L (sepsis) |
| SpO₂ | 92 - 95% | 94% (hypoxemia) |
| GCS | 7 - 9 | 8 (intubation threshold) |
| Creatinine | 0.8 - 1.5 mg/dL | 1.2 mg/dL (AKI) |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `conda not found` | Install Miniconda: https://docs.conda.io/en/latest/miniconda.html |
| `DEAP import error` | Run: `pip install deap>=1.3.1` |
| `lgo_v3 not importable` | Set `PYTHONPATH`: `export PYTHONPATH="$(pwd)/exp_engins:$PYTHONPATH"` |
| `lgo engine not importable` | Ensure `exp_engins/LGO_v2_1.py` exists; run with `LGO_DEBUG=1` for details |
| `run_v3_8_2.py not found` | Run script from repository root: `cd LGO` |
| `Out of memory` | Use `--quick` flag or reduce `pop_size` |
| `Permission denied` | Run: `chmod +x smoke_test/run_smoke_test.sh` |

### Verify Environment

```bash
# Check Python version (should be 3.10.x)
python --version

# Check DEAP installation
python -c "from deap import gp; print('DEAP OK')"

# Check lgo_v3 module (from repository root)
export PYTHONPATH="$(pwd)/exp_engins:$PYTHONPATH"
python -c "import lgo_v3; print('lgo_v3 OK')"

# Check LGO engine with debug output
LGO_DEBUG=1 python -c "from LGO_v2_1 import run_lgo_sr_v2; print('LGO engine OK')"

# Check pandas
python -c "import pandas; print(f'Pandas {pandas.__version__}')"
```

### Clean Restart

```bash
# Remove existing environment
conda env remove -n py310_smoke

# Remove previous results
rm -rf smoke_test/results/

# Re-run
bash smoke_test/run_smoke_test.sh
```

### Quick Test Mode

For faster testing (~3 minutes):

```bash
bash smoke_test/run_smoke_test.sh --quick
```

### Minimal Test (No Visualization)

If visualization scripts fail, core functionality can still be verified:

```bash
# Set flag to skip visualization
SKIP_VIZ=1 bash smoke_test/run_smoke_test.sh
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Disk | 2 GB | 5 GB |
| CPU | 2 cores | 4+ cores |
| OS | Linux, macOS, Windows (WSL2) | Linux/macOS |
| Time | 10 min | 5-10 min |

No GPU required.

---

## Contact

If the smoke test fails after following troubleshooting steps, please open an issue at:
https://github.com/oudeng/LGO/issues

Include:
1. Full error message
2. Output of `conda list`
3. Operating system and version
