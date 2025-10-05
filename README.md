# LGO: Logistic-Gated Operators for Interpretable Symbolic Regression

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/) 

**Logistic-Gated Operators (LGO)** enable unit-aware, auditable thresholds in symbolic regression by treating cut-points as first-class parameters inside equations. LGO maps thresholds back to physical units for direct comparison with domain guidelines, turning interpretability from a post-hoc narrative into a modeling constraint.

## 🎯 Key Features

- **Unit-aware thresholds**: Explicit cut-points in physical units (e.g., mmHg, mmol/L, mg/dL)
- **Hard/Soft gate variants**: Crisp decision boundaries or smooth transitions
- **Parsimonious by design**: Automatic gate pruning when not warranted by data
- **Clinical alignment**: ~71% of thresholds within 10% of guideline anchors
- **Built on DEAP**: Extends proven genetic programming framework

## 🔬 How It Works

LGO introduces two gating operators as symbolic regression primitives:

```python
# Hard gate: Pure threshold (sparse switching)
LGO_hard(x; a, b) = σ(a(x - b))

# Soft gate: Magnitude-preserving (graded modulation)  
LGO_soft(x; a, b) = x · σ(a(x - b))
```

Where:
- `b`: Threshold location (learned in z-score space, mapped to physical units)
- `a`: Transition steepness (controls sharpness of transition)
- `σ`: Logistic sigmoid function


## 📁 Repository Structure
```
LGO/
├── code/                        # All source code & scripts
│   ├── core/                    # Core LGO implementation
│   │   ├── LGO_v2_1.py          # Main LGO engine
│   │   └── lgo_v3/              # Supporting modules
│   ├── baselines/               # PySR, Operon, PSTree, RILS-ROLS
│   ├── run_v3_8.py              # Main experiment runner
│   ├── utility_analysis/        # Results aggregation & analysis
│   ├── utility_plots/           # Visualization scripts
│   └── configs/                 # Configuration files
│       └── guidelines.yaml      # Clinical guidelines
│
├── results/                     # All outputs [NOT IN GIT]
│   ├── experiments/             # Raw experimental results
│   │   ├── ICU/                 # ICU composite risk score
│   │   ├── NHANES/              # NHANES metabolic score
│   │   └── UCI/                 # CTG, Cleveland, Hydraulic
│   └── figures/                 # Generated visualizations
│
├── data/                        # Datasets & preprocessing
│   ├── ICU/
│   │   ├── ICU_composite_risk_score.csv
│   │   ├── mimic_extract_v7.py
│   │   └── README.md
│   ├── NHANES/          
│   │   ├── NHANES_metabolic_score.csv
│   │   ├── fm_XPT_toCSV_v4_3.py
│   │   └── README.md        
│   └── UCI/
│       ├── CTG.csv
│       ├── Cleveland.csv
│       ├── Hydraulic.csv
│       └── prepare_uci.py
│
├── env_setup/                   # Environment configurations
│   ├── environment_py310.yml   # LGO/PySR/Operon
│   ├── environment_pstree.yml  # PSTree (Python 3.9)
│   ├── environment_rils-rols.yml # RILS-ROLS (Python 3.11)
│   └── README_env.md
│
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/oudeng/LGO.git
cd LGO
```

### 2. Environment Setup

⚠️ **Important**: Different methods require separate conda environments due to dependency conflicts.

| Method | Environment | Python | Primary Use |
|--------|------------|--------|-------------|
| **LGO / PySR / Operon** | `py310` | 3.10 | Main experiments |
| **PSTree** | `pstree` | 3.9 | Baseline comparison |
| **RILS-ROLS** | `rils-rols` | 3.11 | Baseline comparison |

```bash
# Main environment (LGO, PySR, Operon)
conda env create -f env_setup/environment_py310.yml
conda activate py310

# Additional baseline environments (if needed)
conda env create -f env_setup/environment_pstree.yml
conda env create -f env_setup/environment_rils-rols.yml
```

### 3. Run Sample Experiment

```bash
# Quick test with LGO
conda activate py310
python code/run_v3_8.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments lgo_soft \
  --seeds 1,2,3 \
  --test_size 0.2 \
  --outdir results/experiments/ICU/
```

## 📈 Reproduce Full Results

### Run Complete Experiments

#### LGO with Main Baselines
```bash
conda activate py310
python code/run_v3_8.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir results/experiments/ICU/
```

#### Additional Baselines
```bash
# PSTree (requires different environment)
conda activate pstree
python code/run_v3_8.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments pstree \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir results/experiments/ICU/

# RILS-ROLS
conda activate rils-rols
python code/run_v3_8.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments rils_rols \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir results/experiments/ICU/
```

### Analyze & Visualize Results

```bash
conda activate py310
cd ~/LGO/
```

#### Aggregate Results (run scripts 01-07 in sequence)
```bash
# 01. Collect hyperparameters
python code/utility_analysis/01_collect_hyperparams.py \
  --dataset_dir results/experiments/ICU \
  --dataset ICU_composite_risk_score

# 02. Export expressions and complexity
python code/utility_analysis/02_export_expressions.py \
  --dataset_dir results/experiments/ICU \
  --method all \
  --dataset ICU_composite_risk_score \
  --topk 10

# 03. Generate complexity and stability metrics
python code/utility_analysis/03_gen_complexity_and_stability.py \
  --dataset_dir results/experiments/ICU \
  --method all \
  --dataset ICU_composite_risk_score

# 04. Generate Pareto front
python code/utility_analysis/04_gen_pareto_front.py \
  --dataset_dir results/experiments/ICU \
  --method all \
  --dataset ICU_composite_risk_score

# 05. LGO specific: gate usage frequency
python code/utility_analysis/05_gen_gating_usage.py \
  --dataset_dir results/experiments/ICU \
  --method lgo \
  --dataset ICU_composite_risk_score 

# 06. Build ablation matrix
python code/utility_analysis/06_build_ablation_matrix.py \
  --dataset_dir results/experiments/ICU \
  --dataset ICU_composite_risk_score

# 07. Stability summary
python code/utility_analysis/07_stability_summary.py \
  --dataset_dir results/experiments/ICU \
  --dataset ICU_composite_risk_score
```

#### Generate Figures (run scripts 01-04 in sequence)
```bash
# 01. Performance comparison (violin plots)
python code/utility_plots/01_median_performance_violin.py \
  --roots results/experiments/ICU \
          results/experiments/NHANES \
          results/experiments/UCI/CTG \
          results/experiments/UCI/Cleveland \
          results/experiments/UCI/Hydraulic

# 02. Pareto front visualization
python code/utility_plots/02_pareto_front.py \
  --roots results/experiments/ICU \
          results/experiments/NHANES \
          results/experiments/UCI/CTG \
          results/experiments/UCI/Cleveland \
          results/experiments/UCI/Hydraulic

# 03. Gating usage analysis (LGO specific)
python code/utility_plots/03_gating_usage.py \
  --roots results/experiments/ICU \
          results/experiments/NHANES \
          results/experiments/UCI/CTG \
          results/experiments/UCI/Cleveland \
          results/experiments/UCI/Hydraulic

# 04. Threshold analysis and clinical alignment
python code/utility_plots/04_thresholds.py \
  --dataset_dirs data/ICU data/NHANES \
  --result_dirs results/experiments/ICU results/experiments/NHANES \
  --config_dir code/configs \
  --method lgo \
  --experiment lgo_hard \
  --only_anchored \
  --topk 5 \
  --annotate \
  --outdir results/figures/thresholds

# 04b. Generate threshold comparison plot
python code/utility_plots/04_thresholds_plot.py \
  --csv results/figures/thresholds/thresholds_summary.csv \
  --outdir results/figures/
```

### Automated Pipeline

Save as `run_all_experiments.sh`:
```bash
#!/bin/bash
# Complete experimental pipeline

SEEDS="1,2,3,5,8,13,21,34,55,89"

# Function to run experiment
run_experiment() {
    local env=$1
    local experiments=$2
    local dataset_path=$3
    local dataset_name=$4
    local target=$5
    local task=$6
    
    conda activate $env
    python code/run_v3_8.py \
        --csv data/${dataset_path}/${dataset_name}.csv \
        --target $target \
        --task $task \
        --experiments $experiments \
        --seeds $SEEDS \
        --test_size 0.2 \
        --outdir results/experiments/${dataset_path}/
}

# ICU experiments
run_experiment py310 "base,lgo_soft,lgo_hard,pysr,operon" ICU ICU_composite_risk_score composite_risk_score regression
run_experiment pstree "pstree" ICU ICU_composite_risk_score composite_risk_score regression  
run_experiment rils-rols "rils_rols" ICU ICU_composite_risk_score composite_risk_score regression

# NHANES experiments
run_experiment py310 "base,lgo_soft,lgo_hard,pysr,operon" NHANES NHANES_metabolic_score metabolic_score classification
run_experiment pstree "pstree" NHANES NHANES_metabolic_score metabolic_score classification
run_experiment rils-rols "rils_rols" NHANES NHANES_metabolic_score metabolic_score classification

# UCI experiments
for dataset in CTG Cleveland Hydraulic; do
    run_experiment py310 "base,lgo_soft,lgo_hard,pysr,operon" UCI/${dataset} ${dataset} label classification
    run_experiment pstree "pstree" UCI/${dataset} ${dataset} label classification
    run_experiment rils-rols "rils_rols" UCI/${dataset} ${dataset} label classification
done

echo "✓ All experiments completed!"
```


## ⚙️ Parameters Reference

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--csv` | Dataset path | Required | `data/ICU/ICU_composite_risk_score.csv` |
| `--target` | Target column | Required | `composite_risk_score` |
| `--task` | Task type | Required | `regression` or `classification` |
| `--experiments` | Methods to run | `base,lgo_soft` | `lgo_soft,pysr,operon` |
| `--seeds` | Random seeds | `1,2,3` | `1,2,3,5,8,13,21,34,55,89` |
| `--test_size` | Test proportion | `0.2` | `0.3` |
| `--outdir` | Output directory | `results/experiments/` | Custom path |

### Available Methods

| Method | Environment | Description |
|--------|-------------|-------------|
| `base` | `py310` | Baseline GP without gates |
| `lgo_soft` | `py310` | LGO with soft gates (recommended) |
| `lgo_hard` | `py310` | LGO with hard gates |
| `pysr` | `py310` | PySR baseline |
| `operon` | `py310` | Operon C++ baseline |
| `pstree` | `pstree` | PS-Tree baseline |
| `rils_rols` | `rils-rols` | RILS-ROLS baseline |

## 🐛 Troubleshooting

### Environment Issues
- **PSTree fails on Python 3.10+**: Must use `pstree` environment with Python 3.9
- **NumPy conflicts**: The `py310` environment resolves PySR/Operon compatibility
- **RILS-ROLS installation**: Requires Python 3.11 and specific dependencies

### Common Solutions
```bash
# Reset environment
conda deactivate
conda env remove -n <env_name>
conda env create -f env_setup/environment_<env_name>.yml

# Verify installation
conda activate <env_name>
python -c "import <package_name>"
```

## 📝 Citation

```bibtex
@article{lgo2025,
  title={Logistic-Gated Operators Enable Auditable Unit-Aware Thresholds in Symbolic Regression},
  author={Deng, Ou and Cong, Ruichen and Xu, Jianting and Nishimura, Shoji and Ogihara, Atsushi and Jin, Qun},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built on [DEAP](https://github.com/DEAP/deap) | Benchmarked against [PySR](https://github.com/MilesCranmer/PySR), [Operon](https://github.com/heal-research/pyoperon), [PS-Tree](https://github.com/hengzhe-zhang/PS-Tree), [RILS-ROLS](https://github.com/kartelj/rils-rols)
