# LGO: Logistic-Gated Operators for Interpretable Symbolic Regression

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/) 

**Logistic-Gated Operators (LGO)** enable unit-aware, auditable thresholds in symbolic regression by treating cut-points as first-class parameters inside equations. LGO maps thresholds back to physical units for direct comparison with domain guidelines, turning interpretability from a post-hoc narrative into a modeling constraint.

## Key Features

- **Unit-aware thresholds**: Explicit cut-points in physical units (e.g.: mmHg, mmol/L, mg/dL)
- **Hard/Soft gate variants**: Crisp decision boundaries or smooth transitions
- **Parsimonious by design**: Automatic gate pruning when not warranted by data
- **Clinical alignment**: ~71% of thresholds within 10% of guideline anchors
- **Built on DEAP**: Extends proven genetic programming framework

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/oudeng/LGO.git
cd LGO
```

### 2. Environment Setup

⚠️ **Important**: Different methods require different conda environments due to dependency conflicts.

| Method | Environment | Python |
|--------|------------|--------|
| **LGO / PySR / Operon** | `py310` | 3.10 | 
| **PSTree** | `pstree` | 3.9 | 
| **RILS-ROLS** | `rils-rols` | 3.11 |

#### Setup Main Environment (LGO/PySR/Operon)
```bash
# Create main environment for LGO, PySR, and Operon
conda env create -f env_setup/environment_py310.yml
conda activate py310
```

#### Setup Baseline-Specific Environments
```bash
# PSTree (requires Python 3.9)
conda env create -f env_setup/environment_pstree.yml

# RILS-ROLS (requires Python 3.11)
conda env create -f env_setup/environment_rils-rols.yml
```

For detailed setup instructions, see [env_setup/README_env.md](env_setup/README_env.md).

## Run Sample Experiments

### LGO with PySR and Operon (Using `py310` Environment)
```bash
# Activate the main environment
conda activate py310
cd ~/LGO/

# Run LGO, PySR, and Operon experiments
python code/run_v3_8.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir results/experiments/ICU/
```

### PSTree Baseline (Using `pstree` Environment)
```bash
# Switch to PSTree environment
conda activate pstree
cd ~/LGO/

# Run PSTree experiment
python code/run_v3_8.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments pstree \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir results/experiments/ICU/
```

### RILS-ROLS Baseline (Using `rils-rols` Environment)
```bash
# Switch to RILS-ROLS environment
conda activate rils-rols
cd ~/LGO/

# Run RILS-ROLS experiment
python code/run_v3_8.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments rils_rols \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir results/experiments/ICU/
```

### Running All Experiments (Script)
```bash
#!/bin/bash
# run_all_baselines.sh

SEEDS="1,2,3,5,8,13,21,34,55,89"
CSV="data/ICU/ICU_composite_risk_score.csv"
TARGET="composite_risk_score"
TASK="regression"

# LGO, PySR, Operon
conda activate py310
python code/run_v3_8.py --csv $CSV --target $TARGET --task $TASK \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --seeds $SEEDS --test_size 0.2 --outdir results/experiments/ICU/

# PSTree
conda activate pstree
python code/run_v3_8.py --csv $CSV --target $TARGET --task $TASK \
  --experiments pstree \
  --seeds $SEEDS --test_size 0.2 --outdir results/experiments/ICU/

# RILS-ROLS
conda activate rils-rols
python code/run_v3_8.py --csv $CSV --target $TARGET --task $TASK \
  --experiments rils_rols \
  --seeds $SEEDS --test_size 0.2 --outdir results/experiments/ICU/

echo "All experiments completed!"
```

### Analyze Results
```bash
# Activate any environment with analysis dependencies
conda activate py310
cd ~/LGO/
python utility_analysis/<01 to 10>.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU_composite_risk_score

# Generate visualizations
python code/analysis/plot_results.py \
  --results_dir results/experiments/ \
  --output_dir results/figures/
```

## How It Works

LGO introduces two gating operators as symbolic regression primitives:

```python
# Hard gate: Pure threshold (sparse switching)
LGO_hard(x; a, b) = σ(a(x - b))

# Soft gate: Magnitude-preserving (graded modulation)  
LGO_soft(x; a, b) = x · σ(a(x - b))
```

Where:
- `b`: Threshold location (learned in z-score space, mapped to physical units)
- `a`: Transition steepness
- `σ`: Logistic function

## Repository Structure
```
LGO/
├── code/                       # All source code & scripts
│   ├── core/                   # Core LGO implementation
│   │   ├── LGO_v2_1.py         # LGO main function script
│   │   └── lgo_v3/             # Supporting modules
│   ├── baselines/              # PySR, Operon, PSTree, RILS-ROLS
│   ├── run_v3_8.py             # Main experiment runner
│   ├── utility_analysis/       # Analysis & Outpit aggregated results
│   ├── utility_plots/          # Visualization by analysis results
│   └── configs/                # Configuration files
│       └── guidelines.yaml     # Clinical guildlines
│
├── results/                     # All outputs [NOT IN GIT]
│   ├── experiments/             # Aggregated results
│   │   ├── ICU/                 # ICU composite risk score
│   │   ├── NHANES/              # NHANES metabolic score
│   │   └── UCI/                 # CTG, Cleveland, Hydraulic
│   └── figures/                 # Visualization results
│
├── data/                        # Datasets & preprocessing scripts
│   ├── ICU/
│   │   ├── ICU_composite_risk_score.csv
│   │   ├── mimic_extract_v7.py
│   │   └── README.md            # Data description & stats
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
├── env_setup/                   # Environment setup files
│   ├── environment_py310.yml    # LGO/PySR/Operon
│   ├── environment_pstree.yml   # PSTree (Python 3.9)
│   ├── environment_rils-rols.yml   # RILS-ROLS (Python 3.11)
│   └── README_env.md            # Detailed setup guide
│
├── .gitignore
└── README.md
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--csv` | Path to dataset CSV file | Required |
| `--target` | Target column name | Required |
| `--task` | Task type: `regression` or `classification` | Required |
| `--experiments` | Methods to run (comma-separated) | `base,lgo_soft` |
| `--seeds` | Random seeds (comma-separated) | `1,2,3` |
| `--test_size` | Test set proportion | `0.2` |
| `--outdir` | Output directory | `results/experiments/` |

### Available Methods

- `base`: Baseline genetic programming
- `lgo_soft`: LGO with soft constraints
- `lgo_hard`: LGO with hard constraints  
- `pysr`: PySR baseline (requires `py310` env)
- `operon`: Operon C++ baseline (requires `py310` env)
- `pstree`: PS-Tree baseline (requires `pstree` env)
- `rils_rols`: RILS-ROLS baseline (requires `rils-rols` env)

## Datasets & Results

We evaluate on healthcare (MIMIC-IV ICU, NHANES) and UCI benchmark datasets. LGO discovers clinically meaningful thresholds while maintaining competitive accuracy:

| Dataset | Task | Key Finding | Details |
|---------|------|-------------|---------|
| **ICU Risk Score** | Mortality Prediction | MAP threshold: 63.7 vs 65 mmHg guideline (Δ2.0%) | See paper Table 4 |
| **NHANES Metabolic** | Metabolic Syndrome | SBP threshold: 128.3 vs 130 mmHg guideline (Δ1.3%) | See paper Table 4 |
| **UCI CTG** | Fetal State Classification | Interpretable fetal monitoring rules | - |
| **UCI Cleveland** | Heart Disease Prediction | Clinically aligned risk factors | - |
| **UCI Hydraulic** | System Condition Monitoring | Clear fault detection boundaries | - |

Full results, equations, and threshold audits are in the [paper](https://arxiv.org/).

## Troubleshooting

### Environment Issues
- **PSTree not working on Python 3.10/3.11**: Use the provided `pstree` environment with Python 3.9
- **PySR/Operon numpy conflicts**: The `py310` environment includes a technical solution for version compatibility
- **RILS-ROLS installation**: Recommand Python 3.11 with the specific dependencies in `environment_rils-rols.yml`

### Running Experiments
- Always activate the correct environment before running each baseline
- Check `env_setup/README_env.md` for detailed troubleshooting steps
- Results are saved to `results/experiments/` (not tracked by git)

## Citation

```bibtex
@article{lgo2025,
  title={Logistic-Gated Operators Enable Auditable Unit-Aware Thresholds in Symbolic Regression},
  author={Deng, Ou and Cong, Ruichen and Xu, Jianting and Nishimura, Shoji and Ogihara, Atsushi and Jin, Qun},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

Built on [DEAP](https://github.com/DEAP/deap) | Comparisons with [PySR](https://github.com/MilesCranmer/PySR), [Operon](https://github.com/heal-research/pyoperon), [PS-Tree](https://github.com/hengzhe-zhang/PS-Tree), [RILS-ROLS](https://github.com/kartelj/rils-rols)
