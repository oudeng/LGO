# LGO: Logistic-Gated Operators for Interpretable Symbolic Regression

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2510.05178) 
[![ZENODO](https://zenodo.org/badge/DOI/10.5281/zenodo.17299423.svg)](https://zenodo.org/records/17299423)

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
│
│(Datasets)
├── data/                                 # Datasets & preprocessing
│   ├── ICU/
│   │   ├── ICU_composite_risk_score.csv
│   │   └── mimic_extract_v7.py
│   ├── NHANES/          
│   │   ├── NHANES_metabolic_score.csv
│   │   └── fm_XPT_toCSV_v4_3.py       
│   └── UCI/
│       ├── CTG_nsp_bin.csv
│       ├── Heart_Cleveland_num.csv
│       └── HydraulicSys_fault_score.csv
│
│(Environment setup) 
├── env_setup/                            # Environment configurations
│   ├── env_py310.yml                     # LGO/PySR/Operon
│   ├── env_pstree.yml                    # PSTree (Python 3.9)
│   ├── env_rils-rols.yml                 # RILS-ROLS (Python 3.11)
│   └── README_env.md
├── configs/                              # Configuration files
│   └── guidelines.yaml                   # Clinical guidelines
│
│(Function scripts)
├── lgo_v3/                               # Supporting modules
├── LGO_v2_1.py                           # LGO engine
├── Operon_v1.py                          # Operon engine
├── PSTree_v2_2.py                        # PSTree engine
├── RILS_ROLS_v2_1.py                     # RILS-ROLS engine
├── run_v3_8                              # Main entrance
├── run_command_lines/                    # Running commands
│
│(Results raw)
├── overall_ICU_composite_risk_score      # Results of ICU
├── overall_NHANES_metabolic_score        # Results of NHANES
├── overall_UCI_CTG_NSPbin                # Results of CTG
├── overall_UCI_Heart_Cleveland_num       # Results of Cleveland
├── overall_UCI_HydraulicSys_fault_score  # Results of Hydraulic
│
│(Results analysis)
├── utility_analysis/                     # Detailed results analysis
├── utility_plots/                        # Visualization
│
│(Smoke test)                             
├── smoke_test
│   ├── run_smoke_NHANES.sh               # NHANES as example
│   └── README_smoke_test.md
│
├── .gitignore
└── README.md
```
---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/oudeng/LGO.git
cd LGO
```

### 2. Environment Setup

⚠️**Important**: Different methods require separate conda environments due to dependency conflicts.

| Method | Environment | Python | Primary Use |
|--------|------------|--------|-------------|
| **LGO / PySR / Operon** | `py310` | 3.10 | Main experiments |
| **PSTree** | `pstree` | 3.9 | Baseline comparison |
| **RILS-ROLS** | `rils-rols` | 3.11 | Baseline comparison |

```bash
# Main environment (LGO, PySR, Operon)
conda env create -f env_setup/env_py310.yml
conda activate py310

# Additional baseline environments (if needed)
conda env create -f env_setup/env_pstree.yml
conda env create -f env_setup/env_rils-rols.yml
```

### 3. Smoke Test
Please see in [smoke_test](smoke_test/)

## Reproduce Full Results

Please see md files in [run_command_lines/](run_command_lines/)

### Analyze & Visualize Results

Please see scripts in [run_command_lines/](run_command_lines/) and [run_command_plots/](run_command_lines/).


## Parameters Reference

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--csv` | Dataset path | Required | `data/ICU/ICU_composite_risk_score.csv` |
| `--target` | Target column | Required | `composite_risk_score` |
| `--task` | Task type | Required | `regression` or `classification` |
| `--experiments` | Methods to run | `base,lgo_soft` | `lgo_soft,pysr,operon` |
| `--seeds` | Random seeds | `1,2,3` | `1,2,3,5,8,13,21,34,55,89` |
| `--test_size` | Test proportion | `0.2` | `0.3` |
| `--outdir` | Output directory | `overall_*/` | Custom path |

### Available Methods

| Method | Environment | Description |
|--------|-------------|-------------|
| `base` | `py310` | Baseline GP without gates |
| `lgo_soft` | `py310` | LGO with soft gates |
| `lgo_hard` | `py310` | LGO with hard gates |
| `pysr` | `py310` | PySR baseline |
| `operon` | `py310` | Operon C++ baseline |
| `pstree` | `pstree` | PS-Tree baseline |
| `rils_rols` | `rils-rols` | RILS-ROLS baseline |

## Citation

```bibtex
@article{deng2025lgo,
  title={Logistic-Gated Operators Enable Auditable Unit-Aware Thresholds in Symbolic Regression},
  author={Deng, Ou and Cong, Ruichen and Xu, Jianting and Nishimura, Shoji and Ogihara, Atsushi and Jin, Qun},
  journal={arXiv preprint https://arxiv.org/abs/2510.05178},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

Built on [DEAP](https://github.com/DEAP/deap) | Benchmarked against [PySR](https://github.com/MilesCranmer/PySR), [Operon](https://github.com/heal-research/pyoperon), [PS-Tree](https://github.com/hengzhe-zhang/PS-Tree), [RILS-ROLS](https://github.com/kartelj/rils-rols)
