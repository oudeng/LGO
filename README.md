# LGO: Logistic-Gated Operators for Interpretable Symbolic Regression

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2510.05178) 
[![ZENODO](https://zenodo.org/badge/DOI/10.5281/zenodo.18117379.svg)](https://zenodo.org/records/18117379)

**Logistic-Gated Operators (LGO)** enable unit-aware, auditable thresholds in symbolic regression by treating cut-points as first-class parameters inside equations. LGO maps thresholds back to physical units for direct comparison with domain guidelines, turning interpretability from a post-hoc narrative into a modeling constraint.

---

## Key Features

- **Unit-aware thresholds**: Explicit cut-points in physical units (mmHg, mmol/L, mg/dL)
- **Hard/Soft gate variants**: Crisp decision boundaries or smooth transitions
- **Parsimonious by design**: Automatic gate pruning when not warranted by data
- **Clinical alignment**: Analyzing thresholds from data and auditing them with guideline anchors
- **Built on DEAP**: Extends proven genetic programming framework

## How It Works

LGO introduces two gating operators as symbolic regression primitives:

```python
# Hard gate: Pure threshold (sparse switching)
LGO_hard(x; a, b) = σ(a(x - b))

# Soft gate: Magnitude-preserving (graded modulation)  
LGO_soft(x; a, b) = x · σ(a(x - b))
```

Where `b` is the threshold location (learned in z-score space, mapped to physical units), `a` is the transition steepness, and `σ` is the logistic sigmoid function.

![Graphical abstract](https://github.com/oudeng/LGO/blob/main/Graphical_Abstract.png)

---

## Quick Start

### Clone & Run Smoke Test (3 Commands)

```bash
git clone https://github.com/oudeng/LGO.git && cd LGO
chmod +x smoke_test/run_smoke_test.sh
bash smoke_test/run_smoke_test.sh
```

**Expected time: ~10 minutes** on a standard laptop (no GPU required).

The smoke test will:
1. Set up a minimal conda environment
2. Run LGO on the NHANES metabolic syndrome dataset
3. Extract clinical thresholds and audit against guidelines
4. Generate visualization outputs

Results appear in `smoke_test/results/`. See [smoke_test/README_smoke_test.md](smoke_test/README_smoke_test.md) for details.

---

## Repository Structure

```
LGO/
├── data/                          # Datasets (ICU, NHANES, UCI)
│   └── README_data.md
├── env_setup/                     # Conda environments
│   └── README_env.md
│
├── exp_engins/                    # Experiment engines
│   ├── LGO_v2_1.py                # LGO (proposed method)
│   ├── PySR_v2.py                 # PySR baseline
│   ├── Operon_v2.py               # Operon baseline
│   ├── PSTree_v3.py               # PS-Tree baseline
│   ├── RILS_ROLS_v2_1.py          # RILS-ROLS baseline
│   ├── AutoScore_v2.py            # Clinical baseline
│   ├── InterpretML_v1.py          # EBM baseline
│   └── README_engins.md
│
├── run_v3_8_2.py                  # Main experiment runner
├── exp_CLs/                       # Reproduction command lines
│
├── exp_results
│   ├── overall_*/                 # Experiment results (6 datasets + SRBench strictly aligned on ICU)
│   ├── LGO_AutoScore_v3_8         # LGO vs AutoScore baseline
│   └── LGO_Interpret_v1_4         # LGO vs EBM baseline
│
├── utility_analysis/              # Result aggregation & analysis
│   └── README_utility_analysis.md
├── utility_plots/                 # Visualization scripts
│   └── README_utility_plots.md
│
├── smoke_test/                    # Quick verification
│   ├── run_smoke_test.sh
│   └── README_smoke_test.md
│
└── README.md                      # This file
```

---

## Environment Setup

Different methods require separate conda environments due to dependency conflicts:

| Environment | Python | Methods | Setup |
|-------------|--------|---------|-------|
| `py310` | 3.10 | LGO, PySR, Operon | `conda env create -f env_setup/env_py310.yml` |
| `pstree` | 3.9 | PS-Tree | `conda env create -f env_setup/env_pstree.yml` |
| `rils-rols` | 3.11 | RILS-ROLS | `conda env create -f env_setup/env_rils-rols.yml` |

For smoke test only, use the minimal environment:
```bash
conda env create -f env_setup/env_py310_smoke.yml
```

See [env_setup/README_env.md](env_setup/README_env.md) for detailed instructions.

---

## Datasets

| Dataset | Domain | Task | Target | N | Features |
|---------|--------|------|--------|---|----------|
| MIMIC-IV ICU | Healthcare | Regression | Composite risk score | 23,944 | 12 |
| eICU | Healthcare | Regression | Composite risk score | 21,535 | 12 |
| NHANES | Healthcare | Regression | Metabolic score | 2,548 | 7 |
| UCI CTG | Healthcare | Classification | Fetal status | 2,126 | 21 |
| UCI Cleveland | Healthcare | Classification | Heart disease | 303 | 13 |
| UCI Hydraulic | Industrial | Regression | Fault score | 2,205 | 17 |

See [data/README_data.md](data/README_data.md) for preprocessing and clinical thresholds.

---

## Reproduce Full Results

### Run Experiments

```bash
conda activate py310

python run_v3_8_2.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --task regression \
  --experiments base,lgo_soft,lgo_hard \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --outdir overall_NHANES_metabolic_score
```

See [run_command_lines/](run_command_lines/) for all dataset commands.

### Analyze Results

```bash
# Aggregate metrics and thresholds
python utility_analysis/01_collect_hyperparams.py --roots overall_*
python utility_analysis/07_gen_thresholds_units.py --dataset_dir overall_NHANES_metabolic_score

# Generate visualizations
python utility_plots/01_median_performance_violin.py --roots overall_* --outdir figs
```

See [utility_analysis/README_utility_analysis.md](utility_analysis/README_utility_analysis.md) and [utility_plots/README_utility_plots.md](utility_plots/README_utility_plots.md) for complete pipelines.

---

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--experiments` | Methods to run | `base,lgo_soft,lgo_hard` |
| `--seeds` | Random seeds (Fibonacci) | `1,2,3,5,8,13,21,34,55,89` |
| `--hparams_json` | Hyperparameters | See defaults in engine |
| `--unit_map_json` | Feature → unit mapping | Dataset-specific |

### Available Methods

| Method | Description | Environment |
|--------|-------------|-------------|
| `base` | GP without gates | py310 |
| `lgo_soft` | LGO with soft gates | py310 |
| `lgo_hard` | LGO with hard gates | py310 |
| `pysr` | PySR (Julia) | py310 |
| `operon` | Operon (C++) | py310 |
| `pstree` | PS-Tree | pstree |
| `rils_rols` | RILS-ROLS | rils-rols |

---

## Citation
Due to major revision, the following preprint will be updated soon!
```bibtex
@article{deng2025lgo,
  title={Logistic-Gated Operators Enable Auditable Unit-Aware Thresholds in Symbolic Regression},
  author={Deng, Ou and Cong, Ruichen and Xu, Jianting and Nishimura, Shoji and Ogihara, Atsushi and Jin, Qun},
  journal={arXiv preprint arXiv:2510.05178},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

Built on [DEAP](https://github.com/DEAP/deap) | Benchmarked against [PySR](https://github.com/MilesCranmer/PySR), [Operon](https://github.com/heal-research/pyoperon), [PS-Tree](https://github.com/hengzhe-zhang/PS-Tree), [RILS-ROLS](https://github.com/kartelj/rils-rols), [AutoScore](https://github.com/nliulab/AutoScore), [InterpretML](https://github.com/interpretml/interpret)
