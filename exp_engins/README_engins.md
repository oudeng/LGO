# LGO Experiment Engines

Version 1.0 — December 2025

Symbolic regression and interpretable machine learning engines used in LGO experiments.

---

## Table of Contents

- [Overview](#overview)
- [Engine Summary](#engine-summary)
- [Directory Structure](#directory-structure)
- [Engine Reference](#engine-reference)
  - [LGO_v2_1.py](#lgo_v2_1py)
  - [PySR_v2.py](#pysr_v2py)
  - [Operon_v2.py](#operon_v2py)
  - [PSTree_v3.py](#pstree_v3py)
  - [RILS_ROLS_v2_1.py](#rils_rols_v2_1py)
  - [AutoScore_v2.py](#autoscore_v2py)
  - [InterpretML_v1.py](#interpretml_v1py)
- [Environment Requirements](#environment-requirements)
- [References](#references)

---

## Overview

This package contains wrapper implementations for symbolic regression and interpretable ML methods, providing a unified interface for fair comparison in the LGO benchmark study.

| Category | Engines | Description |
|----------|---------|-------------|
| Proposed Method | LGO | Logistic-Gated Operators with soft/hard gating |
| SR Baselines | PySR, Operon, PSTree, RILS-ROLS | State-of-the-art symbolic regression methods |
| Clinical Baselines | AutoScore, InterpretML (EBM) | Interpretable models for healthcare |

All engines follow SRBench-aligned configurations where applicable (La Cava et al., NeurIPS 2021).

---

## Engine Summary

| Engine | Version | Method | Task | Framework | Key Features |
|--------|---------|--------|------|-----------|--------------|
| LGO_v2_1.py | v2.1 | LGO | Regression | DEAP | Soft/hard gating, typed GP |
| PySR_v2.py | v2.0 | PySR | Regression | Julia | Multi-island GP, parsimony |
| Operon_v2.py | v2.0 | Operon | Regression | C++ | LM local search, fast |
| PSTree_v3.py | v3.0 | PS-Tree | Regression | Python | Piecewise symbolic trees |
| RILS_ROLS_v2_1.py | v2.1 | RILS-ROLS | Regression | Python | Iterated local search |
| AutoScore_v2.py | v2.0 | AutoScore | Classification | Python | Clinical scoring, binning |
| InterpretML_v1.py | v1.0 | EBM | Both | Python | GAM + boosting |

---

## Directory Structure

```
exp_engins/
├── LGO_v2_1.py           # LGO engine (proposed method)
├── PySR_v2.py            # PySR baseline
├── Operon_v2.py          # Operon baseline
├── PSTree_v3.py          # PS-Tree baseline
├── RILS_ROLS_v2_1.py     # RILS-ROLS baseline
├── AutoScore_v2.py       # AutoScore (clinical baseline)
└── InterpretML_v1.py     # EBM (clinical baseline)
```

---

## Engine Reference

### LGO_v2_1.py

**Logistic-Gated Operators** — The proposed method for interpretable symbolic regression with gating mechanisms.

#### Features

- **Soft gating** (`lgo`): `x * sigmoid(a * (x - b))`
- **Hard gating** (`lgo_thre`): `sigmoid(a * (x - b))`
- **Multi-arity gates**: `lgo_and2`, `lgo_or2`, `lgo_and3`, `lgo_pair`
- **Typed GP**: Light/strict mode for domain-specific constraints
- **Safe math primitives**: Protected division, sqrt, log, pow

#### Main Function

```python
from LGO_v2_1 import run_lgo_sr_v2

results = run_lgo_sr_v2(
    X=X_train,                    # Training features
    y=y_train,                    # Training targets
    feature_names=feature_names,  # Feature names
    experiment="lgo_hard",        # "base", "lgo_soft", "lgo_hard"
    typed_mode="light",           # "none", "light", "strict"
    pop_size=600,                 # Population size
    ngen=80,                      # Number of generations
    hof_size=20,                  # Hall of fame size
    random_state=42,              # Random seed
    X_test=X_test,                # Test features (optional)
    y_test=y_test,                # Test targets (optional)
)
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment` | `"base"` | Experiment type: `base`, `lgo_soft`, `lgo_hard` |
| `typed_mode` | `"light"` | Type system: `none`, `light`, `strict` |
| `pop_size` | `600` | Population size |
| `ngen` | `80` | Number of generations |
| `tournament_size` | `5` | Tournament selection size |
| `cx_pb` | `0.85` | Crossover probability |
| `mut_pb` | `0.15` | Mutation probability |
| `max_height` | `10` | Maximum tree height |
| `hof_size` | `20` | Hall of fame size |
| `topk_cv` | `12` | Top-k for CV evaluation |

---

### PySR_v2.py

**PySR** — Julia-based symbolic regression with multi-island genetic programming.

#### SRBench-Aligned Configuration

Based on La Cava et al. (NeurIPS 2021) Table 5:
- `niterations=40`
- `population_size=1000`
- `populations=15` (multi-island)
- `maxsize=30`
- `timeout_in_seconds=3600`
- `parsimony=0.001`

#### Main Function

```python
from PySR_v2 import run_pysr_sr_v2

results = run_pysr_sr_v2(
    X=X_train,
    y=y_train,
    feature_names=feature_names,
    niterations=40,
    population_size=1000,
    populations=15,
    maxsize=30,
    timeout_in_seconds=3600,
    random_state=42,
    X_test=X_test,
    y_test=y_test,
)
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `niterations` | `40` | Number of iterations |
| `population_size` | `1000` | Population size per island |
| `populations` | `15` | Number of islands |
| `maxsize` | `30` | Maximum expression size |
| `parsimony` | `0.001` | Complexity penalty |
| `timeout_in_seconds` | `3600` | Maximum runtime (1 hour) |

---

### Operon_v2.py

**Operon** — C++ symbolic regression with Levenberg-Marquardt local optimization.

#### SRBench-Aligned Configuration

Based on La Cava et al. (NeurIPS 2021) Table 6:
- `generations=500`
- `population_size=1000`
- `max_length=50`
- `max_evaluations=500000`
- `tournament_size=5`
- `local_iterations=5` (LM optimization)

#### Main Function

```python
from Operon_v2 import run_operon_sr_v2

results = run_operon_sr_v2(
    X=X_train,
    y=y_train,
    feature_names=feature_names,
    generations=500,
    population_size=1000,
    max_length=50,
    max_evaluations=500000,
    random_state=42,
    X_test=X_test,
    y_test=y_test,
)
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generations` | `500` | Number of generations |
| `population_size` | `1000` | Population size |
| `max_length` | `50` | Maximum expression length |
| `max_evaluations` | `500000` | Maximum evaluations |
| `tournament_size` | `5` | Tournament selection size |
| `local_iterations` | `5` | LM optimization steps |

---

### PSTree_v3.py

**PS-Tree** — Piecewise Symbolic Tree combining decision trees with symbolic regression.

#### Features

- Hybrid decision tree + symbolic regression
- Cluster-based GP approach
- Automatic feature preprocessing
- Ridge regression fallback

#### Main Function

```python
from PSTree_v3 import run_pstree_once, PSTreeConfig

cfg = PSTreeConfig(
    height=6,
    feat_mut_rate=0.1,
    tournament_size=5,
    pop_size=500,
    generations=100,
)

result = run_pstree_once(
    X_tr=X_train,
    y_tr=y_train,
    X_te=X_test,
    y_te=y_test,
    seed=42,
    cfg=cfg,
    feature_names=feature_names,
)
```

#### Key Parameters (PSTreeConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `height` | `6` | Maximum tree height |
| `feat_mut_rate` | `0.1` | Feature mutation rate |
| `tournament_size` | `5` | Tournament selection size |
| `pop_size` | `500` | Population size |
| `generations` | `100` | Number of generations |

**Note:** PS-Tree requires Python 3.9 (compatibility issues with 3.10+).

---

### RILS_ROLS_v2_1.py

**RILS-ROLS** — Randomized Iterated Local Search with Ordinary Least Squares.

#### Features

- Iterated local search optimization
- OLS coefficient fitting
- Expression simplification
- Weighted complexity calculation

#### Main Function

```python
from RILS_ROLS_v2_1 import run_rils_rols_once, RILSROLSConfig

cfg = RILSROLSConfig(
    max_time=300,
    max_fit_calls=100000,
    max_complexity=50,
    seed=42,
)

result = run_rils_rols_once(
    X_tr=X_train,
    y_tr=y_train,
    X_te=X_test,
    y_te=y_test,
    seed=42,
    cfg=cfg,
    feature_names=feature_names,
)
```

#### Key Parameters (RILSROLSConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_time` | `300` | Maximum runtime (seconds) |
| `max_fit_calls` | `100000` | Maximum fitness evaluations |
| `max_complexity` | `50` | Maximum expression complexity |

---

### AutoScore_v2.py

**AutoScore** — Automated clinical scoring system development framework.

#### Features

Pure Python implementation compatible with R version:
1. **Variable ranking**: Random forest feature importance
2. **Variable selection**: Parsimony analysis
3. **Variable transformation**: Quantile binning `[0, 0.05, 0.2, 0.8, 0.95, 1]`
4. **Score derivation**: Two-step logistic regression
5. **Score fine-tuning**: Reference category adjustment
6. **Performance evaluation**: AUROC, Brier score

#### Main Classes

```python
from AutoScore_v2 import AutoScore, AutoScoreConfig

config = AutoScoreConfig(
    n_trees=100,           # RF trees for ranking
    max_variables=20,      # Maximum variables to select
    max_score=100,         # Maximum total score
    random_state=42,
)

model = AutoScore(config)

# Module 1: Variable ranking
ranking = model.compute_variable_ranking(X_train, y_train, feature_names)

# Module 2: Variable selection
selected = model.compute_variable_selection(X_train, y_train, feature_names, max_vars=10)

# Module 3-4: Build scoring model
model.fit(X_train, y_train, feature_names, n_variables=10)

# Module 6: Evaluate
scores = model.predict_score(X_test)
probs = model.predict_proba(X_test)
```

#### Key Parameters (AutoScoreConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_trees` | `100` | Random forest trees for ranking |
| `max_variables` | `20` | Maximum variables to select |
| `quantiles` | `[0, 0.05, 0.2, 0.8, 0.95, 1]` | Binning quantiles |
| `max_score` | `100` | Maximum total score |

---

### InterpretML_v1.py

**InterpretML (EBM)** — Explainable Boosting Machine for interpretable ML.

#### Features

- Generalized Additive Model (GAM) with boosting
- Automatic interaction detection
- Feature importance and shape functions
- Both classification and regression support

#### Main Classes

```python
from InterpretML_v1 import EBMWrapper, EBMConfig

config = EBMConfig(
    max_bins=256,
    interactions=10,
    outer_bags=8,
    learning_rate=0.01,
    random_state=42,
)

model = EBMWrapper(config)

# Fit model
model.fit(X_train, y_train, feature_names)

# Predict
probs = model.predict_proba(X_test)
scores = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance()
```

#### Key Parameters (EBMConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_bins` | `256` | Feature binning granularity |
| `interactions` | `10` | Number of auto-detected interactions |
| `outer_bags` | `8` | Outer bagging iterations |
| `learning_rate` | `0.01` | Boosting learning rate |
| `min_samples_leaf` | `2` | Minimum samples per leaf |
| `max_leaves` | `3` | Maximum leaves per tree |
| `early_stopping_rounds` | `50` | Early stopping patience |

---

## Environment Requirements

### Main Environment (py310)

For LGO, PySR, Operon:

```bash
conda activate py310
```

Required packages:
- Python 3.10
- numpy ≥1.21.0
- pandas ≥1.3.0
- scikit-learn ≥1.0.0
- deap ≥1.3.1 (LGO)
- pysr (PySR, requires Julia)
- pyoperon (Operon)

### PS-Tree Environment

```bash
conda activate pstree
```

Required packages:
- Python 3.9 (required)
- pstree
- gplearn

### RILS-ROLS Environment

```bash
conda activate rils-rols
```

Required packages:
- Python 3.11
- rils-rols

### Clinical Baselines

In main environment (py310):

```bash
pip install interpret  # InterpretML/EBM
```

AutoScore is implemented in pure Python with no additional dependencies.

---

## References

### Proposed Method

- **LGO**: Deng et al. (2025). Logistic-Gated Operators for Interpretable Symbolic Regression.

### Symbolic Regression Baselines

- **PySR**: Cranmer (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. arXiv:2305.01582
- **Operon**: Burlacu et al. (2020). Operon C++: An Efficient Genetic Programming Framework. GECCO 2020
- **PS-Tree**: Zhang et al. (2022). PS-Tree: A Piecewise Symbolic Regression Tree. Swarm and Evolutionary Computation
- **RILS-ROLS**: Kartelj & Djukanović (2023). RILS-ROLS: Robust Symbolic Regression via Iterated Local Search. ACM TELO

### Clinical Baselines

- **AutoScore**: Xie et al. (2020). AutoScore: A Machine Learning–Based Automatic Clinical Score Generator. JMIR Medical Informatics
- **InterpretML/EBM**: Nori et al. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. arXiv:1909.09223

### Benchmark

- **SRBench**: La Cava et al. (2021). Contemporary Symbolic Regression Methods and their Relative Performance. NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2107.14351

---
