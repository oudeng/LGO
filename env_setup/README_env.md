# Environment Setup for LGO and Baseline Experiments

Version 1.0 — December 2025

This guide explains how to set up conda environments for running **LGO (Logistic-Gated Operators)** and baseline symbolic regression methods.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Summary](#environment-summary)
- [Setup Instructions](#setup-instructions)
  - [1. Install Miniconda](#1-install-miniconda)
  - [2. Create Conda Environments](#2-create-conda-environments)
  - [3. Install Additional Packages](#3-install-additional-packages)
- [Method-Specific Notes](#method-specific-notes)
- [Troubleshooting](#troubleshooting)

---

## Overview

Due to dependency conflicts between different symbolic regression libraries, we use **separate conda environments** for different methods:

| Environment | Python | Primary Use | Key Packages |
|-------------|--------|-------------|--------------|
| `py310` | 3.10 | LGO, PySR, Operon | DEAP, PyTorch, Julia |
| `pstree` | 3.9 | PS-Tree | scikit-learn, glmnet |
| `rils-rols` | 3.11 | RILS-ROLS | NumPy 2.x, SciPy, Jupyter |

---

## Prerequisites

- Linux (Ubuntu 20.04+ recommended) or macOS
- 8GB+ RAM (16GB recommended for large experiments)
- 10GB+ disk space for all environments

---

## Environment Summary

### py310 (Main Environment)

**For:** LGO, PySR, Operon, AutoScore, InterpretML (EBM)

Key dependencies:
- Python 3.10
- DEAP ≥1.3.1 (genetic programming for LGO)
- PyTorch ≥1.10.0 (optional, for neural baselines)
- Julia ≥1.6.0 (required for PySR)
- scikit-learn ≥1.0.0
- SHAP ≥0.40.0

### pstree

**For:** PS-Tree (Piecewise Symbolic Tree)

Key dependencies:
- Python 3.9 (PS-Tree has compatibility issues with 3.10+)
- scikit-learn 1.1.1
- glmnet 2.2.1

### rils-rols

**For:** RILS-ROLS (Randomized Iterated Local Search)

Key dependencies:
- Python 3.11
- NumPy 2.x
- SciPy 1.16+
- JupyterLab (for interactive experiments)

---

## Setup Instructions

### 1. Install Miniconda

If you don't have conda installed:

```bash
# Download Miniconda (Linux x86_64)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# For macOS (Intel)
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# For macOS (Apple Silicon)
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh -b

# Initialize conda
~/miniconda3/bin/conda init

# Reload shell
source ~/.bashrc
# or for zsh: source ~/.zshrc
```

### 2. Create Conda Environments

```bash
# Navigate to the env_setup directory
cd env_setup

# (Optional) Remove old environments if they exist
conda env remove -n py310 -y
conda env remove -n pstree -y
conda env remove -n rils-rols -y
```

#### Main Environment (py310)

```bash
# Create the main environment for LGO, PySR, Operon
conda env create -f env_py310.yml

# Activate the environment
conda activate py310
```

#### PS-Tree Environment

```bash
# Create environment for PS-Tree
conda env create -f env_pstree.yml

# Activate the environment
conda activate pstree
```

#### RILS-ROLS Environment

```bash
# Create environment for RILS-ROLS
conda env create -f env_rils-rols.yml

# Activate the environment
conda activate rils-rols
```

### 3. Install Additional Packages

After creating environments, install method-specific packages:

#### PySR (in py310)

```bash
conda activate py310

# Install PySR
pip install pysr

# Initialize Julia dependencies (first run only, takes a few minutes)
python -c "import pysr; pysr.install()"
```

#### Operon (in py310)

```bash
conda activate py310

# Install Operon Python bindings
pip install pyoperon

# Or build from source for latest features:
# git clone https://github.com/heal-research/operon.git
# cd operon && mkdir build && cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j4 && make install
```

#### PS-Tree (in pstree)

```bash
conda activate pstree

# Install PS-Tree
pip install pstree
```

#### RILS-ROLS (in rils-rols)

```bash
conda activate rils-rols

# Clone and install RILS-ROLS
git clone https://github.com/kartelj/rils-rols.git
cd rils-rols
pip install -e .
```

#### AutoScore (in py310)

AutoScore is included in the LGO comparison framework as a pure Python implementation (`AutoScore_v2.py`). No additional installation required.

#### InterpretML / EBM (in py310)

```bash
conda activate py310

# Install InterpretML
pip install interpret

# Or via conda
# conda install -c conda-forge interpret
```

---

## Method-Specific Notes

### LGO

LGO is implemented in the project files (`LGO_v2_2.py`). No separate installation needed.

```python
# Verify LGO is working
from LGO_v2_2 import run_lgo_sr_v3
print("LGO OK")
```

### PySR

PySR requires Julia. The first run will download and compile Julia packages (~5-10 minutes).

```python
# Verify PySR is working
from pysr import PySRRegressor
print("PySR OK")
```

**Note:** PySR and Operon have different NumPy version requirements. A compatibility shim is implemented in the comparison scripts.

### Operon

Operon requires C++ compilation. Pre-built wheels are available for most platforms.

```python
# Verify Operon is working
import pyoperon
print("Operon OK")
```

### PS-Tree

PS-Tree requires Python 3.9. It has known issues with Python 3.10+.

```python
# Verify PS-Tree is working (in pstree environment)
from pstree import PSTreeClassifier
print("PS-Tree OK")
```

### RILS-ROLS

RILS-ROLS requires Python 3.11+ for best performance.

```python
# Verify RILS-ROLS is working (in rils-rols environment)
from rils_rols import RILSROLSRegressor
print("RILS-ROLS OK")
```

---

## Troubleshooting

### Julia/PySR Issues

```bash
# If PySR fails to initialize Julia
python -c "import pysr; pysr.install(precompile=True)"

# If Julia is not found
export JULIA_HOME=/path/to/julia
export PATH=$JULIA_HOME/bin:$PATH
```

### Operon Build Issues

```bash
# Ensure CMake and build tools are installed
conda install cmake gcc gxx -y

# Try installing pre-built wheel
pip install pyoperon --no-build-isolation
```

### PS-Tree Compatibility

If PS-Tree fails on Python 3.10+, use the dedicated `pstree` environment with Python 3.9.

### Memory Issues

For large experiments, increase memory limits:

```bash
# Set environment variable before running
export OMP_NUM_THREADS=4  # Limit OpenMP threads
export JULIA_NUM_THREADS=4  # Limit Julia threads
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate py310 | `conda activate py310` |
| Activate pstree | `conda activate pstree` |
| Activate rils-rols | `conda activate rils-rols` |
| Deactivate environment | `conda deactivate` |
| List environments | `conda env list` |
| Update environment | `conda env update -f env_*.yml` |
| Export environment | `conda env export > environment.yml` |

---

## Official Documentation

- **PySR:** https://github.com/MilesCranmer/PySR
- **Operon:** https://github.com/heal-research/operon
- **PS-Tree:** https://github.com/hengzhe-zhang/PS-Tree
- **RILS-ROLS:** https://github.com/kartelj/rils-rols
- **InterpretML:** https://github.com/interpretml/interpret
- **DEAP:** https://github.com/DEAP/deap

---
