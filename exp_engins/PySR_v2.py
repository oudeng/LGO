# -*- coding: utf-8 -*-
"""
PySR_v2.py - SRBench-Aligned PySR Wrapper
Author: Ou Deng
Updated: Nov 2025 (R2 revision - SRBench alignment)

Based on SRBench (La Cava et al., NeurIPS 2021) recommended configurations:
- niterations=40 (default)
- population_size=1000 (SRBench Table 5)
- populations=15 (multi-island default)
- maxsize=30 (SRBench hyperparameter space)
- timeout_in_seconds=3600 (SRBench: 1 hour max)
- parsimony=0.001 (complexity penalty)

References:
- SRBench: https://github.com/cavalab/srbench
- La Cava et al. (2021) arXiv:2107.14351
- Cranmer (2023) arXiv:2305.01582
"""
from __future__ import annotations

import time
import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

try:
    from pysr import PySRRegressor
except Exception:
    PySRRegressor = None

# ==========================================================
# Safe math and metrics 
# ==========================================================
EPS = 1e-12

def mse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    yhat_clipped = np.clip(yhat, -1e15, 1e15)
    with np.errstate(all="ignore"):
        diff = y - yhat_clipped
        diff_clipped = np.clip(diff, -1e7, 1e7)
        mse_val = np.mean(diff_clipped ** 2)
    return float(mse_val) if np.isfinite(mse_val) else 1e9

def r2(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - yhat) ** 2)
    if ss_tot < EPS:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def clip_and_sanitize(yhat, y_clip=None):
    yhat = np.asarray(yhat, dtype=float)
    if y_clip is not None and len(y_clip) == 2:
        yhat = np.clip(yhat, y_clip[0], y_clip[1])
    return np.where(np.isfinite(yhat), yhat, 0.0)

# ==========================================================
# Evaluation functions
# ==========================================================
def evaluate_pysr_expr(expr_str: str, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Evaluate a PySR expression string on data X"""
    try:
        namespace = {}
        for i, fname in enumerate(feature_names):
            namespace[fname] = X[:, i]
            namespace[f'x{i}'] = X[:, i]
        
        namespace.update({
            'sqrt': np.sqrt,
            'square': lambda x: x**2,
            'cube': lambda x: x**3,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'tanh': np.tanh,
            'abs': np.abs,
            'sign': np.sign,
            'inv': lambda x: 1.0 / (x + EPS),
        })
        
        expr_clean = str(expr_str).strip()
        
        # Handle subscript notation
        subscripts = {'₀':'0', '₁':'1', '₂':'2', '₃':'3', '₄':'4', '₅':'5', '₆':'6', '₇':'7', '₈':'8', '₉':'9'}
        for sub, digit in subscripts.items():
            expr_clean = expr_clean.replace(sub, digit)
        
        with np.errstate(all='ignore'):
            result = eval(expr_clean, {"__builtins__": {}}, namespace)
        
        return np.asarray(result, dtype=float)
    except Exception:
        return np.zeros(X.shape[0])

def calc_weighted_complexity(expr_str: str) -> float:
    """Calculate weighted complexity matching LGO's approach"""
    if not expr_str or expr_str in ["N/A", "Error extracting", None]:
        return 0.0
    
    expr_lower = str(expr_str).lower()
    
    # Weights matching LGO's DEFAULT_COMPLEXITY
    weights = {
        '+': 1.0, '-': 1.0, '*': 1.0, '/': 1.5,
        'sqrt': 1.5, 'square': 1.5, 'cube': 1.5,
        'log': 1.5, 'exp': 1.5,
        'sin': 1.5, 'cos': 1.5, 'tan': 1.5, 'tanh': 1.5,
        'abs': 1.5, 'sign': 1.0, 'pow': 2.0, 'inv': 1.5
    }
    
    complexity = 0.0
    
    for op, weight in weights.items():
        if op in ['+', '-', '*', '/']:
            complexity += expr_lower.count(op) * weight
        else:
            pattern = r'\b' + re.escape(op) + r'\b'
            complexity += len(re.findall(pattern, expr_lower)) * weight
    
    variables = set(re.findall(r'\b[A-Za-z_]\w*\b', expr_str))
    func_names = set(weights.keys())
    variables = {v for v in variables if v.lower() not in func_names}
    complexity += len(variables) * 0.5
    
    constants = len(re.findall(r'\b\d+\.?\d*\b', expr_str))
    complexity += constants * 0.25
    
    return max(1.0, complexity)

def crossval_pysr(expr_str: str, X: np.ndarray, y: np.ndarray, 
                  feature_names: List[str], n_splits: int = 5, 
                  random_state: int = 0) -> tuple:
    """Compute CV metrics for a PySR expression"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mses, r2s = [], []
    
    for tr, va in kf.split(X):
        yhat_va = evaluate_pysr_expr(expr_str, X[va], feature_names)
        yhat_va = clip_and_sanitize(yhat_va)
        mses.append(mse(y[va], yhat_va))
        r2s.append(r2(y[va], yhat_va))
    
    cv_loss = float(np.median(mses)) if mses else 1e9
    instability = float(np.std(mses) / max(np.mean(mses), EPS)) if mses else 1.0
    cv_r2 = float(np.mean(r2s)) if r2s else 0.0
    
    return cv_loss, instability, cv_r2

# ==========================================================
# Main PySR runner (SRBench-Aligned)
# ==========================================================
def run_pysr_sr_v2(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    experiment: str = "base",
    # ========== SRBench-Aligned Parameters ==========
    # Reference: SRBench Tables 4-6, La Cava et al. (2021)
    pysr_niterations: int = 40,           # SRBench default
    pysr_population_size: int = 1000,     # SRBench Table 5: npop=1000
    pysr_populations: int = 15,           # SRBench default multi-island
    pysr_maxsize: int = 30,               # SRBench hyperparameter space
    pysr_maxdepth: int = 10,              # Tree depth limit
    pysr_timeout_in_seconds: int = 3600,  # SRBench: 1 hour max per run
    pysr_parsimony: float = 0.001,        # SRBench complexity penalty
    pysr_ncyclesperiteration: int = 550,  # SRBench Table 5
    pysr_binary_operators: List[str] = None,
    pysr_unary_operators: List[str] = None,
    # ========== Legacy Parameters (for compatibility) ==========
    pop_size: int = 600,
    ngen: int = 80,
    tournament_size: int = 5,
    cx_pb: float = 0.85,
    mut_pb: float = 0.15,
    max_height: int = 10,
    hof_size: int = 20,
    topk_cv: int = 12,
    topk_local_opt: int = 6,
    local_opt_steps: int = 60,
    micro_mutation_prob: float = 0.10,
    cv_proxy_weight: float = 0.05,
    cv_proxy_subsample: float = 0.30,
    cv_proxy_folds: int = 2,
    random_state: int = 0,
    y_clip=None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Run PySR with SRBench-aligned configuration.
    
    SRBench Reference Configuration (from Tables 4-6):
    - Population: 1000 individuals
    - Multi-island: 15 populations
    - Iterations: 40
    - Max size: 30 nodes
    - Timeout: 1 hour
    - Termination: 500k evaluations or timeout
    """
    
    if PySRRegressor is None:
        raise RuntimeError("PySR is not installed. Install with: pip install pysr")
    
    t0 = time.time()
    
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # SRBench-aligned operator sets
    # Reference: SRBench algorithms/pysr/regressor.py
    if pysr_binary_operators is None:
        pysr_binary_operators = ["+", "-", "*", "/"]
    if pysr_unary_operators is None:
        # Extended operator set for comprehensive search
        pysr_unary_operators = ["sqrt", "exp", "log", "sin", "cos"]
    
    # Build complexity weights for fair comparison
    complexity_weights = {
        "+": 1, "-": 1, "*": 1, "/": 1.5,
        "sqrt": 1.5, "exp": 1.5, "log": 1.5,
        "sin": 1.5, "cos": 1.5, "tan": 1.5,
        "square": 1.5, "cube": 1.5, "inv": 1.5
    }
    
    if verbose:
        print(f"[PySR-SRBench] Configuration:")
        print(f"  niterations={pysr_niterations}")
        print(f"  population_size={pysr_population_size}")
        print(f"  populations={pysr_populations}")
        print(f"  maxsize={pysr_maxsize}")
        print(f"  timeout={pysr_timeout_in_seconds}s")
        print(f"  parsimony={pysr_parsimony}")
        print(f"  binary_operators={pysr_binary_operators}")
        print(f"  unary_operators={pysr_unary_operators}")
    
    # SRBench-aligned PySR configuration
    model = PySRRegressor(
        niterations=pysr_niterations,
        population_size=pysr_population_size,
        populations=pysr_populations,
        ncycles_per_iteration=pysr_ncyclesperiteration,
        binary_operators=pysr_binary_operators,
        unary_operators=pysr_unary_operators,
        maxsize=pysr_maxsize,
        maxdepth=pysr_maxdepth,
        parsimony=pysr_parsimony,
        timeout_in_seconds=pysr_timeout_in_seconds,
        complexity_of_operators=complexity_weights,
        progress=verbose,
        random_state=random_state,
        procs=1,  # Single process for fair comparison
        temp_equation_file=False,
        # Additional SRBench-recommended settings
        batching=False,
        turbo=False,
    )
    
    if verbose:
        print(f"[PySR-SRBench] Starting evolution...")
    
    # Fit model
    model.fit(X, y, variable_names=list(feature_names))
    runtime = time.time() - t0
    
    # Extract equations
    rows = []
    
    try:
        eq_df = model.equations_
        if eq_df is None or len(eq_df) == 0:
            print("[PySR-SRBench] Warning: No equations found")
            return pd.DataFrame()
        
        eq_df = eq_df.sort_values(["loss", "complexity"], ascending=[True, True])
        
        K = min(topk_cv, len(eq_df))
        
        if verbose:
            print(f"[PySR-SRBench] Evaluating top {K} candidates...")
        
        for i in range(K):
            row_data = eq_df.iloc[i]
            
            if "sympy_format" in row_data and row_data["sympy_format"] is not None:
                expr_str = str(row_data["sympy_format"])
            elif "equation" in row_data and row_data["equation"] is not None:
                expr_str = str(row_data["equation"])
            else:
                continue
            
            if verbose and i < 5:
                print(f"  Candidate {i+1}/{K}: {expr_str[:60]}...")
            
            cv_loss, instability, cv_r2 = crossval_pysr(
                expr_str, X, y, feature_names, n_splits=5, random_state=random_state
            )
            
            test_loss, test_r2 = (np.nan, np.nan)
            if X_test is not None and y_test is not None:
                yhat_test = evaluate_pysr_expr(expr_str, X_test, feature_names)
                yhat_test = clip_and_sanitize(yhat_test)
                test_loss = mse(y_test, yhat_test)
                test_r2 = r2(y_test, yhat_test)
            
            rows.append({
                "rank": i,
                "expr": expr_str,
                "cv_loss": float(cv_loss),
                "cv_r2": float(cv_r2),
                "instability": float(instability),
                "test_loss": float(test_loss) if np.isfinite(test_loss) else np.nan,
                "test_r2": float(test_r2) if np.isfinite(test_r2) else np.nan,
                "complexity": calc_weighted_complexity(expr_str),
                "height": int(row_data.get("depth", 0)),
                "size": int(row_data.get("complexity", len(expr_str))),
                "eval_calls": None,
                "runtime_sec": float(runtime),
                "used_lgo": False,
                "used_lgo_thre": False,
                "used_lgo_pair": False,
                "scaler_json": json.dumps({"mean": [0]*X.shape[1], "std": [1]*X.shape[1]}),
                "experiment": "base",
                "typed_mode": "none",
                "use_zscore": False,
                "prior_mode": "baseline",
                # SRBench metadata
                "srbench_aligned": True,
                "pysr_niterations": pysr_niterations,
                "pysr_population_size": pysr_population_size,
                "pysr_populations": pysr_populations,
                "pysr_maxsize": pysr_maxsize,
                "pysr_timeout": pysr_timeout_in_seconds,
            })
    
    except Exception as e:
        print(f"[PySR-SRBench] Error processing equations: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["cv_loss", "complexity", "instability"], 
                            ascending=[True, True, True]).reset_index(drop=True)
    
    if verbose:
        print(f"[PySR-SRBench] Completed in {runtime:.1f}s, returned {len(df)} expressions")
    
    return df

run_pysr_sr = run_pysr_sr_v2
