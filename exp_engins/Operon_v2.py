# -*- coding: utf-8 -*-
"""
Operon_v2.py - SRBench-Aligned Operon Wrapper
Author: Ou Deng
Updated: Nov 2025 (R2 revision - SRBench alignment)

Based on SRBench (La Cava et al., NeurIPS 2021) recommended configurations:
- generations=500 (SRBench Table 6)
- population_size=1000 (SRBench Table 6: popsize=1000)
- max_length=50 (SRBench hyperparameter space)
- max_evaluations=500000 (SRBench termination criterion)
- tournament_size=5 (SRBench default)
- local_iterations=5 (Levenberg-Marquardt local search)

References:
- SRBench: https://github.com/cavalab/srbench
- La Cava et al. (2021) arXiv:2107.14351
- Burlacu et al. (2020) GECCO - Operon C++
"""
from __future__ import annotations

import time
import json
import re
import inspect
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Try different Operon imports
OperonRegressor = None
operon_module = None
try:
    import operon.sklearn
    from operon.sklearn import SymbolicRegressor as OperonRegressor
    operon_module = operon.sklearn
except Exception:
    try:
        import pyoperon.sklearn
        from pyoperon.sklearn import SymbolicRegressor as OperonRegressor
        operon_module = pyoperon.sklearn
    except Exception:
        try:
            import pyoperon.sklearn
            OperonRegressor = getattr(pyoperon.sklearn, "OperonRegressor", None)
            if OperonRegressor is None:
                OperonRegressor = getattr(pyoperon.sklearn, "Regressor", None)
            operon_module = pyoperon.sklearn
        except Exception:
            OperonRegressor = None

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
# Expression processing
# ==========================================================
def map_operon_to_feature_names(expr_str: str, feature_names: List[str]) -> str:
    """Map X1, X2... in Operon expression to actual feature names for display"""
    s = str(expr_str)
    
    _sub_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = s.translate(_sub_digits)
    
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    
    def replace_xn(match):
        n = int(match.group(1))
        if n > 0 and n <= len(feature_names):
            return feature_names[n-1]
        return match.group(0)
    
    s = re.sub(r'\bX(\d+)\b', replace_xn, s)
    
    return s

def clean_operon_for_eval(expr_str: str, feature_names: List[str]) -> str:
    """Clean Operon expression for evaluation"""
    _sub_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = str(expr_str).translate(_sub_digits)
    
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    
    s = re.sub(r"\bX(\d+)\b", lambda m: f"x{int(m.group(1))-1}" if int(m.group(1)) > 0 else "x0", s)
    
    for i, fname in enumerate(feature_names):
        s = s.replace(fname, f"x{i}")
    
    return s

def extract_equations_operon(model) -> List[str]:
    """Extract equation strings from Operon model"""
    eqs = []
    
    for attr in ['pareto_front_', 'individuals_', 'model_', 'program_', 
                 'program_strings_', 'equations_', 'hall_of_fame_']:
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if isinstance(val, list):
                    for item in val[:20]:
                        if isinstance(item, dict) and 'model' in item:
                            model_str = str(item['model'])
                            if model_str and not model_str.startswith("<"):
                                eqs.append(model_str)
                        elif hasattr(item, '__str__'):
                            item_str = str(item)
                            if item_str and not item_str.startswith("<"):
                                eqs.append(item_str)
                elif val is not None:
                    val_str = str(val)
                    if val_str and not val_str.startswith("<"):
                        eqs.append(val_str)
            except Exception:
                pass
    
    if not eqs:
        try:
            model_str = str(model)
            if model_str and not model_str.startswith("<") and "SymbolicRegressor" not in model_str:
                eqs.append(model_str)
        except Exception:
            pass
    
    seen = set()
    dedup = []
    for e in eqs:
        e = e.strip()
        if e and e not in seen and not e.startswith("<"):
            seen.add(e)
            dedup.append(e)
    
    return dedup

def evaluate_operon_expr(expr_str: str, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Evaluate an Operon expression on data X"""
    try:
        expr_clean = clean_operon_for_eval(expr_str, feature_names)
        
        namespace = {}
        for i in range(X.shape[1]):
            namespace[f'x{i}'] = X[:, i]
        
        namespace.update({
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'tanh': np.tanh,
            'abs': np.abs,
            'aq': lambda a, b: a / np.sqrt(1 + b**2),  # Analytic quotient
        })
        
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
    
    weights = {
        '+': 1.0, '-': 1.0, '*': 1.0, '/': 1.5,
        '×': 1.0, '÷': 1.5, '−': 1.0,
        'sqrt': 1.5, 'log': 1.5, 'exp': 1.5,
        'sin': 1.5, 'cos': 1.5, 'tan': 1.5, 'tanh': 1.5,
        'abs': 1.5, 'sign': 1.0, 'pow': 2.0, 'aq': 1.5
    }
    
    complexity = 0.0
    
    for op, weight in weights.items():
        if op in ['+', '-', '*', '/', '×', '÷', '−']:
            complexity += expr_lower.count(op) * weight
        else:
            pattern = r'\b' + re.escape(op) + r'\b'
            complexity += len(re.findall(pattern, expr_lower)) * weight
    
    variables = set(re.findall(r'\b[A-Za-z_]\w*\b', expr_str))
    func_names = {'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'tanh', 'abs', 'sign', 'pow', 'aq'}
    variables = {v for v in variables if v.lower() not in func_names}
    complexity += len(variables) * 0.5
    
    constants = len(re.findall(r'\b\d+\.?\d*\b', expr_str))
    complexity += constants * 0.25
    
    return max(1.0, complexity)

def crossval_operon(expr_str: str, X: np.ndarray, y: np.ndarray, 
                    feature_names: List[str], n_splits: int = 5, 
                    random_state: int = 0) -> tuple:
    """Compute CV metrics for an Operon expression"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mses, r2s = [], []
    
    for tr, va in kf.split(X):
        yhat_va = evaluate_operon_expr(expr_str, X[va], feature_names)
        yhat_va = clip_and_sanitize(yhat_va)
        mses.append(mse(y[va], yhat_va))
        r2s.append(r2(y[va], yhat_va))
    
    cv_loss = float(np.median(mses)) if mses else 1e9
    instability = float(np.std(mses) / max(np.mean(mses), EPS)) if mses else 1.0
    cv_r2 = float(np.mean(r2s)) if r2s else 0.0
    
    return cv_loss, instability, cv_r2

# ==========================================================
# Main Operon runner (SRBench-Aligned)
# ==========================================================
def run_operon_sr_v2(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    experiment: str = "base",
    # ========== SRBench-Aligned Parameters ==========
    # Reference: SRBench Table 6, La Cava et al. (2021)
    operon_generations: int = 500,            # SRBench large generation count
    operon_population_size: int = 1000,       # SRBench Table 6: popsize=1000
    operon_pool_size: int = 1000,             # SRBench: pool_size=popsize
    operon_max_length: int = 50,              # SRBench: max_length=50
    operon_max_evaluations: int = 500000,     # SRBench: 500k evaluation budget
    operon_tournament_size: int = 5,          # SRBench default
    operon_local_iterations: int = 5,         # SRBench: Levenberg-Marquardt local search
    operon_allowed_symbols: str = "add,mul,aq,exp,log,sin,cos,sqrt,constant,variable",
    operon_offspring_generator: str = "basic",
    operon_reinserter: str = "keep-best",
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
    Run Operon with SRBench-aligned configuration.
    
    SRBench Reference Configuration (from Table 6):
    - Population: 1000 individuals
    - Generations: 500
    - Max length: 50 nodes
    - Max evaluations: 500,000
    - Tournament size: 5
    - Local iterations: 5 (Levenberg-Marquardt)
    - Symbols: add, mul, aq, exp, log, sin, cos, sqrt, constant, variable
    """
    
    if OperonRegressor is None:
        raise RuntimeError("Operon is not installed. Install operon or pyoperon")
    
    t0 = time.time()
    
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # Get valid parameters for this Operon version
    sig = inspect.signature(OperonRegressor.__init__)
    valid_params = set(sig.parameters.keys())
    
    if verbose:
        print(f"[Operon-SRBench] Configuration:")
        print(f"  generations={operon_generations}")
        print(f"  population_size={operon_population_size}")
        print(f"  max_length={operon_max_length}")
        print(f"  max_evaluations={operon_max_evaluations}")
        print(f"  tournament_size={operon_tournament_size}")
        print(f"  local_iterations={operon_local_iterations}")
        print(f"  allowed_symbols={operon_allowed_symbols}")
    
    # SRBench-aligned parameters
    base_params = {
        "generations": operon_generations,
        "population_size": operon_population_size,
        "random_state": random_state,
    }
    
    # Add optional parameters if supported by this Operon version
    optional_params = {
        "pool_size": operon_pool_size,
        "max_length": operon_max_length,
        "max_evaluations": operon_max_evaluations,
        "tournament_size": operon_tournament_size,
        "local_iterations": operon_local_iterations,
        "allowed_symbols": operon_allowed_symbols,
        "offspring_generator": operon_offspring_generator,
        "reinserter": operon_reinserter,
        "n_threads": 1,  # Single thread for fair comparison
    }
    
    for param, value in optional_params.items():
        if param in valid_params:
            base_params[param] = value
    
    if verbose:
        print(f"[Operon-SRBench] Applied parameters: {list(base_params.keys())}")
    
    # Create and fit model
    model = None
    try:
        model = OperonRegressor(**base_params)
        model.fit(X, y)
        runtime = time.time() - t0
    except Exception as e:
        if verbose:
            print(f"[Operon-SRBench] Full config failed ({e}), trying minimal config...")
        try:
            minimal_params = {k: v for k, v in base_params.items() 
                            if k in ["generations", "population_size", "random_state"]}
            model = OperonRegressor(**minimal_params)
            model.fit(X, y)
            runtime = time.time() - t0
        except Exception as e2:
            print(f"[Operon-SRBench] Error: Could not fit model: {e2}")
            return pd.DataFrame()
    
    # Extract equations
    eqs_raw = extract_equations_operon(model)
    
    if not eqs_raw:
        print("[Operon-SRBench] Warning: No equations extracted")
        return pd.DataFrame()
    
    rows = []
    K = min(topk_cv, len(eqs_raw))
    
    if verbose:
        print(f"[Operon-SRBench] Evaluating {K} expressions...")
    
    for i in range(K):
        expr_raw = eqs_raw[i]
        expr_display = map_operon_to_feature_names(expr_raw, feature_names)
        
        if verbose and i < 5:
            print(f"  Candidate {i+1}/{K}: {expr_display[:60]}...")
        
        try:
            cv_loss, instability, cv_r2 = crossval_operon(
                expr_raw, X, y, feature_names, n_splits=5, random_state=random_state
            )
            
            test_loss, test_r2 = (np.nan, np.nan)
            if X_test is not None and y_test is not None:
                yhat_test = evaluate_operon_expr(expr_raw, X_test, feature_names)
                yhat_test = clip_and_sanitize(yhat_test)
                test_loss = mse(y_test, yhat_test)
                test_r2 = r2(y_test, yhat_test)
            
            nodes = max(1, expr_display.count('+') + expr_display.count('-') + 
                       expr_display.count('*') + expr_display.count('/') + 
                       expr_display.count('sqrt') + expr_display.count('exp') +
                       expr_display.count('log') + expr_display.count('sin') +
                       expr_display.count('cos'))
            
            rows.append({
                "rank": i,
                "expr": expr_display,
                "cv_loss": float(cv_loss),
                "cv_r2": float(cv_r2),
                "instability": float(instability),
                "test_loss": float(test_loss) if np.isfinite(test_loss) else np.nan,
                "test_r2": float(test_r2) if np.isfinite(test_r2) else np.nan,
                "complexity": calc_weighted_complexity(expr_display),
                "height": min(nodes, max_height),
                "size": nodes,
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
                "operon_generations": operon_generations,
                "operon_population_size": operon_population_size,
                "operon_max_length": operon_max_length,
                "operon_max_evaluations": operon_max_evaluations,
            })
        except Exception as e:
            if verbose:
                print(f"    Failed to process expression {i}: {e}")
            continue
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["cv_loss", "complexity", "instability"], 
                            ascending=[True, True, True]).reset_index(drop=True)
    
    if verbose:
        print(f"[Operon-SRBench] Completed in {runtime:.1f}s, returned {len(df)} expressions")
    
    return df

# Backward compatibility alias
run_operon_sr = run_operon_sr_v2
