# -*- coding: utf-8 -*-
"""
Operon_v1_fixed.py - Self-contained Operon wrapper with weighted complexity
Author: Upgraded from v5.4 to v5.6
Fixed: Proper mapping of X1, X2... to actual feature names in output
Fixed: Changed to calc_weighted_complexity
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
    
    # Handle subscript numbers first
    _sub_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = s.translate(_sub_digits)
    
    # Replace Operon operators
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    
    # Map X1, X2... to feature names (X1 -> feature_names[0], X2 -> feature_names[1], etc.)
    def replace_xn(match):
        n = int(match.group(1))
        if n > 0 and n <= len(feature_names):
            return feature_names[n-1]  # X1 corresponds to index 0
        return match.group(0)  # Keep original if out of range
    
    s = re.sub(r'\bX(\d+)\b', replace_xn, s)
    
    return s

def clean_operon_for_eval(expr_str: str, feature_names: List[str]) -> str:
    """Clean Operon expression for evaluation (maps to x0, x1... for eval)"""
    # Handle subscript numbers
    _sub_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = str(expr_str).translate(_sub_digits)
    
    # Replace Operon operators
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    
    # Handle X1, X2 format (1-based) -> x0, x1 (0-based)
    s = re.sub(r"\bX(\d+)\b", lambda m: f"x{int(m.group(1))-1}" if int(m.group(1)) > 0 else "x0", s)
    
    # Also replace feature names with x{i} if they appear
    for i, fname in enumerate(feature_names):
        s = s.replace(fname, f"x{i}")
    
    return s

def extract_equations_operon(model) -> List[str]:
    """Extract equation strings from Operon model"""
    eqs = []
    
    # Try various attributes
    for attr in ['pareto_front_', 'individuals_', 'model_', 'program_', 
                 'program_strings_', 'equations_', 'hall_of_fame_']:
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if isinstance(val, list):
                    for item in val[:20]:  # Take top 20
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
    
    # Try basic string conversion as last resort
    if not eqs:
        try:
            model_str = str(model)
            if model_str and not model_str.startswith("<") and "SymbolicRegressor" not in model_str:
                eqs.append(model_str)
        except Exception:
            pass
    
    # Remove duplicates
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
        # Clean expression for evaluation
        expr_clean = clean_operon_for_eval(expr_str, feature_names)
        
        # Create namespace
        namespace = {}
        for i in range(X.shape[1]):
            namespace[f'x{i}'] = X[:, i]
        
        # Add functions
        namespace.update({
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'abs': np.abs,
        })
        
        # Evaluate
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
        '×': 1.0, '÷': 1.5, '−': 1.0,  # Handle Operon's special characters
        'sqrt': 1.5, 'log': 1.5, 'exp': 1.5,
        'sin': 1.5, 'cos': 1.5, 'tan': 1.5,
        'abs': 1.5, 'sign': 1.0, 'pow': 2.0
    }
    
    complexity = 0.0
    
    # Count operators
    for op, weight in weights.items():
        if op in ['+', '-', '*', '/', '×', '÷', '−']:
            complexity += expr_lower.count(op) * weight
        else:
            # Use word boundary for function names to avoid partial matches
            pattern = r'\b' + re.escape(op) + r'\b'
            complexity += len(re.findall(pattern, expr_lower)) * weight
    
    # Add complexity for variables (lighter weight than operations)
    variables = set(re.findall(r'\b[A-Za-z_]\w*\b', expr_str))
    # Filter out function names
    func_names = {'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'abs', 'sign', 'pow'}
    variables = {v for v in variables if v.lower() not in func_names}
    complexity += len(variables) * 0.5
    
    # Add minimal complexity for constants
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
# Main Operon runner 
# ==========================================================
def run_operon_sr_v1(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    experiment: str = "base",
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
    **kwargs  # Accept and ignore other LGO-specific parameters
) -> pd.DataFrame:
    """Run Operon and return results in LGO v1 format"""
    
    if OperonRegressor is None:
        raise RuntimeError("Operon is not installed. Install operon or pyoperon")
    
    t0 = time.time()
    
    # Setup feature names
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # Build allowed symbols
    symbols = ["add", "sub", "mul", "div", "sqrt", "constant", "variable"]
    allowed_symbols_str = ",".join(symbols)
    
    # Get valid parameters for this Operon version
    sig = inspect.signature(OperonRegressor.__init__)
    valid_params = set(sig.parameters.keys())
    
    # Build parameters
    base_params = {
        "generations": ngen,
        "population_size": pop_size,
        "random_state": random_state,
    }
    
    # Add optional parameters if supported
    if "max_depth" in valid_params:
        base_params["max_depth"] = max_height
    if "max_length" in valid_params:
        base_params["max_length"] = 30  # max nodes
    elif "max_size" in valid_params:
        base_params["max_size"] = 30
    
    if "allowed_symbols" in valid_params:
        base_params["allowed_symbols"] = allowed_symbols_str
    if "n_threads" in valid_params:
        base_params["n_threads"] = 1
    
    if verbose:
        print(f"[Operon] Starting evolution with population size {pop_size}")
    
    # Create and fit model
    model = None
    try:
        model = OperonRegressor(**base_params)
        model.fit(X, y)
        runtime = time.time() - t0
    except Exception as e:
        # Try with minimal parameters
        try:
            minimal_params = {k: v for k, v in base_params.items() 
                            if k in ["generations", "population_size", "random_state"]}
            model = OperonRegressor(**minimal_params)
            model.fit(X, y)
            runtime = time.time() - t0
        except Exception as e2:
            print(f"[Operon] Error: Could not fit model: {e2}")
            return pd.DataFrame()
    
    # Extract equations (these will have X1, X2... notation)
    eqs_raw = extract_equations_operon(model)
    
    if not eqs_raw:
        print("[Operon] Warning: No equations extracted")
        return pd.DataFrame()
    
    rows = []
    K = min(topk_cv, len(eqs_raw))
    
    if verbose:
        print(f"[Operon] Evaluating {K} expressions...")
    
    for i in range(K):
        expr_raw = eqs_raw[i]  # Original expression with X1, X2...
        
        # Map to feature names for display
        expr_display = map_operon_to_feature_names(expr_raw, feature_names)
        
        if verbose and i < 5:
            print(f"  Candidate {i+1}/{K}: Evaluating...")
        
        try:
            # Compute CV metrics (using raw expression for evaluation)
            cv_loss, instability, cv_r2 = crossval_operon(
                expr_raw, X, y, feature_names, n_splits=5, random_state=random_state
            )
            
            # Test evaluation
            test_loss, test_r2 = (np.nan, np.nan)
            if X_test is not None and y_test is not None:
                yhat_test = evaluate_operon_expr(expr_raw, X_test, feature_names)
                yhat_test = clip_and_sanitize(yhat_test)
                test_loss = mse(y_test, yhat_test)
                test_r2 = r2(y_test, yhat_test)
            
            # Estimate complexity
            nodes = max(1, expr_display.count('+') + expr_display.count('-') + 
                       expr_display.count('*') + expr_display.count('/') + expr_display.count('sqrt'))
            
            # Create row in LGO format - USE calc_weighted_complexity
            rows.append({
                "rank": i,
                "expr": expr_display,  # Use the expression with feature names
                "cv_loss": float(cv_loss),
                "cv_r2": float(cv_r2),
                "instability": float(instability),
                "test_loss": float(test_loss) if np.isfinite(test_loss) else np.nan,
                "test_r2": float(test_r2) if np.isfinite(test_r2) else np.nan,
                "complexity": calc_weighted_complexity(expr_display),  # FIXED: Changed from calc_complexity_operon
                "height": min(nodes, max_height),
                "size": nodes,
                "eval_calls": None,  # Not available
                "runtime_sec": float(runtime),
                "used_lgo": False,  # Operon doesn't use lgo
                "used_lgo_thre": False,
                "used_lgo_pair": False,
                "scaler_json": json.dumps({"mean": [0]*X.shape[1], "std": [1]*X.shape[1]}),
                "experiment": "base",  # Operon always uses base operators
                "typed_mode": "none",
                "use_zscore": False,
                "prior_mode": "baseline",
            })
        except Exception as e:
            if verbose:
                print(f"    Failed to process expression {i}: {e}")
            continue
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["cv_loss", "complexity", "instability"], 
                            ascending=[True, True, True]).reset_index(drop=True)
    
    return df