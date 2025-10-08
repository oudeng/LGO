# rils_rols_v2_fixed.py - RILS-ROLS with standardized metrics
"""
RILS-ROLS v2 Fixed - With standardized CV metrics and added instability/complexity

Author: Ou Deng on Sep 15, 2025
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Set
import time
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression

# Import RILS-ROLS
def _import_rils_rols_regressor():
    """Import RILSROLSRegressor from the correct location"""
    try:
        from rils_rols.rils_rols import RILSROLSRegressor
        return RILSROLSRegressor, "rils_rols.rils_rols"
    except ImportError as e:
        return None, f"import_failed: {str(e)}"

# Global variable to store feature names for expression mapping
FEATURE_NAMES = []

# Gate/binary features that should not be filtered even if imbalanced
GATE_WHITELIST = {
    "gender_std", "age_band", "mechanical_ventilation_std", 
    "vasopressor_use_std", "DS", "sex", "is_male", "is_female",
    "gender", "sex_male", "sex_female", "binary_*"
}

# Add weighted complexity calculation
def calc_weighted_complexity(expr_str):
    """Calculate weighted complexity like LGO"""
    if not expr_str or expr_str in ["N/A", "Error extracting", None]:
        return 0.0
    
    # Check if constant
    try:
        float(expr_str)
        return 1.0  # Constant has minimal complexity
    except:
        pass
    
    expr_lower = str(expr_str).lower()
    
    weights = {
        '+': 1.0, '-': 1.0, '*': 1.0, '/': 1.5,
        'sqrt': 1.5, 'log': 1.5, 'exp': 1.5,
        'sin': 1.5, 'cos': 1.5, 'tan': 1.5,
        'pow': 2.0, 'abs': 1.5, 'sign': 1.0
    }
    
    complexity = 0.0
    for op, weight in weights.items():
        # Count occurrences, avoiding partial matches
        if op in ['+', '-', '*', '/']:
            complexity += expr_lower.count(op) * weight
        else:
            # Use word boundary for function names
            pattern = r'\b' + re.escape(op) + r'\b'
            complexity += len(re.findall(pattern, expr_lower)) * weight
    
    # Add complexity for variables
    variables = set(re.findall(r'\b[A-Za-z_]\w*\b', expr_str))
    # Filter out function names
    func_names = {'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'abs', 'sign', 'pow'}
    variables = {v for v in variables if v.lower() not in func_names}
    complexity += len(variables) * 0.5
    
    # Add complexity for constants
    constants = len(re.findall(r'\b\d+\.?\d*\b', expr_str))
    complexity += constants * 0.25
    
    return max(1.0, complexity)

@dataclass
class RILSROLSConfig:
    """Configuration for RILS-ROLS experiments"""
    seed: int = 0
    max_fit_calls: int = 100000
    max_time: int = 100
    complexity_penalty: float = 0.001
    max_complexity: int = 50
    sample_size: float = 1.0
    verbose: bool = False
    timeout_sec: Optional[int] = None

def safe_eval_expr(expr_str, X, feature_names=None):
    """
    Safely evaluate expression with NaN/Inf protection
    Returns: (predictions, is_valid)
    """
    try:
        if not expr_str or expr_str in ["N/A", "Error extracting", None]:
            return np.full(X.shape[0], np.nan), False
        
        # If expression is just a constant
        try:
            const_val = float(expr_str)
            return np.full(X.shape[0], const_val), True
        except:
            pass
        
        # Map feature names to X columns
        if feature_names:
            # Create a safe namespace with numpy functions
            namespace = {
                'exp': np.exp,
                'log': lambda x: np.log(np.abs(x) + 1e-9),  # Safe log
                'sqrt': lambda x: np.sqrt(np.abs(x) + 1e-9),  # Safe sqrt
                'abs': np.abs,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
            }
            
            # Add feature columns to namespace
            for i, fname in enumerate(feature_names):
                if i < X.shape[1]:
                    namespace[fname] = X[:, i]
            
            # Evaluate expression
            result = eval(expr_str, namespace)
            
            # Check for NaN/Inf
            if np.any(~np.isfinite(result)):
                return result, False
            
            return result, True
        else:
            return np.full(X.shape[0], np.nan), False
            
    except Exception as e:
        print(f"    [WARNING] Expression evaluation failed: {e}")
        return np.full(X.shape[0], np.nan), False

def linear_refit(y_true, f_x):
    """
    Refit linear scaling: y = alpha * f(x) + beta
    Returns: alpha, beta, y_pred
    """
    try:
        if np.all(~np.isfinite(f_x)) or np.std(f_x) < 1e-10:
            # Constant or invalid predictions
            return 0.0, np.mean(y_true), np.full_like(y_true, np.mean(y_true))
        
        # Use LinearRegression for robust fitting
        lr = LinearRegression()
        f_x_reshaped = f_x.reshape(-1, 1)
        lr.fit(f_x_reshaped, y_true)
        
        alpha = lr.coef_[0]
        beta = lr.intercept_
        y_pred = lr.predict(f_x_reshaped)
        
        return alpha, beta, y_pred
        
    except Exception as e:
        print(f"    [WARNING] Linear refit failed: {e}")
        return 0.0, np.mean(y_true), np.full_like(y_true, np.mean(y_true))

def _complexity_from_expression(expr_str):
    """Extract complexity metrics from RILS-ROLS expression - renamed depth to height"""
    try:
        if not expr_str or expr_str in ["N/A", "Error extracting", None]:
            return np.nan, np.nan
        
        # Check if constant
        try:
            float(expr_str)
            return 1, 0  # Constant: size=1, height=0
        except:
            pass
        
        # Count operations and terms
        operations = len(re.findall(r'[+\-*/^]|exp|log|sin|cos|tan|sqrt|abs', expr_str))
        
        # Count unique variables
        variables = set(re.findall(r'\bx\d+\b|[A-Za-z_]\w*(?<!exp)(?<!log)(?<!sin)(?<!cos)(?<!sqrt)(?<!abs)', expr_str))
        variables = {v for v in variables if v not in ['exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'abs']}
        n_vars = len(variables)
        
        # Estimate height from parenthesis nesting
        max_height = 0
        current_depth = 0
        for char in expr_str:
            if char == '(':
                current_depth += 1
                max_height = max(max_height, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Count constants
        constants = len(re.findall(r'\b\d+\.?\d*\b', expr_str))
        
        # Size is roughly operations + variables + constants
        size = max(1, operations + n_vars + constants)
        
        return size, max(0, max_height)  # Return height instead of depth
        
    except Exception as e:
        print(f"    [WARNING] Error calculating complexity: {e}")
        return np.nan, np.nan

def _filter_near_constant_features_safe(X_tr, X_te, names, var_eps=1e-12, uniq_ratio=0.01, whitelist=None):
    """Filter near-constant features with whitelist support and robust fallback"""
    if whitelist is None:
        whitelist = GATE_WHITELIST
    
    X_tr = np.asarray(X_tr)
    X_te = np.asarray(X_te)
    keep = []
    dropped_names = []
    variances = []
    
    for j in range(X_tr.shape[1]):
        name = names[j] if j < len(names) else f'X{j}'
        
        col = X_tr[:, j]
        col_clean = col[~np.isnan(col)]
        var = np.nanvar(col_clean) if col_clean.size > 0 else 0
        variances.append(var)
        
        # Whitelist check
        is_whitelisted = False
        for white_pattern in whitelist:
            if white_pattern.endswith('*'):
                if name.startswith(white_pattern[:-1]):
                    is_whitelisted = True
                    break
            elif name == white_pattern:
                is_whitelisted = True
                break
        
        if is_whitelisted:
            keep.append(j)
            continue
        
        if col_clean.size == 0:
            dropped_names.append(name)
            continue
        
        unique = np.unique(col_clean)
        if unique.size <= 1:
            dropped_names.append(name)
            continue
        
        std = np.nanstd(col_clean)
        uniq_frac = unique.size / col_clean.size
        
        if std <= var_eps or uniq_frac <= uniq_ratio:
            dropped_names.append(name)
            continue
        
        keep.append(j)
    
    keep = np.array(keep, dtype=int)
    
    if keep.size == 0:
        print(f"    [WARNING] All features were near-constant. Keeping features with highest variance.")
        variances = np.array(variances)
        n_keep = min(3, len(variances))
        keep = np.argsort(variances)[-n_keep:][::-1]
        keep = np.array(keep, dtype=int)
    
    X_tr_filtered = X_tr[:, keep]
    X_te_filtered = X_te[:, keep]
    names_filtered = [names[i] for i in keep]
    
    all_indices = set(range(len(names)))
    keep_set = set(keep.tolist())
    drop_mask = [1 if i not in keep_set else 0 for i in range(len(names))]
    
    if dropped_names:
        print(f"    [INFO] Dropped {len(dropped_names)} near-constant features: {dropped_names[:5]}{'...' if len(dropped_names) > 5 else ''}")
    
    return X_tr_filtered, X_te_filtered, names_filtered, keep, drop_mask

def _map_expression_to_features(expr_str, feature_names):
    """Map x0, x1, etc. to actual feature names in RILS-ROLS expression"""
    if not expr_str or not feature_names:
        return expr_str
    
    mapped_expr = expr_str
    
    # Sort by index in descending order to avoid x1 being replaced before x10
    indices = []
    for i, fname in enumerate(feature_names):
        indices.append((i, fname))
    indices.sort(key=lambda x: x[0], reverse=True)
    
    for i, fname in indices:
        mapped_expr = re.sub(f'\\bx{i}\\b', fname, mapped_expr)
    
    return mapped_expr

def _kfold_cv_with_expressions(X, y, build_fn, seed: int, feature_names, k: int = 5):
    """
    K-fold cross validation with standardized metrics (median MSE, mean R², instability)
    Returns: cv_results (list of dicts with r2, expr, scaler, etc.)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    cv_results = []
    mses = []  # Collect all MSEs for instability calculation
    r2s = []   # Collect all R²s
    
    fold_idx = 0
    for tr, va in kf.split(X):
        fold_idx += 1
        
        # Standardize within each fold
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        
        yt, yv = y[tr], y[va]
        
        m = build_fn()
        try:
            # Train model
            m.fit(Xt, yt)
            
            # Get expression
            expr_raw = m.model_string()
            
            # Handle different return types
            if isinstance(expr_raw, (float, int, np.number)):
                expr = f"{float(expr_raw):.6f}"
                is_constant = True
            else:
                expr = str(expr_raw)
                is_constant = False
            
            # Map to feature names
            expr = _map_expression_to_features(expr, feature_names)
            
            # Direct prediction for validation
            yh = m.predict(Xv)
            mse = mean_squared_error(yv, yh)
            r2 = r2_score(yv, yh)
            
            mses.append(mse)
            r2s.append(r2)
            
            # Safe evaluation check
            yh_safe, eval_ok = safe_eval_expr(expr, Xv, feature_names)
            
            print(f"      Fold {fold_idx}/{k}: R² = {r2:.4f}, Expr = {expr[:50]}...")
            
            cv_results.append({
                'fold': fold_idx,
                'r2': r2,
                'mse': mse,
                'expr': expr,
                'is_constant': is_constant,
                'eval_ok': eval_ok,
                'scaler': scaler,
                'model': m
            })
            
        except Exception as e:
            print(f"      Fold {fold_idx}/{k} failed: {e}")
            cv_results.append({
                'fold': fold_idx,
                'r2': np.nan,
                'mse': np.nan,
                'expr': None,
                'is_constant': True,
                'eval_ok': False,
                'scaler': scaler,
                'model': None
            })
            mses.append(np.nan)
            r2s.append(np.nan)
    
    # Calculate instability
    EPS = 1e-12
    valid_mses = [m for m in mses if not np.isnan(m)]
    if valid_mses:
        instability = float(np.std(valid_mses) / max(np.mean(valid_mses), EPS))
    else:
        instability = 1.0
    
    # Add instability to results
    for result in cv_results:
        result['instability'] = instability
    
    return cv_results

def run_rils_rols_once(X_tr, y_tr, X_te, y_te, seed, cfg: RILSROLSConfig, feature_names: List[str] = None):
    """Run RILS-ROLS experiment with standardized metrics"""
    global FEATURE_NAMES
    
    print(f"    [Seed {seed}] Starting RILS-ROLS experiment...")
    
    if feature_names is None:
        feature_names = [f'x{i}' for i in range(X_tr.shape[1])]
    
    # Filter near-constant features
    original_n_features = X_tr.shape[1]
    X_tr, X_te, feature_names, keep_idx, drop_mask = _filter_near_constant_features_safe(
        X_tr, X_te, feature_names, whitelist=GATE_WHITELIST
    )
    
    print(f"    [Seed {seed}] Features: {original_n_features} -> {X_tr.shape[1]} after filtering")
    
    assert X_tr.shape[1] == X_te.shape[1] == len(feature_names), \
        f"Feature count mismatch after filtering"
    assert X_tr.shape[1] > 0, "No features remaining after filtering"
    
    FEATURE_NAMES = feature_names
    
    RILSROLSRegressor, backend = _import_rils_rols_regressor()
    t0 = time.time()

    row = {
        "experiment": "rils_rols",
        "engine": "rils_rols",
        "seed": seed,
        "split_id": 0,
        "runtime_sec": None,
        "cv_loss": np.nan,
        "cv_r2": np.nan,
        "test_loss": np.nan,
        "test_r2": np.nan,
        "size": np.nan,
        "height": np.nan,  # Changed from depth
        "complexity": np.nan,  # Added weighted complexity
        "instability": np.nan,  # Added instability
        "n_nodes": np.nan,
        "n_leaves": np.nan,
        "expr_str": None,
        "expr_cv_best": None,
        "expr_sympy": None,
        "candidates_csv": "",
        "rank": 1,
        "error": "",
        "engine_status": backend,
        "kept_features": json.dumps(list(feature_names)),
        "drop_mask": json.dumps(drop_mask),
        "alpha": np.nan,
        "beta": np.nan,
        "train_replay_ok": False,
        "test_replay_ok": False
    }

    if RILSROLSRegressor is None:
        row["error"] = f"rils_rols_import_failed: {backend}"
        row["runtime_sec"] = round(time.time() - t0, 4)
        return row

    def build():
        """Build RILS-ROLS model"""
        return RILSROLSRegressor(
            max_fit_calls=cfg.max_fit_calls,
            max_time=cfg.max_time,
            complexity_penalty=cfg.complexity_penalty,
            max_complexity=cfg.max_complexity,
            sample_size=cfg.sample_size,
            verbose=cfg.verbose,
            random_state=seed if seed != 0 else None
        )

    # Run K-fold CV and collect expressions
    print(f"    [Seed {seed}] Running 5-fold CV...")
    cv_results = _kfold_cv_with_expressions(X_tr, y_tr, build, seed, feature_names, k=5)
    
    # Calculate CV statistics with standardized metrics
    valid_mses = [r['mse'] for r in cv_results if not np.isnan(r['mse'])]
    valid_r2s = [r['r2'] for r in cv_results if not np.isnan(r['r2'])]
    
    if valid_mses and valid_r2s:
        row["cv_loss"] = float(np.median(valid_mses))  # Use median for cv_loss
        row["cv_r2"] = float(np.mean(valid_r2s))      # Use mean for cv_r2
        row["instability"] = cv_results[0]['instability']  # Already calculated
        print(f"    [Seed {seed}] CV R² = {row['cv_r2']:.4f}, Instability = {row['instability']:.4f}")
    
    # Select best CV expression
    best_fold = None
    best_r2 = -np.inf
    for result in cv_results:
        if not np.isnan(result['r2']) and result['r2'] > best_r2:
            if not result['is_constant']:  # Prefer non-constant models
                best_r2 = result['r2']
                best_fold = result
    
    # Fallback to best constant if no non-constant found
    if best_fold is None:
        for result in cv_results:
            if not np.isnan(result['r2']) and result['r2'] > best_r2:
                best_r2 = result['r2']
                best_fold = result
    
    if best_fold is None:
        row["error"] = "cv_all_failed"
        row["runtime_sec"] = round(time.time() - t0, 4)
        return row
    
    # Use best CV expression for final evaluation
    print(f"    [Seed {seed}] Using best CV expression from fold {best_fold['fold']} (R² = {best_fold['r2']:.4f})")
    
    try:
        # Standardize full training and test sets
        scaler_final = StandardScaler()
        Xtr_s = scaler_final.fit_transform(X_tr)
        Xte_s = scaler_final.transform(X_te)
        
        # Get expression
        expr = best_fold['expr']
        row["expr_cv_best"] = expr
        row["expr_str"] = expr
        
        # Calculate weighted complexity
        row["complexity"] = calc_weighted_complexity(expr)
        
        # Evaluate on training set with linear refit
        yh_train, train_ok = safe_eval_expr(expr, Xtr_s, feature_names)
        row["train_replay_ok"] = train_ok
        
        if train_ok and not np.all(np.isnan(yh_train)):
            # Linear refit on training data
            alpha, beta, yh_train_fitted = linear_refit(y_tr, yh_train)
            row["alpha"] = alpha
            row["beta"] = beta
            
            # Apply to test set
            yh_test, test_ok = safe_eval_expr(expr, Xte_s, feature_names)
            row["test_replay_ok"] = test_ok
            
            if test_ok and not np.all(np.isnan(yh_test)):
                # Apply linear scaling
                yh_test_fitted = alpha * yh_test + beta
                
                row["test_loss"] = float(mean_squared_error(y_te, yh_test_fitted))
                row["test_r2"] = float(r2_score(y_te, yh_test_fitted))
                # CRITICAL: Export native predictions with linear refit for aligned evaluation
                row["y_pred_test"] = [float(v) for v in np.asarray(yh_test_fitted).ravel()]
                print(f"    [Seed {seed}] Test R² = {row['test_r2']:.4f} (with linear refit α={alpha:.3f}, β={beta:.3f})")
            else:
                print(f"    [Seed {seed}] Test evaluation failed")
                row["error"] = "test_eval_failed"
        else:
            print(f"    [Seed {seed}] Train evaluation failed")
            row["error"] = "train_eval_failed"
        
        # Calculate size and height
        size, height = _complexity_from_expression(expr)
        if not np.isnan(size):
            row["size"] = int(size)
        if not np.isnan(height):
            row["height"] = int(height)  # Changed from depth
            
    except Exception as e:
        row["error"] = f"evaluation_failed: {str(e)}"
        print(f"    [Seed {seed}] Evaluation failed: {e}")

    row["runtime_sec"] = round(time.time() - t0, 4)
    print(f"    [Seed {seed}] Completed in {row['runtime_sec']}s")
    
    return row