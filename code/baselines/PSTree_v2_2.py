# PSTree_v2_1_fixed.py - Corrected version with standardized metrics
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
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

def _import_pstree_regressor():
    """Import PSTreeRegressor from the correct location"""
    try:
        from pstree.cluster_gp_sklearn import PSTreeRegressor
        return PSTreeRegressor, "pstree.cluster_gp_sklearn"
    except ImportError as e:
        return None, f"import_failed: {str(e)}"

# Global variable to store feature names for expression mapping
FEATURE_NAMES = []

# Gate/binary features that should not be filtered even if imbalanced
GATE_WHITELIST = {
    "gender_std", "age_band", "mechanical_ventilation_std", 
    "vasopressor_use_std", "DS", "sex", "is_male", "is_female"
}

# Add weighted complexity calculation
def calc_weighted_complexity(expr_str):
    """Calculate weighted complexity like LGO"""
    if not expr_str or expr_str in ["N/A", "PSTree model", None, "Error extracting"]:
        return 0.0
    
    expr_lower = str(expr_str).lower()
    
    weights = {
        '+': 1.0, '-': 1.0, '*': 1.0, '/': 1.5,
        'sqrt': 1.5, 'log': 1.5, 'exp': 1.5,
        'sin': 1.5, 'cos': 1.5, 'pow': 2.0,
        'abs': 1.5, 'sign': 1.0, 'neg': 1.0
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
    variables = {v for v in variables if v.lower() not in weights.keys()}
    complexity += len(variables) * 0.5
    
    return max(1.0, complexity)

class PSTreeCompatibleRegressor:
    """Wrapper class compatible with PSTree requirements - Ridge baseline"""
    
    def __init__(self, **kwargs):
        self.model = Ridge(alpha=1.0)
        self.best_pop = None
        self.original_features = "original"
        self.adaptive_tree = True
        self.n_features_ = None
        
        # Store all parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y, category=None):
        """Fit method that accepts category parameter"""
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self.best_pop = self
        return self
    
    def predict(self, X, y=None, category=None):
        """Predict method that accepts optional parameters"""
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def feature_synthesis(self, X, best_pop=None, original_features=None):
        """Required for PSTree predict"""
        return X

class GPLearnWrapper:
    """Wrapper for using gplearn with PSTree"""
    def __init__(self, **kwargs):
        from gplearn.genetic import SymbolicRegressor
        
        # Create a custom callback for progress updates
        self.generation_count = 0
        self.best_fitness = float('inf')
        
        def progress_callback(program):
            """Callback to show progress during GP evolution"""
            self.generation_count += 1
            current_fitness = program.fitness_
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                # Show progress every 10 generations or when fitness improves significantly
                if self.generation_count % 10 == 0 or (self.best_fitness < current_fitness * 0.95):
                    print(f"      Gen {self.generation_count}: Best fitness = {self.best_fitness:.4f}")
            return False  # Don't stop early
        
        self.gp = SymbolicRegressor(
            population_size=kwargs.get('population_size', 100),
            generations=kwargs.get('generations', 20),
            tournament_size=20,
            init_depth=(2, 4),
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs'),
            parsimony_coefficient=0.01,
            p_crossover=0.9,
            verbose=1 if kwargs.get('generations', 20) > 50 else 0,  # Verbose for long runs
            random_state=kwargs.get('random_state', None),
            n_jobs=-1,
            # Add callback for progress
            warm_start=False,
            low_memory=False
        )
        
        self.best_pop = None
        self.original_features = "original"
        self.adaptive_tree = True
        self.n_features_ = None
        self._expr_str = None
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y, category=None):
        print(f"      Starting GP evolution ({self.gp.population_size} pop, {self.gp.generations} gen)...")
        self.generation_count = 0
        self.best_fitness = float('inf')
        
        # Fit with periodic updates
        self.gp.fit(X, y)
        
        self.n_features_ = X.shape[1]
        self.best_pop = self.gp._program
        
        # Extract and store expression with feature names
        if hasattr(self.gp, '_program'):
            self._expr_str = self._map_expression_to_features(str(self.gp._program))
            print(f"      Final best: {self._expr_str[:100]}..." if len(self._expr_str) > 100 else f"      Final best: {self._expr_str}")
        
        return self
    
    def _map_expression_to_features(self, expr_str):
        """Map X0, X1, etc. to actual feature names"""
        global FEATURE_NAMES
        if FEATURE_NAMES:
            for i, fname in enumerate(FEATURE_NAMES):
                expr_str = expr_str.replace(f'X{i}', fname)
        return expr_str
    
    def predict(self, X, y=None, category=None):
        return self.gp.predict(X)
    
    def score(self, X, y):
        return self.gp.score(X, y)
    
    def feature_synthesis(self, X, best_pop=None, original_features=None):
        return X

@dataclass
class PSTreeConfig:
    seed: int = 0
    max_depth: int = 3
    max_leaf_nodes: int = 4
    min_samples_leaf: int = 20
    pop_size: int = 100
    ngen: int = 60
    timeout_sec: Optional[int] = None

def _tree_complexity_safe(model, expr_str=None):
    """Extract tree complexity with multiple fallback options - renamed depth to height"""
    height = None  # Changed from depth
    n_nodes = None
    
    # 1) sklearn tree interface
    try:
        if hasattr(model, "get_depth"):
            height = int(model.get_depth())
    except Exception:
        pass
    
    try:
        if hasattr(model, "tree_") and hasattr(model.tree_, "node_count"):
            n_nodes = int(model.tree_.node_count)
            if hasattr(model.tree_, "max_depth"):
                height = int(model.tree_.max_depth)
    except Exception:
        pass
    
    # 2) PSTree / other implementation interfaces
    for attr in ("max_depth", "depth_", "n_nodes_", "n_nodes", "node_count"):
        try:
            val = getattr(model, attr, None)
            if val is not None:
                if "depth" in attr and height is None:
                    height = int(val)
                if ("node" in attr or "nodes" in attr) and n_nodes is None:
                    n_nodes = int(val)
        except Exception:
            pass
    
    # 3) Expression-based estimation as last resort
    if (height is None or n_nodes is None) and isinstance(expr_str, str) and len(expr_str) > 0:
        try:
            # Token counting
            toks = re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|<=|>=|!=|[()^*/+\-]", expr_str)
            if n_nodes is None:
                n_nodes = max(1, len(toks) // 2)
            
            # Height from parenthesis nesting
            if height is None:
                max_height = 0
                current_depth = 0
                for char in expr_str:
                    if char == '(':
                        current_depth += 1
                        max_height = max(max_height, current_depth)
                    elif char == ')':
                        current_depth -= 1
                height = max(1, max_height)
        except Exception:
            pass
    
    # Ensure we return numbers, not None
    return height if height is not None else np.nan, n_nodes if n_nodes is not None else np.nan

def _filter_near_constant_features_safe(X_tr, X_te, names, var_eps=1e-12, uniq_ratio=0.01, whitelist=None):
    """Filter near-constant features with whitelist support and robust fallback"""
    if whitelist is None:
        whitelist = GATE_WHITELIST
    
    X_tr = np.asarray(X_tr)
    X_te = np.asarray(X_te)
    keep = []
    dropped_names = []
    variances = []  # Track variances for fallback
    
    for j in range(X_tr.shape[1]):
        name = names[j] if j < len(names) else f'X{j}'
        
        # Calculate variance for potential fallback
        col = X_tr[:, j]
        col_clean = col[~np.isnan(col)]
        var = np.nanvar(col_clean) if col_clean.size > 0 else 0
        variances.append(var)
        
        # Whitelist check
        if name in whitelist:
            keep.append(j)
            continue
        
        if col_clean.size == 0:
            dropped_names.append(name)
            continue
        
        # Check for near-constant
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
    
    # Robust fallback: if all filtered, keep the feature with highest variance
    if keep.size == 0:
        print(f"    [WARNING] All features were near-constant. Keeping feature with highest variance.")
        variances = np.array(variances)
        # Keep top 1-3 features by variance
        n_keep = min(3, len(variances))  # Keep up to 3 features for robustness
        keep = np.argsort(variances)[-n_keep:][::-1]  # Indices of top variance features
        keep = np.array(keep, dtype=int)
    
    X_tr_filtered = X_tr[:, keep]
    X_te_filtered = X_te[:, keep]
    names_filtered = [names[i] for i in keep]
    
    # Create drop mask for all original features
    all_indices = set(range(len(names)))
    keep_set = set(keep.tolist())
    drop_mask = [1 if i not in keep_set else 0 for i in range(len(names))]
    
    if dropped_names:
        print(f"    [INFO] Dropped {len(dropped_names)} near-constant features: {dropped_names[:5]}{'...' if len(dropped_names) > 5 else ''}")
    
    return X_tr_filtered, X_te_filtered, names_filtered, keep, drop_mask

def _expr_to_sympy_safe(expr_str):
    """Convert GP expression to sympy format with error handling"""
    try:
        if not expr_str or expr_str in ["N/A", "PSTree model", None, "Error extracting"]:
            return None
        
        # Check if sympy is available
        try:
            import sympy
        except ImportError:
            return None  # Sympy not available
        
        # Clean up the expression
        sympy_expr = str(expr_str)
        
        # Common function mappings
        replacements = [
            ('add(', '('),
            ('sub(', '('),  
            ('mul(', '('),
            ('div(', '('),
            ('sqrt(', 'sqrt('),
            ('log(', 'log('),
            ('exp(', 'exp('),
            ('abs(', 'Abs('),
            ('neg(', '-('),
            ('^', '**'),  # Power operator
        ]
        
        for old, new in replacements:
            sympy_expr = sympy_expr.replace(old, new)
        
        # Handle if_else/conditional
        sympy_expr = re.sub(
            r'if_else\((.*?),(.*?),(.*?)\)',
            r'Piecewise((\2, \1), (\3, True))',
            sympy_expr
        )
        
        return sympy_expr
        
    except Exception:
        return None

def _safe_expr_and_complexity(model: Any):
    """Extract complete expression from PSTree model with all leaf expressions"""
    try:
        expressions = []
        total_complexity = 0
        
        # Check if PSTree has multiple regression models (one per leaf)
        if hasattr(model, 'regr_list') and model.regr_list:
            # Multiple regressors case
            for i, regr in enumerate(model.regr_list):
                if hasattr(regr, '_expr_str'):
                    expr = regr._expr_str
                    expressions.append(f"Leaf_{i}: {expr}")
                    if hasattr(regr.gp, '_program') and hasattr(regr.gp._program, 'program'):
                        total_complexity += len(regr.gp._program.program)
                elif hasattr(regr, 'gp') and hasattr(regr.gp, '_program'):
                    expr = str(regr.gp._program)
                    # Map feature names
                    if hasattr(regr, '_map_expression_to_features'):
                        expr = regr._map_expression_to_features(expr)
                    expressions.append(f"Leaf_{i}: {expr}")
                    if hasattr(regr.gp._program, 'program'):
                        total_complexity += len(regr.gp._program.program)
        
        # Single regressor case
        elif hasattr(model, 'regr'):
            if hasattr(model.regr, '_expr_str'):
                expressions.append(model.regr._expr_str)
                if hasattr(model.regr, 'gp') and hasattr(model.regr.gp, '_program'):
                    if hasattr(model.regr.gp._program, 'program'):
                        total_complexity = len(model.regr.gp._program.program)
            elif hasattr(model.regr, 'gp') and hasattr(model.regr.gp, '_program'):
                expr = str(model.regr.gp._program)
                # Map feature names
                if hasattr(model.regr, '_map_expression_to_features'):
                    expr = model.regr._map_expression_to_features(expr)
                expressions.append(expr)
                if hasattr(model.regr.gp._program, 'program'):
                    total_complexity = len(model.regr.gp._program.program)
        
        # Return combined expression
        if expressions:
            if len(expressions) == 1:
                return expressions[0], total_complexity if total_complexity > 0 else None
            else:
                combined = f"PSTree[{len(expressions)} leaves]:\n" + "\n".join(expressions)
                return combined, total_complexity if total_complexity > 0 else None
        
        # Fallback
        if hasattr(model, 'regr_class'):
            return f"PSTree[{model.regr_class.__name__}]", None
        else:
            return "PSTree model", None
            
    except Exception as e:
        return f"Error extracting: {str(e)}", None

def _kfold_cv(X, y, build_fn, seed: int, k: int = 5):
    """K-fold cross validation with standardized metrics (median MSE, mean R², instability)"""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    mses, r2s = [], []
    
    for tr, va in kf.split(X):
        # Standardize within each fold - no data leakage
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])  # Fit only on training fold
        Xv = scaler.transform(X[va])      # Transform validation fold
        
        yt, yv = y[tr], y[va]  # y is not standardized
        
        m = build_fn()
        try:
            m.fit(Xt, yt)
            yh = m.predict(Xv)
            mses.append(mean_squared_error(yv, yh))
            r2s.append(r2_score(yv, yh))
        except Exception as e:
            print(f"    CV fold failed: {e}")
            mses.append(np.nan)
            r2s.append(np.nan)
    
    # Calculate instability
    EPS = 1e-12
    instability = float(np.std(mses) / max(np.mean(mses), EPS)) if mses else 1.0
    
    # Return median MSE, mean R², and instability (standardized with LGO)
    return float(np.nanmedian(mses)), float(np.nanmean(r2s)), instability

def run_pstree_once(X_tr, y_tr, X_te, y_te, seed, cfg: PSTreeConfig, feature_names: List[str] = None):
    """Run PSTree experiment once with all improvements"""
    global FEATURE_NAMES
    
    # Add initial message
    print(f"    [Seed {seed}] Starting PSTree experiment...")
    
    # Validate input
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X_tr.shape[1])]
    
    # Filter near-constant features with whitelist
    original_n_features = X_tr.shape[1]
    X_tr, X_te, feature_names, keep_idx, drop_mask = _filter_near_constant_features_safe(
        X_tr, X_te, feature_names, whitelist=GATE_WHITELIST
    )
    
    # Validation: ensure we have features and names are aligned
    assert X_tr.shape[1] == X_te.shape[1] == len(feature_names), \
        f"Feature count mismatch after filtering: train={X_tr.shape[1]}, test={X_te.shape[1]}, names={len(feature_names)}"
    
    assert X_tr.shape[1] > 0, "No features remaining after filtering"
    
    # Set global FEATURE_NAMES for expression mapping - CRITICAL: use filtered names!
    FEATURE_NAMES = feature_names
    
    PSTreeRegressor, backend = _import_pstree_regressor()
    t0 = time.time()

    row = {
        "experiment": "pstree",
        "engine": "pstree",
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
        "n_leaves": cfg.max_leaf_nodes,
        "expr_str": None,
        "expr_sympy": None,
        "candidates_csv": "",
        "rank": 1,
        "error": "",
        "engine_status": backend,
        "kept_features": json.dumps(list(feature_names)),
        "drop_mask": json.dumps(drop_mask)
    }

    # Standardize data for final model
    scaler_X = StandardScaler()
    Xtr_s = scaler_X.fit_transform(X_tr)
    Xte_s = scaler_X.transform(X_te)

    if PSTreeRegressor is None:
        row["error"] = f"pstree_import_failed: {backend}, fallback_to_sklearn_tree"
        # Fallback to sklearn DecisionTree
        try:
            model = DecisionTreeRegressor(max_depth=cfg.max_depth, random_state=seed)
            model.fit(Xtr_s, y_tr)
            yh = model.predict(Xte_s)
            row["test_loss"] = float(mean_squared_error(y_te, yh))
            row["test_r2"] = float(r2_score(y_te, yh))
            row["height"], row["n_nodes"] = _tree_complexity_safe(model)
            # >>> 新增：逐样本预测
            row["y_pred_test"] = [float(v) for v in np.asarray(yh).ravel()]
        except Exception as e:
            row["error"] += f"|fallback_failed: {e}"
        
        row["runtime_sec"] = round(time.time() - t0, 4)
        return row

    def build():
        """Build PSTree model with GP"""
        try:
            from gplearn.genetic import SymbolicRegressor
            
            # Create a factory function for GPLearnWrapper
            def create_gp_wrapper(**kwargs):
                wrapper = GPLearnWrapper(
                    population_size=cfg.pop_size,
                    generations=cfg.ngen,
                    random_state=seed
                )
                return wrapper
            
            return PSTreeRegressor(
                regr_class=create_gp_wrapper,
                tree_class=DecisionTreeRegressor,
                max_depth=cfg.max_depth,
                max_leaf_nodes=cfg.max_leaf_nodes,
                min_samples_leaf=cfg.min_samples_leaf,
                random_seed=seed
            )
        except ImportError:
            # Fallback to Ridge
            return PSTreeRegressor(
                regr_class=PSTreeCompatibleRegressor,
                tree_class=DecisionTreeRegressor,
                max_depth=cfg.max_depth,
                max_leaf_nodes=cfg.max_leaf_nodes,
                min_samples_leaf=cfg.min_samples_leaf,
                random_seed=seed
            )

    # Cross-validation with standardized metrics
    try:
        print(f"    [Seed {seed}] Running 5-fold CV...")
        cv_mse, cv_r2, instability = _kfold_cv(X_tr, y_tr, build, seed=seed, k=5)
        row["cv_loss"] = float(cv_mse)
        row["cv_r2"] = float(cv_r2)
        row["instability"] = float(instability)  # Added instability
        print(f"    [Seed {seed}] CV R² = {cv_r2:.4f}, Instability = {instability:.4f}")
    except Exception as e:
        row["error"] = f"cv_failed: {str(e)}"
        print(f"    [Seed {seed}] CV failed: {e}")

    # Final model training and testing
    try:
        model = build()
        print(f"    [Seed {seed}] Training final model (pop={cfg.pop_size}, gen={cfg.ngen})...")
        
        # Add progress tracking for long runs
        start_train = time.time()
        model.fit(Xtr_s, y_tr)
        train_time = time.time() - start_train

        print(f"    [Seed {seed}] Training completed in {train_time:.1f}s, evaluating...")
  
        yh = model.predict(Xte_s)
        row["test_loss"] = float(mean_squared_error(y_te, yh))
        row["test_r2"] = float(r2_score(y_te, yh))
        # >>> 新增：逐样本预测
        row["y_pred_test"] = [float(v) for v in np.asarray(yh).ravel()] 

        print(f"    [Seed {seed}] Test R² = {row['test_r2']:.4f}")
        
        # Get model expression
        expr, comp = _safe_expr_and_complexity(model)
        row["expr_str"] = expr
        if comp is not None:
            row["size"] = comp
        
        # Calculate weighted complexity
        row["complexity"] = calc_weighted_complexity(expr)
        
        # Show expression preview
        if expr and expr not in ["N/A", "PSTree model", "Error extracting"]:
            expr_preview = expr[:150] + "..." if len(expr) > 150 else expr
            print(f"    [Seed {seed}] Expression: {expr_preview}")

        # Get actual tree complexity with robust fallback
        real_height, n_nodes = _tree_complexity_safe(model, expr)
        if not np.isnan(real_height):
            row["height"] = int(real_height)  # Changed from depth
        if not np.isnan(n_nodes):
            row["n_nodes"] = int(n_nodes)
        
        # Convert to sympy format with error handling
        row["expr_sympy"] = _expr_to_sympy_safe(expr)
        if row["expr_sympy"] is None and expr and expr not in ["N/A", "PSTree model"]:
            # Note in error field if sympy conversion failed but don't break
            if not row["error"]:
                row["error"] = "sympy_conversion_failed"
            
    except Exception as e:
        row["error"] = (row["error"] + f"|fit_failed: {str(e)}").strip("|")

    row["runtime_sec"] = round(time.time() - t0, 4)
    return row