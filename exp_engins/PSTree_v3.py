# PSTree_v3.py - Fixed version with proper PSTree integration
# Key fixes:
# 1. Correctly use PSTreeRegressor native GP functionality
# 2. Add detailed logging to track execution
# 3. Ensure CV and training actually run with GP evolution
# on Dec 2, 2025


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
from sklearn.tree import DecisionTreeRegressor

def _import_pstree_regressor():
    """Import PSTreeRegressor from the correct location"""
    try:
        from pstree.cluster_gp_sklearn import PSTreeRegressor
        return PSTreeRegressor, "pstree.cluster_gp_sklearn"
    except ImportError as e:
        return None, f"import_failed: {str(e)}"

def _import_gplearn():
    """Import gplearn SymbolicRegressor"""
    try:
        from gplearn.genetic import SymbolicRegressor
        return SymbolicRegressor, "gplearn.genetic"
    except ImportError as e:
        return None, f"import_failed: {str(e)}"

# Global variable to store feature names for expression mapping
FEATURE_NAMES = []

# Gate/binary features that should not be filtered even if imbalanced
GATE_WHITELIST = {
    "gender_std", "age_band", "mechanical_ventilation_std", 
    "vasopressor_use_std", "DS", "sex", "is_male", "is_female"
}

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
        if op in ['+', '-', '*', '/']:
            complexity += expr_lower.count(op) * weight
        else:
            pattern = r'\b' + re.escape(op) + r'\b'
            complexity += len(re.findall(pattern, expr_lower)) * weight
    
    # Add complexity for variables
    variables = set(re.findall(r'\b[A-Za-z_]\w*\b', expr_str))
    variables = {v for v in variables if v.lower() not in weights.keys()}
    complexity += len(variables) * 0.5
    
    return max(1.0, complexity)


class GPLearnRegressor:
    """
    Wrapper for gplearn.SymbolicRegressor that is compatible with PSTree's regr_class interface.
    
    PSTree expects regr_class to be instantiated as: regr_class(**params)
    and have fit(X, y) and predict(X) methods.
    """
    
    # Class-level configuration (will be set before instantiation)
    _population_size = 100
    _generations = 20
    _random_state = None
    _verbose = 0
    
    @classmethod
    def configure(cls, population_size=100, generations=20, random_state=None, verbose=0):
        """Configure class-level parameters before PSTree uses it"""
        cls._population_size = population_size
        cls._generations = generations
        cls._random_state = random_state
        cls._verbose = verbose
    
    def __init__(self, **kwargs):
        """Initialize wrapper - gplearn regressor created on fit()"""
        from gplearn.genetic import SymbolicRegressor
        
        self.gp = SymbolicRegressor(
            population_size=self._population_size,
            generations=self._generations,
            tournament_size=min(20, self._population_size // 5),
            init_depth=(2, 4),
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs'),
            parsimony_coefficient=0.01,
            p_crossover=0.9,
            p_subtree_mutation=0.05,
            p_hoist_mutation=0.01,
            p_point_mutation=0.04,
            verbose=self._verbose,
            random_state=self._random_state,
            n_jobs=1,  # Single thread to avoid issues
            low_memory=False
        )
        
        self._fitted = False
        self._expr_str = None
        self.n_features_ = None
        
        # For PSTree compatibility
        self.best_pop = None
        self.original_features = "original"
        self.adaptive_tree = True
        
        # Store any extra kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y, category=None):
        """Fit the GP regressor"""
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if self._verbose >= 1:
            print(f"        [GPLearn] Fitting on {X.shape[0]} samples, {X.shape[1]} features")
            print(f"        [GPLearn] pop={self._population_size}, gen={self._generations}")
        
        t0 = time.time()
        self.gp.fit(X, y)
        elapsed = time.time() - t0
        
        self._fitted = True
        self.n_features_ = X.shape[1]
        self.best_pop = self
        
        # Extract expression
        if hasattr(self.gp, '_program'):
            self._expr_str = self._map_expression_to_features(str(self.gp._program))
            if self._verbose >= 1:
                expr_preview = self._expr_str[:80] + "..." if len(self._expr_str) > 80 else self._expr_str
                print(f"        [GPLearn] Done in {elapsed:.1f}s: {expr_preview}")
        
        return self
    
    def predict(self, X, y=None, category=None):
        """Predict using fitted GP"""
        if not self._fitted:
            raise RuntimeError("GPLearnRegressor not fitted yet")
        return self.gp.predict(np.asarray(X))
    
    def score(self, X, y):
        """R² score"""
        return self.gp.score(np.asarray(X), np.asarray(y).ravel())
    
    def feature_synthesis(self, X, best_pop=None, original_features=None):
        """Required for PSTree interface"""
        return np.asarray(X)
    
    def _map_expression_to_features(self, expr_str):
        """Map X0, X1, etc. to actual feature names"""
        global FEATURE_NAMES
        if FEATURE_NAMES:
            for i, fname in enumerate(FEATURE_NAMES):
                expr_str = expr_str.replace(f'X{i}', fname)
        return expr_str
    
    def get_expression(self):
        """Return the symbolic expression"""
        return self._expr_str if self._expr_str else "N/A"


class RidgeRegressor:
    """Fallback Ridge regressor compatible with PSTree interface"""
    
    def __init__(self, **kwargs):
        self.model = Ridge(alpha=1.0)
        self.best_pop = None
        self.original_features = "original"
        self.adaptive_tree = True
        self.n_features_ = None
        self._expr_str = None
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y, category=None):
        self.model.fit(X, y)
        self.n_features_ = X.shape[1]
        self.best_pop = self
        
        # Create expression from coefficients
        global FEATURE_NAMES
        coefs = self.model.coef_
        intercept = self.model.intercept_
        
        terms = []
        for i, c in enumerate(coefs):
            if abs(c) > 1e-10:
                fname = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'X{i}'
                terms.append(f"{c:.4f}*{fname}")
        
        if terms:
            self._expr_str = " + ".join(terms)
            if abs(intercept) > 1e-10:
                self._expr_str += f" + {intercept:.4f}"
        else:
            self._expr_str = f"{intercept:.4f}"
        
        return self
    
    def predict(self, X, y=None, category=None):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def feature_synthesis(self, X, best_pop=None, original_features=None):
        return X
    
    def get_expression(self):
        return self._expr_str if self._expr_str else "Ridge model"


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
    """Extract tree complexity with multiple fallback options"""
    height = None
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
            toks = re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|<=|>=|!=|[()^*/+\-]", expr_str)
            if n_nodes is None:
                n_nodes = max(1, len(toks) // 2)
            
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
    
    return height if height is not None else np.nan, n_nodes if n_nodes is not None else np.nan


def _filter_near_constant_features_safe(X_tr, X_te, names, var_eps=1e-12, uniq_ratio=0.01, whitelist=None):
    """Filter near-constant features with whitelist support"""
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
        
        if name in whitelist:
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
        print(f"    [WARNING] All features were near-constant. Keeping feature with highest variance.")
        variances = np.array(variances)
        n_keep = min(3, len(variances))
        keep = np.argsort(variances)[-n_keep:][::-1]
        keep = np.array(keep, dtype=int)
    
    X_tr_filtered = X_tr[:, keep]
    X_te_filtered = X_te[:, keep]
    names_filtered = [names[i] for i in keep]
    
    drop_mask = [1 if i not in set(keep.tolist()) else 0 for i in range(len(names))]
    
    if dropped_names:
        print(f"    [INFO] Dropped {len(dropped_names)} near-constant features: {dropped_names[:5]}{'...' if len(dropped_names) > 5 else ''}")
    
    return X_tr_filtered, X_te_filtered, names_filtered, keep, drop_mask


def _expr_to_sympy_safe(expr_str):
    """Convert GP expression to sympy format"""
    try:
        if not expr_str or expr_str in ["N/A", "PSTree model", None, "Error extracting"]:
            return None
        
        try:
            import sympy
        except ImportError:
            return None
        
        sympy_expr = str(expr_str)
        
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
            ('^', '**'),
        ]
        
        for old, new in replacements:
            sympy_expr = sympy_expr.replace(old, new)
        
        sympy_expr = re.sub(
            r'if_else\((.*?),(.*?),(.*?)\)',
            r'Piecewise((\2, \1), (\3, True))',
            sympy_expr
        )
        
        return sympy_expr
        
    except Exception:
        return None


def _safe_expr_and_complexity(model: Any):
    """Extract complete expression from PSTree model"""
    try:
        expressions = []
        total_complexity = 0
        
        # Check if PSTree has multiple regression models (one per leaf)
        if hasattr(model, 'regr_list') and model.regr_list:
            for i, regr in enumerate(model.regr_list):
                expr = None
                
                # Try get_expression method first
                if hasattr(regr, 'get_expression'):
                    expr = regr.get_expression()
                elif hasattr(regr, '_expr_str') and regr._expr_str:
                    expr = regr._expr_str
                elif hasattr(regr, 'gp') and hasattr(regr.gp, '_program'):
                    expr = str(regr.gp._program)
                    if hasattr(regr, '_map_expression_to_features'):
                        expr = regr._map_expression_to_features(expr)
                
                if expr and expr not in ["N/A", None]:
                    expressions.append(f"Leaf_{i}: {expr}")
                    
                # Count complexity
                if hasattr(regr, 'gp') and hasattr(regr.gp, '_program'):
                    if hasattr(regr.gp._program, 'program'):
                        total_complexity += len(regr.gp._program.program)
        
        # Single regressor case
        elif hasattr(model, 'regr'):
            regr = model.regr
            expr = None
            
            if hasattr(regr, 'get_expression'):
                expr = regr.get_expression()
            elif hasattr(regr, '_expr_str') and regr._expr_str:
                expr = regr._expr_str
            elif hasattr(regr, 'gp') and hasattr(regr.gp, '_program'):
                expr = str(regr.gp._program)
            
            if expr:
                expressions.append(expr)
        
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


def _kfold_cv(X, y, build_fn, seed: int, k: int = 5, verbose: bool = True):
    """K-fold cross validation with standardized metrics"""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    mses, r2s = [], []
    
    for fold_idx, (tr, va) in enumerate(kf.split(X)):
        if verbose:
            print(f"      [CV] Fold {fold_idx + 1}/{k}...")
        
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        
        yt, yv = y[tr], y[va]
        
        m = build_fn()
        try:
            m.fit(Xt, yt)
            yh = m.predict(Xv)
            fold_mse = mean_squared_error(yv, yh)
            fold_r2 = r2_score(yv, yh)
            mses.append(fold_mse)
            r2s.append(fold_r2)
            if verbose:
                print(f"      [CV] Fold {fold_idx + 1}: MSE={fold_mse:.6f}, R²={fold_r2:.4f}")
        except Exception as e:
            print(f"      [CV] Fold {fold_idx + 1} failed: {e}")
            mses.append(np.nan)
            r2s.append(np.nan)
    
    EPS = 1e-12
    instability = float(np.std(mses) / max(np.mean(mses), EPS)) if mses else 1.0
    
    return float(np.nanmedian(mses)), float(np.nanmean(r2s)), instability


def run_pstree_once(X_tr, y_tr, X_te, y_te, seed, cfg: PSTreeConfig, feature_names: List[str] = None):
    """
    Run PSTree experiment once with proper GP integration.
    
    Key changes from v2.2:
    1. Configure GPLearnRegressor class before PSTree uses it
    2. Add detailed logging throughout
    3. Better error handling and fallbacks
    """
    global FEATURE_NAMES
    
    print(f"    [Seed {seed}] Starting PSTree experiment...")
    print(f"    [Seed {seed}] Config: max_depth={cfg.max_depth}, max_leaf_nodes={cfg.max_leaf_nodes}, pop={cfg.pop_size}, ngen={cfg.ngen}")
    
    # Validate input
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X_tr.shape[1])]
    
    print(f"    [Seed {seed}] Input: {X_tr.shape[0]} train samples, {X_te.shape[0]} test samples, {len(feature_names)} features")
    
    # Filter near-constant features
    original_n_features = X_tr.shape[1]
    X_tr, X_te, feature_names, keep_idx, drop_mask = _filter_near_constant_features_safe(
        X_tr, X_te, feature_names, whitelist=GATE_WHITELIST
    )
    
    if X_tr.shape[1] != original_n_features:
        print(f"    [Seed {seed}] Filtered to {X_tr.shape[1]} features (from {original_n_features})")
    
    # Validation
    assert X_tr.shape[1] == X_te.shape[1] == len(feature_names), \
        f"Feature count mismatch: train={X_tr.shape[1]}, test={X_te.shape[1]}, names={len(feature_names)}"
    assert X_tr.shape[1] > 0, "No features remaining after filtering"
    
    # Set global FEATURE_NAMES for expression mapping
    FEATURE_NAMES = feature_names
    
    PSTreeRegressor, backend = _import_pstree_regressor()
    SymbolicRegressor, gplearn_backend = _import_gplearn()
    
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
        "height": np.nan,
        "complexity": np.nan,
        "instability": np.nan,
        "n_nodes": np.nan,
        "n_leaves": cfg.max_leaf_nodes,
        "expr_str": None,
        "expr_sympy": None,
        "candidates_csv": "",
        "rank": 1,
        "error": "",
        "engine_status": f"pstree={backend}, gplearn={gplearn_backend}",
        "kept_features": json.dumps(list(feature_names)),
        "drop_mask": json.dumps(drop_mask)
    }
    
    # Standardize data
    scaler_X = StandardScaler()
    Xtr_s = scaler_X.fit_transform(X_tr)
    Xte_s = scaler_X.transform(X_te)
    
    print(f"    [Seed {seed}] Data standardized")
    
    # Determine which regressor to use
    use_gplearn = (PSTreeRegressor is not None) and (SymbolicRegressor is not None)
    
    if not use_gplearn:
        if PSTreeRegressor is None:
            row["error"] = f"pstree_import_failed: {backend}"
            print(f"    [Seed {seed}] ERROR: PSTree import failed: {backend}")
        if SymbolicRegressor is None:
            row["error"] = (row["error"] + f"|gplearn_import_failed: {gplearn_backend}").strip("|")
            print(f"    [Seed {seed}] ERROR: GPLearn import failed: {gplearn_backend}")
    
    # Fallback to sklearn DecisionTree if PSTree unavailable
    if PSTreeRegressor is None:
        print(f"    [Seed {seed}] Falling back to sklearn DecisionTreeRegressor")
        try:
            model = DecisionTreeRegressor(max_depth=cfg.max_depth, random_state=seed)
            model.fit(Xtr_s, y_tr)
            yh = model.predict(Xte_s)
            row["test_loss"] = float(mean_squared_error(y_te, yh))
            row["test_r2"] = float(r2_score(y_te, yh))
            row["height"], row["n_nodes"] = _tree_complexity_safe(model)
            row["y_pred_test"] = [float(v) for v in np.asarray(yh).ravel()]
            row["expr_str"] = "DecisionTree (fallback)"
        except Exception as e:
            row["error"] += f"|fallback_failed: {e}"
        
        row["runtime_sec"] = round(time.time() - t0, 4)
        print(f"    [Seed {seed}] Completed (fallback) in {row['runtime_sec']:.2f}s")
        return row
    
    # Configure GPLearnRegressor class with our parameters
    if use_gplearn:
        GPLearnRegressor.configure(
            population_size=cfg.pop_size,
            generations=cfg.ngen,
            random_state=seed,
            verbose=1  # Enable verbose for debugging
        )
        regr_class = GPLearnRegressor
        print(f"    [Seed {seed}] Using GPLearnRegressor (pop={cfg.pop_size}, gen={cfg.ngen})")
    else:
        regr_class = RidgeRegressor
        print(f"    [Seed {seed}] Using RidgeRegressor (gplearn unavailable)")
    
    def build():
        """Build PSTree model"""
        return PSTreeRegressor(
            regr_class=regr_class,
            tree_class=DecisionTreeRegressor,
            max_depth=cfg.max_depth,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            random_seed=seed
        )
    
    # Cross-validation
    try:
        print(f"    [Seed {seed}] Running 5-fold CV...")
        cv_mse, cv_r2, instability = _kfold_cv(X_tr, y_tr, build, seed=seed, k=5, verbose=True)
        row["cv_loss"] = float(cv_mse)
        row["cv_r2"] = float(cv_r2)
        row["instability"] = float(instability)
        print(f"    [Seed {seed}] CV Complete: R²={cv_r2:.4f}, MSE={cv_mse:.6f}, Instability={instability:.4f}")
    except Exception as e:
        row["error"] = f"cv_failed: {str(e)}"
        print(f"    [Seed {seed}] CV failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final model training
    try:
        model = build()
        print(f"    [Seed {seed}] Training final model...")
        
        start_train = time.time()
        model.fit(Xtr_s, y_tr)
        train_time = time.time() - start_train
        
        print(f"    [Seed {seed}] Training completed in {train_time:.1f}s")
        
        # Evaluate
        yh = model.predict(Xte_s)
        row["test_loss"] = float(mean_squared_error(y_te, yh))
        row["test_r2"] = float(r2_score(y_te, yh))
        row["y_pred_test"] = [float(v) for v in np.asarray(yh).ravel()]
        
        print(f"    [Seed {seed}] Test R² = {row['test_r2']:.4f}, RMSE = {np.sqrt(row['test_loss']):.4f}")
        
        # Get expression
        expr, comp = _safe_expr_and_complexity(model)
        row["expr_str"] = expr
        if comp is not None:
            row["size"] = comp
        
        row["complexity"] = calc_weighted_complexity(expr)
        
        # Show expression
        if expr and expr not in ["N/A", "PSTree model", "Error extracting"]:
            expr_preview = expr[:150] + "..." if len(expr) > 150 else expr
            print(f"    [Seed {seed}] Expression: {expr_preview}")
        
        # Tree complexity
        real_height, n_nodes = _tree_complexity_safe(model, expr)
        if not np.isnan(real_height):
            row["height"] = int(real_height)
        if not np.isnan(n_nodes):
            row["n_nodes"] = int(n_nodes)
        
        row["expr_sympy"] = _expr_to_sympy_safe(expr)
        
    except Exception as e:
        row["error"] = (row["error"] + f"|fit_failed: {str(e)}").strip("|")
        print(f"    [Seed {seed}] Final model training failed: {e}")
        import traceback
        traceback.print_exc()
    
    row["runtime_sec"] = round(time.time() - t0, 4)
    print(f"    [Seed {seed}] PSTree experiment completed in {row['runtime_sec']:.2f}s")
    
    return row


# For compatibility with run_v3_9.py imports
__all__ = ['run_pstree_once', 'PSTreeConfig']