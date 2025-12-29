#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LGO vs AutoScore Fair Comparison (v2.5) - Core Engine
======================================================
Version: 2.5.0
Date: Dec 8, 2025

Thin core script: only runs experiments and writes CSV summaries.
For visualization, use run_comparison_plot_v1.py.

v2.5: Changes from v2.4:
1. LGO engine change to LGO v2.2 (paper engine v2.1 + external calibration).
2. Fix: when --binarize_threshold is provided and task is "auto"/"binary",
   we always binarize the outcome first and treat the task as binary.
   This is important for ordinal risk scores (e.g., 0..8 ICU composite scores).

v2.4: Changes from v2.3:
1. Multi-seed experiment now passes arguments explicitly (n_bootstrap, fair_features),
   avoiding hard-coded values and hidden dependencies on global CLI args.
2. Added CLI option --max_height to control LGO expression tree height.

v2.3: Changes from v2.2:
------------------
1. 移除所有可视化代码（移至run_comparison_plot_v1.py）
2. 代码量减少，专注于核心实验功能和CSV输出

Usage (examples):
-----------------
# 1) Full multi-seed (LGO uses all features, AutoScore uses top-n variables)
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_30k_full

python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_100k_full

python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 200 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_200k_full

python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 300 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_300k_full

python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 500 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method platt \
  --output_dir ICU_results_500k_full

# 2) Fair multi-seed (LGO also restricted to top-n variables)
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --fair_features \
  --calibration_method platt \
  --output_dir ICU_results_30k_fair

# 3) Turn off calibration (LGO raw logits vs AutoScore)
python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method none \
  --output_dir ICU_results_30k_raw

python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method none \
  --output_dir ICU_results_100k_raw

python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 200 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method none \
  --output_dir ICU_results_200k_raw

python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 300 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method none \
  --output_dir ICU_results_300k_raw  

python run_comparison_v2_5.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 500 \
  --population_size 1000 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --n_variables 6 \
  --calibration_method none \
  --output_dir ICU_results_500k_raw
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# =============================================================================
# Import LGO v2.2 (paper engine + external calibration)
# =============================================================================
try:
    # v2.2: run_lgo_sr_v2 核心逻辑与 paper 中 v2.1 相同，
    # 但在 LGOFairComparison 里通过 calibration_method 做外层校准。
    from LGO_v2_2 import run_lgo_sr_v2 as run_lgo_sr_v3, ZScaler
    LGO_AVAILABLE = True
    print("[INFO] Using LGO_v2_2 (DEAP v2.1 engine with external calibration)")
except ImportError:
    LGO_AVAILABLE = False
    print("[WARNING] LGO_v2_2 not found. LGO experiments will be disabled.")

# =============================================================================
# Import AutoScore v2 (R-compatible)
# =============================================================================
try:
    from AutoScore_v2 import AutoScore, AutoScoreConfig, get_autoscore
    AUTOSCORE_AVAILABLE = True
    print("[INFO] Using AutoScore_v2.py")
except ImportError:
    try:
        from LGO_AutoScore_v3_7.archive.autoscore_engine import AutoScore, AutoScoreConfig, get_autoscore
        AUTOSCORE_AVAILABLE = True
        print("[WARNING] AutoScore_v2.py not found, using autoscore_engine.py")
    except ImportError:
        AUTOSCORE_AVAILABLE = False
        print("[WARNING] AutoScore not available.")


# =============================================================================
# Enhanced LGO Wrapper with Feature Selection and Calibration
# =============================================================================
class LGOFairComparison:
    """LGO wrapper for fair/ablation comparison with AutoScore.

    This class:
      - optionally performs feature ranking and selection (top-n variables),
      - calls the underlying LGO symbolic regression engine (run_lgo_sr_v3),
      - supports optional probability calibration (Platt or isotonic),
      - exposes a sklearn-like interface (fit/predict_proba/predict).

    Parameters
    ----------
    gate_type : {"hard", "soft"}
        Type of logical gates used by LGO.
    n_generations : int
        Number of evolutionary generations (controls budget).
    population_size : int
        Population size per generation (controls budget).
    n_variables : int or None
        If not None, restrict to top-n features ranked by RandomForest importance.
    max_height : int
        Maximum expression tree height in LGO.
    random_state : int
        Random seed for reproducibility.
    calibration_method : {"platt", "isotonic", "none"}
        External calibration method applied on raw LGO outputs.
    """

    def __init__(
        self,
        gate_type: str = 'hard',
        n_generations: int = 100,
        population_size: int = 300,
        n_variables: Optional[int] = None,
        max_height: int = 10,
        random_state: int = 42,
        verbose: bool = True,
        calibration_method: str = 'platt'
    ):
        self.gate_type = gate_type
        self.n_generations = n_generations
        self.population_size = population_size
        self.n_variables = n_variables
        self.max_height = max_height
        self.random_state = random_state
        self.verbose = verbose
        self.calibration_method = calibration_method

        self.selected_features: Optional[List[str]] = None
        self.feature_ranking: Optional[pd.Series] = None
        self.scaler: Optional[ZScaler] = None
        self._result_df: Optional[pd.DataFrame] = None
        self._full_feature_names: Optional[List[str]] = None
        self._calibrator = None
        self._is_calibrated: bool = False

    def _rank_features(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        """Rank features using Random Forest importance."""
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=X.columns)
        self.feature_ranking = importance.sort_values(ascending=False)
        return list(self.feature_ranking.index)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'LGOFairComparison':
        """Fit LGO model with optional feature selection.

        Notes
        -----
        - If ``n_variables`` is not None, the top-n features are selected via RF ranking.
        - LGO is then run on the selected feature subset.
        """
        if not LGO_AVAILABLE:
            raise RuntimeError("LGO is not available. Please check imports.")

        if self.verbose:
            print(f"\n[LGO] Training with gate_type={self.gate_type}")

        self._full_feature_names = list(X.columns)
        y_arr = y.values if isinstance(y, pd.Series) else y

        if self.n_variables is not None and self.n_variables < len(X.columns):
            if self.verbose:
                print(f"[LGO] Selecting top {self.n_variables} features")
            ranked_features = self._rank_features(X, y_arr)
            self.selected_features = ranked_features[:self.n_variables]
            X_selected = X[self.selected_features]
            if self.verbose:
                print(f"[LGO] Selected: {self.selected_features}")
        else:
            self.selected_features = list(X.columns)
            X_selected = X

        X_arr = X_selected.values
        experiment = 'lgo_hard' if self.gate_type == 'hard' else 'lgo_soft'

        # Run the underlying symbolic regression engine
        self._result_df = run_lgo_sr_v3(
            X=X_arr,
            y=y_arr,
            feature_names=self.selected_features,
            experiment=experiment,
            typed_mode='light',
            pop_size=self.population_size,
            ngen=self.n_generations,
            max_height=self.max_height,
            random_state=self.random_state,
        )

        # The best expression is stored as a string in the result dataframe
        if self.verbose:
            print("[LGO] Training completed. Available columns:")
            print(self._result_df.columns)

        return self

    def get_formula(self) -> str:
        """Return the best LGO formula expression as a string."""
        if self._result_df is None or 'expr' not in self._result_df.columns:
            return ""
        best = self._result_df.iloc[0]
        return str(best['expr'])

    def get_complexity(self) -> int:
        """Return model complexity (e.g., expression size) if available."""
        if self._result_df is None or 'size' not in self._result_df.columns:
            return 0
        return int(self._result_df.iloc[0]['size'])

    def _evaluate_raw_expression(self, X: pd.DataFrame) -> np.ndarray:
        """Evaluate the raw symbolic expression on new data.

        IMPORTANT
        ---------
        LGO_v2_2 trains on *z-scored* features (via ``ZScaler`` inside
        ``run_lgo_sr_v2``). To obtain consistent predictions, we must apply
        the *same* standardization at inference time.

        This helper therefore:
          1) Reconstructs the training scaler from ``scaler_json`` stored in
             the LGO result dataframe (preferred), or
          2) Falls back to fitting a fresh ``ZScaler`` on the current X
             (which is mathematically equivalent if X is the original
             training data).
        """
        if self._result_df is None:
            raise RuntimeError("LGO has not been fit yet.")

        expr_str = self.get_formula()
        if not expr_str:
            return np.zeros(len(X))

        if self.selected_features is None:
            raise RuntimeError("selected_features is None; LGOFairComparison was not fitted correctly.")

        # Import low-level primitives from the LGO engine
        from LGO_v2_2 import (
            sdiv, ssqrt, slog, spow,
            lgo, lgo_thre, lgo_and2, lgo_or2, lgo_and3, gate_expr,
            as_pos_func, as_thr_func,
        )

        # Re-create the evaluation namespace for the expression
        ns: Dict[str, Any] = {}
        ns.update({
            'np': np,
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': sdiv,
            'sqrt': ssqrt,
            'log': slog,
            'pow': spow,
            'id': lambda x: x,
            'idF': lambda x: x,
            'lgo': lgo,
            'lgo_thre': lgo_thre,
            'lgo_and2': lgo_and2,
            'lgo_or2': lgo_or2,
            'lgo_and3': lgo_and3,
            'gate_expr': gate_expr,
            'as_pos': as_pos_func,
            'as_thr': as_thr_func,
            'zero': 0.0,
            'one': 1.0,
        })

        # ------------------------------------------------------------------
        # 1) Ensure we have a scaler consistent with training-time z-scoring
        # ------------------------------------------------------------------
        if self.scaler is None:
            scaler_obj = None

            # Preferred: reconstruct from JSON stored in the result dataframe
            if self._result_df is not None and 'scaler_json' in self._result_df.columns:
                scaler_json = self._result_df.iloc[0].get('scaler_json', None)
                if isinstance(scaler_json, str) and scaler_json:
                    try:
                        params = json.loads(scaler_json)
                        mean = np.asarray(params.get('mean'), dtype=float)
                        std = np.asarray(params.get('std'), dtype=float)
                        scaler_obj = ZScaler(mean_=mean, std_=std)
                    except Exception as e:
                        if self.verbose:
                            print(f"[LGO] Failed to parse scaler_json, will refit scaler: {e}")

            # Fallback: fit a new scaler on current X (same formula as training)
            if scaler_obj is None:
                X_arr_for_fit = X[self.selected_features].values
                scaler_obj = ZScaler.fit(X_arr_for_fit)

            self.scaler = scaler_obj

        # ------------------------------------------------------------------
        # 2) Apply scaling and evaluate the expression
        # ------------------------------------------------------------------
        X_arr = X[self.selected_features].values
        X_scaled = self.scaler.transform(X_arr)

        # Make (scaled) feature columns available in the evaluation namespace
        for i, col in enumerate(self.selected_features):
            ns[col] = X_scaled[:, i]

        try:
            y_raw = eval(expr_str, {"__builtins__": {}}, ns)
            y_raw = np.asarray(y_raw, dtype=float)
            y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=10.0, neginf=-10.0)
            return y_raw
        except Exception as e:
            if self.verbose:
                print(f"[LGO] Raw output error: {e}")
            return np.zeros(len(X))

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series) -> 'LGOFairComparison':
        """Calibrate using Platt scaling or isotonic regression."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        if self.calibration_method == 'none':
            if self.verbose:
                print("[LGO] Calibration disabled (calibration_method='none')")
            return self

        y_raw = self._evaluate_raw_expression(X_val)

        if self.calibration_method == 'platt':
            self._calibrator = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
            )
            self._calibrator.fit(y_raw.reshape(-1, 1), y_val)
        elif self.calibration_method == 'isotonic':
            self._calibrator = IsotonicRegression(
                out_of_bounds='clip'
            )
            self._calibrator.fit(y_raw, y_val)
        else:
            raise ValueError(f"Unknown calibration_method={self.calibration_method}")

        self._is_calibrated = True
        if self.verbose:
            print(f"[LGO] Calibration fitted with method={self.calibration_method}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities for binary classification."""
        if self._result_df is None:
            raise RuntimeError("LGO has not been fit yet.")

        X = X[self.selected_features]
        y_raw = self._evaluate_raw_expression(X)

        if self._calibrator is None or not self._is_calibrated:
            # Fallback: map raw scores via logistic
            probs = 1.0 / (1.0 + np.exp(-y_raw))
        else:
            if self.calibration_method == 'platt':
                probs = self._calibrator.predict_proba(y_raw.reshape(-1, 1))[:, 1]
            else:
                probs = self._calibrator.predict(y_raw)
        return probs

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels with a given probability threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_results_summary(self) -> Dict[str, Any]:
        """Return a compact summary dict of the fitted LGO model."""
        if self._result_df is None or self.selected_features is None:
            return {
                'expr': '',
                'size': np.nan,
                'n_features_used': 0,
            }

        row = self._result_df.iloc[0]
        return {
            'expr': row.get('expr', ''),
            'size': row.get('size', np.nan),
            'n_features_used': len(self.selected_features) if self.selected_features else 0,
        }


# =============================================================================
# Metrics and Utilities
# =============================================================================
def detect_task_type(y: np.ndarray) -> str:
    """Detect task type from target variable."""
    y = np.asarray(y).ravel()
    unique_vals = np.unique(y[~np.isnan(y)])
    n_unique = len(unique_vals)

    if n_unique == 2:
        return 'binary'
    elif n_unique <= 10 and np.allclose(unique_vals, unique_vals.astype(int)):
        return 'multiclass'
    else:
        return 'regression'


def binarize_target(y: np.ndarray, threshold: Union[float, str] = 'median') -> np.ndarray:
    """Convert continuous target to binary.

    If y is already {0,1}, we simply return it.
    Otherwise we threshold at a numeric value, or median/mean.
    """
    y = np.asarray(y).ravel()
    unique_vals = np.unique(y[~np.isnan(y)])

    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        print(f"[INFO] Data is already binary. Skipping binarization.")
        return y.astype(int)

    if threshold == 'median':
        thresh_val = np.nanmedian(y)
    elif threshold == 'mean':
        thresh_val = np.nanmean(y)
    else:
        thresh_val = float(threshold)

    print(f"[INFO] Binarizing target with threshold={thresh_val:.4f}")
    return (y >= thresh_val).astype(int)


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute AUROC, AUPRC, Brier, F1, Accuracy for binary classification."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    try:
        metrics['AUROC'] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics['AUROC'] = np.nan

    try:
        metrics['AUPRC'] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics['AUPRC'] = np.nan

    try:
        metrics['Brier'] = brier_score_loss(y_true, y_prob)
    except Exception:
        metrics['Brier'] = np.nan

    try:
        metrics['F1'] = f1_score(y_true, y_pred)
    except Exception:
        metrics['F1'] = np.nan

    try:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics['Accuracy'] = np.nan

    return metrics


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for AUROC.

    Returns
    -------
    mean_auc : float
    lower : float
    upper : float
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            auc = roc_auc_score(y_true[idx], y_prob[idx])
            aucs.append(auc)
        except Exception:
            continue

    if not aucs:
        return np.nan, np.nan, np.nan

    aucs = np.array(aucs)
    mean_auc = float(np.mean(aucs))
    lower = float(np.percentile(aucs, 100 * alpha / 2))
    upper = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return mean_auc, lower, upper


def split_data(
    data: pd.DataFrame,
    outcome_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    task: str = 'auto',
    binarize_threshold: Optional[Union[float, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Split data into train/validation/test sets with optional binarization.

    Notes
    -----
    - If ``binarize_threshold`` is provided and ``task`` is "auto" or "binary",
      we first convert the outcome into a binary label (e.g., score >= 5),
      and then force the task type to "binary".
    - This behavior is important for ordinal risk scores such as 0..8,
      where we want to define a clinically meaningful binary endpoint.
    """
    y = data[outcome_col].values

    # If user explicitly requests binarization (common for ICU risk scores),
    # we apply it before inferring the task, and treat the task as binary.
    if binarize_threshold is not None and task in ('auto', 'binary'):
        data = data.copy()
        data[outcome_col] = binarize_target(y, threshold=binarize_threshold)
        y = data[outcome_col].values
        inferred_task = 'binary'
    else:
        inferred_task = detect_task_type(y) if task == 'auto' else task

    print(f"[DATA] Inferred task type: {inferred_task}")

    if inferred_task == 'binary':
        stratify = y
        if len(np.unique(y)) < 2:
            print("[WARN] Only one class present. Stratified split disabled.")
            stratify = None
    else:
        stratify = None

    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Second split: train vs val (from train+val)
    if val_size > 0:
        relative_val_size = val_size / (1.0 - test_size)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=train_val_data[outcome_col] if stratify is not None else None,
        )
    else:
        train_data = train_val_data
        val_data = pd.DataFrame(columns=data.columns)

    print(f"[DATA] Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data, inferred_task


def load_data(data_path: Optional[str] = None, n_samples: int = 1000) -> pd.DataFrame:
    """Load real data from CSV or generate synthetic data for testing."""
    if data_path is None:
        print("[DATA] No data_path provided. Generating synthetic dataset...")
        return generate_synthetic_data(n_samples=n_samples)

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"[DATA] Loading data from {data_path}")
    df = pd.read_csv(path)
    print(f"[DATA] Loaded shape: {df.shape}")
    return df


def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic ICU-like dataset with a binary outcome."""
    rng = np.random.RandomState(random_state)
    data = pd.DataFrame({
        'age': rng.normal(65, 10, n_samples).clip(18, 90),
        'heart_rate': rng.normal(90, 15, n_samples).clip(40, 180),
        'sbp': rng.normal(110, 20, n_samples).clip(60, 200),
        'dbp': rng.normal(70, 10, n_samples).clip(40, 120),
        'resp_rate': rng.normal(20, 4, n_samples).clip(8, 40),
        'spo2': rng.normal(96, 3, n_samples).clip(70, 100),
        'temperature': rng.normal(37.0, 0.7, n_samples).clip(35, 41),
        'wbc': np.random.lognormal(2.3, 0.5, n_samples).clip(1, 50),
        'creatinine': np.random.lognormal(-0.2, 0.5, n_samples).clip(0.2, 10),
        'bun': np.random.lognormal(3.0, 0.4, n_samples).clip(2, 80),
        'gcs': rng.normal(13, 2, n_samples).clip(3, 15),
    })

    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    risk_score = (
        0.4 * data_scaled['age'] +
        0.3 * data_scaled['heart_rate'] -
        0.3 * data_scaled['sbp'] +
        0.5 * data_scaled['creatinine'] -
        0.6 * data_scaled['gcs'] +
        0.3 * data_scaled['wbc'] -
        0.4 * data_scaled['spo2'] +
        0.2 * data_scaled['bun'] +
        np.random.normal(0, 1.0, n_samples)
    )

    prob = 1.0 / (1.0 + np.exp(-risk_score + 0.5))
    data['composite_risk_score'] = (np.random.random(n_samples) < prob).astype(int)

    print(f"[DATA] Generated {n_samples} samples, {len(data.columns) - 1} features")
    return data


# =============================================================================
# Fair Comparison Experiment
# =============================================================================
class FairComparisonExperiment:
    """Fair LGO vs AutoScore comparison (core engine, no visualization).

    This object encapsulates:
      - a single experiment configuration (random_state, budget, etc.),
      - running LGO (with/without feature restriction & calibration),
      - running AutoScore (top-n variables),
      - returning comparable metric dictionaries and summary tables.
    """

    def __init__(
        self,
        random_state: int = 42,
        verbose: bool = True,
        n_bootstrap: int = 1000,
        task: str = 'auto',
        binarize_threshold: Optional[Union[float, str]] = None,
        fair_features: bool = False,
        calibration_method: str = 'platt'
    ):
        self.random_state = random_state
        self.verbose = verbose
        self.n_bootstrap = n_bootstrap
        self.task = task
        self.binarize_threshold = binarize_threshold
        self.fair_features = fair_features
        self.calibration_method = calibration_method

        self.results: Dict[str, Any] = {}
        self._detected_task: Optional[str] = None

    def _empty_result(self, method: str) -> Dict[str, Any]:
        """Return an empty result structure for error handling."""
        return {
            'model': None,
            'method': method,
            'train_metrics': {},
            'test_metrics': {},
            'test_probabilities': np.array([]),
            'test_predictions': np.array([]),
            'auroc_ci': (np.nan, np.nan, np.nan),
            'train_time_sec': 0.0,
            'n_generations': np.nan,
            'population_size': np.nan,
        }

    # -------------------------------------------------------------------------
    # LGO
    # -------------------------------------------------------------------------
    def run_lgo(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        outcome_col: str,
        gate_type: str = 'hard',
        n_generations: int = 100,
        population_size: int = 300,
        n_variables: Optional[int] = None,
        max_height: int = 10
    ) -> Dict[str, Any]:
        """Run LGO experiment.

        Notes
        -----
        - If ``self.fair_features`` is True, LGO is also restricted to ``n_variables``
          (top-n features) to match AutoScore.
        - Otherwise, LGO uses all features (full model), while AutoScore still uses
          only the top-n features.
        """

        if not LGO_AVAILABLE:
            print("[ERROR] LGO not available")
            return self._empty_result("LGO")

        task = self._detected_task or 'binary'

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"LGO (gate={gate_type}, gen={n_generations}, pop={population_size})")
            if n_variables and self.fair_features:
                print(f"Feature selection (fair mode): top {n_variables} variables")
            elif n_variables:
                print(f"Feature selection passed but fair_features is False (ignored for LGO)")
            print("=" * 60)

        train_val = pd.concat([train_data, val_data], ignore_index=True)
        X_train = train_val.drop(columns=[outcome_col])
        y_train = train_val[outcome_col]
        X_val = val_data.drop(columns=[outcome_col])
        y_val = val_data[outcome_col]
        X_test = test_data.drop(columns=[outcome_col])
        y_test = test_data[outcome_col]

        t0 = time.time()

        lgo = LGOFairComparison(
            gate_type=gate_type,
            n_generations=n_generations,
            population_size=population_size,
            n_variables=n_variables if self.fair_features else None,
            max_height=max_height,
            random_state=self.random_state,
            verbose=self.verbose,
            calibration_method=self.calibration_method,
        )

        lgo.fit(X_train, y_train)

        # Only perform explicit calibration if requested
        if self.calibration_method != 'none':
            lgo.calibrate(X_val, y_val)

        train_time = time.time() - t0

        y_prob_train = lgo.predict_proba(X_train)
        y_prob_test = lgo.predict_proba(X_test)

        # For binary classification we compute all metrics; for others we skip
        if task == 'binary':
            train_metrics = compute_all_metrics(y_train.values, y_prob_train)
            test_metrics = compute_all_metrics(y_test.values, y_prob_test)
            auroc_ci = bootstrap_auroc_ci(
                y_true=y_test.values,
                y_prob=y_prob_test,
                n_bootstrap=self.n_bootstrap,
                random_state=self.random_state,
            )
        else:
            train_metrics = {}
            test_metrics = {}
            auroc_ci = (np.nan, np.nan, np.nan)

        y_pred_test = (y_prob_test >= 0.5).astype(int)

        results = {
            'model': lgo,
            'method': 'LGO',
            'gate_type': gate_type,
            'formula': lgo.get_formula(),
            'complexity': lgo.get_complexity(),
            'selected_variables': lgo.selected_features,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'test_probabilities': y_prob_test,
            'test_predictions': y_pred_test,
            'auroc_ci': auroc_ci,
            'train_time_sec': train_time,
            'n_generations': n_generations,
            'population_size': population_size,
            'task': task,
            'is_calibrated': lgo._is_calibrated,
            'calibration_method': self.calibration_method,
        }

        if self.verbose:
            print(f"\n[LGO] Training time: {train_time:.1f}s")
            print(f"[LGO] Calibration method: {self.calibration_method}")
            print(f"[LGO] Test AUROC: {test_metrics.get('AUROC', np.nan):.4f}")
            print(f"[LGO] Test Brier: {test_metrics.get('Brier', np.nan):.4f}")

        return results

    # -------------------------------------------------------------------------
    # AutoScore
    # -------------------------------------------------------------------------
    def run_autoscore(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        outcome_col: str,
        n_variables: int = 6
    ) -> Dict[str, Any]:
        """Run AutoScore experiment using AutoScore_v2 pipeline.

        Notes
        -----
        - We call ``AutoScore.run_pipeline`` mainly to build the internal
          scoring model (ranking, parsimony, discretization, score table).
        - AutoScore v2.0 does NOT return raw probabilities in the pipeline
          result dict, so here we obtain probabilities and integer scores
          directly via ``predict_proba`` and ``compute_score``.
        """

        if not AUTOSCORE_AVAILABLE:
            print("[ERROR] AutoScore not available")
            return self._empty_result("AutoScore")

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"AutoScore (n_variables={n_variables})")
            print("=" * 60)

        t0 = time.time()

        # AutoScore_v2.get_autoscore(verbose, random_state=...) -> AutoScore(config)
        autoscore = get_autoscore(verbose=self.verbose, random_state=self.random_state)

        try:
            # Run full AutoScore pipeline to build the score model.
            # In v2.0, this returns a dict with metrics and score_table,
            # but NOT raw probability arrays.
            pipeline_results = autoscore.run_pipeline(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                outcome_col=outcome_col,
                n_variables=n_variables,
                use_combined_training=True,  # Fair comparison: use train+val
            )
        except Exception as e:
            print(f"[ERROR] AutoScore pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result("AutoScore")

        train_time = time.time() - t0

        # Combine train + val for training-metrics / plots
        train_val = pd.concat([train_data, val_data], ignore_index=True)

        # ------------------------------------------------------------------
        # Get probabilities and scores from the fitted AutoScore object
        # ------------------------------------------------------------------
        try:
            train_val_prob = autoscore.predict_proba(train_val)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] AutoScore.predict_proba failed on train+val: {e}")
            train_val_prob = np.full(len(train_val), np.nan)

        try:
            test_prob = autoscore.predict_proba(test_data)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] AutoScore.predict_proba failed on test: {e}")
            test_prob = np.full(len(test_data), np.nan)

        try:
            test_scores = autoscore.compute_score(test_data)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] AutoScore.compute_score failed on test: {e}")
            test_scores = np.full(len(test_data), np.nan)

        # ------------------------------------------------------------------
        # Metrics: prefer pipeline's evaluate() outputs if available;
        # otherwise compute directly from (y, prob) pairs.
        # ------------------------------------------------------------------
        y_train_val = train_val[outcome_col].values
        y_test = test_data[outcome_col].values

        if isinstance(pipeline_results, dict) and 'train_metrics' in pipeline_results:
            # AutoScore_v2 已经在 run_pipeline 里调用了 evaluate()
            train_metrics = pipeline_results.get('train_metrics', {})
            test_metrics = pipeline_results.get('test_metrics', {})
        else:
            train_metrics = compute_all_metrics(y_train_val, train_val_prob)
            test_metrics = compute_all_metrics(y_test, test_prob)

        results = {
            # 用整个 AutoScore 对象作为 "model"（内部包含 score_table 等）
            'model': autoscore,
            'method': 'AutoScore',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'test_probabilities': test_prob,
            'test_predictions': (test_prob >= 0.5).astype(int),
            'test_scores': test_scores,
            'auroc_ci': (np.nan, np.nan, np.nan),  # Not computed for AutoScore
            'train_time_sec': train_time,
        }

        if self.verbose:
            print(f"\n[AutoScore] Training time: {train_time:.1f}s")
            print(f"[AutoScore] Test AUROC: {test_metrics.get('AUROC', np.nan):.4f}")
            print(f"[AutoScore] Test Brier: {test_metrics.get('Brier', np.nan):.4f}")

        return results

    # -------------------------------------------------------------------------
    # Comparison utilities
    # -------------------------------------------------------------------------
    def create_comparison_table(
        self,
        lgo_results: Dict[str, Any],
        autoscore_results: Dict[str, Any],
    ) -> pd.DataFrame:
        """Create a compact comparison table for LGO vs AutoScore."""
        rows = []

        for metric in ['AUROC', 'AUPRC', 'Brier', 'F1', 'Accuracy']:
            lgo_val = lgo_results.get('test_metrics', {}).get(metric, np.nan)
            as_val = autoscore_results.get('test_metrics', {}).get(metric, np.nan)

            if metric == 'Brier':
                winner = 'LGO' if lgo_val < as_val else 'AutoScore'
            else:
                winner = 'LGO' if lgo_val > as_val else 'AutoScore'

            rows.append({
                'Metric': metric,
                'LGO': lgo_val,
                'AutoScore': as_val,
                'Difference': lgo_val - as_val,
                'Winner': winner,
            })

        # Add CI and training time information
        lgo_ci = lgo_results.get('auroc_ci', (np.nan, np.nan, np.nan))
        as_ci = autoscore_results.get('auroc_ci', (np.nan, np.nan, np.nan))
        rows.append({
            'Metric': 'AUROC_CI_lower',
            'LGO': lgo_ci[1],
            'AutoScore': as_ci[1],
            'Difference': '',
            'Winner': '',
        })
        rows.append({
            'Metric': 'AUROC_CI_upper',
            'LGO': lgo_ci[2],
            'AutoScore': as_ci[2],
            'Difference': '',
            'Winner': '',
        })
        rows.append({
            'Metric': 'Training_time_sec',
            'LGO': lgo_results.get('train_time_sec', 0),
            'AutoScore': autoscore_results.get('train_time_sec', 0),
            'Difference': lgo_results.get('train_time_sec', 0) - autoscore_results.get('train_time_sec', 0),
            'Winner': 'LGO' if lgo_results.get('train_time_sec', 0) < autoscore_results.get('train_time_sec', 0) else 'AutoScore',
        })

        return pd.DataFrame(rows)

    def run(
        self,
        data_path: Optional[str] = None,
        outcome_col: str = 'label',
        n_variables: int = 6,
        gate_type: str = 'hard',
        n_generations: int = 100,
        population_size: int = 300,
        max_height: int = 10,
        output_dir: str = './results',
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run full comparison experiment (no visualization)."""
        data = load_data(data_path)

        if outcome_col not in data.columns:
            print(f"[ERROR] Outcome column '{outcome_col}' not found")
            return {}

        train_data, val_data, test_data, inferred_task = split_data(
            data=data,
            outcome_col=outcome_col,
            test_size=0.2,
            val_size=0.1,
            random_state=self.random_state,
            task=self.task,
            binarize_threshold=self.binarize_threshold,
        )
        self._detected_task = inferred_task

        # Run LGO and AutoScore
        lgo_results = self.run_lgo(
            train_data, val_data, test_data, outcome_col,
            gate_type=gate_type,
            n_generations=n_generations,
            population_size=population_size,
            n_variables=n_variables if self.fair_features else None,
            max_height=max_height,
        )

        autoscore_results = self.run_autoscore(
            train_data, val_data, test_data, outcome_col,
            n_variables=n_variables,
        )

        comparison_df = self.create_comparison_table(lgo_results, autoscore_results)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Comparison Results")
            print("=" * 60)
            print(comparison_df.to_string(index=False))

        timestamp = None
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            csv_path = Path(output_dir) / f'comparison_{timestamp}.csv'
            comparison_df.to_csv(csv_path, index=False)
            print(f"[SAVE] CSV: {csv_path}")

            summary = {
                'timestamp': timestamp,
                'version': '2.5.0',
                'config': {
                    'n_variables': n_variables,
                    'gate_type': gate_type,
                    'n_generations': n_generations,
                    'population_size': population_size,
                    'calibration_method': self.calibration_method,
                    'random_state': self.random_state,
                    'fair_features': self.fair_features,
                },
                'lgo': {
                    'test_metrics': lgo_results.get('test_metrics', {}),
                    'formula': lgo_results.get('formula', ''),
                    'train_time_sec': lgo_results.get('train_time_sec', 0),
                },
                'autoscore': {
                    'test_metrics': autoscore_results.get('test_metrics', {}),
                    'train_time_sec': autoscore_results.get('train_time_sec', 0),
                },
            }

            json_path = Path(output_dir) / f'summary_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"[SAVE] JSON: {json_path}")

        self.results = {
            'timestamp': timestamp,
            'lgo_results': lgo_results,
            'autoscore_results': autoscore_results,
            'comparison_df': comparison_df,
            'y_test': test_data[outcome_col].values,  # For visualization
        }
        return self.results


# =============================================================================
# Multi-Seed Experiment
# =============================================================================
def run_multi_seed_experiment(
    data_path: Optional[str] = None,
    outcome_col: str = 'composite_risk_score',
    seeds: List[int] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
    output_dir: str = './results',
    task: str = 'auto',
    binarize_threshold: Optional[Union[float, str]] = None,
    calibration_method: str = 'platt',
    n_bootstrap: int = 1000,
    fair_features: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Run LGO vs AutoScore comparison across multiple random seeds.

    Parameters
    ----------
    data_path : str
        Path to the input CSV file. If ``None``, a synthetic dataset will be generated.
    outcome_col : str
        Name of the outcome column in the dataset.
    seeds : list[int]
        List of random seeds used for train/val/test splits and model training.
    output_dir : str
        Directory to store the multi-seed CSV summaries and detailed pickle file.
    task : {"auto", "binary", "multiclass", "regression"}
        Task type. If "auto", the type will be inferred from the outcome variable.
    binarize_threshold : float or {"median", "mean"} or None
        Threshold for binarizing a continuous outcome when using a binary task.
    calibration_method : {"platt", "isotonic", "none"}
        Probability calibration method for LGO.
    n_bootstrap : int
        Number of bootstrap samples used to estimate AUROC confidence interval per seed.
    fair_features : bool
        If True, LGO is also restricted to top-n features (same ``n_variables`` as AutoScore).
    **kwargs :
        Additional keyword arguments forwarded to :meth:`FairComparisonExperiment.run`,
        e.g. ``n_variables``, ``gate_type``, ``n_generations``, ``population_size``, ``max_height``.
    """
    print("\n" + "=" * 70)
    print(f"Multi-Seed Experiment ({len(seeds)} seeds)")
    print(f"Calibration: {calibration_method}")
    print(f"Fair features (LGO limited to top-n variables): {fair_features}")
    print("=" * 70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    all_detailed: List[Dict[str, Any]] = []  # For visualization / plotting

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running seed={seed}")

        # Instantiate an experiment for this specific seed
        exp = FairComparisonExperiment(
            random_state=seed,
            verbose=False,
            n_bootstrap=n_bootstrap,
            task=task,
            binarize_threshold=binarize_threshold,
            fair_features=fair_features,
            calibration_method=calibration_method,
        )

        # Run experiment for this seed; do not save per-seed CSVs here
        results = exp.run(
            data_path=data_path,
            outcome_col=outcome_col,
            save_results=False,
            **kwargs,
        )

        # Collect detailed outputs for later plotting / analysis
        seed_detail = {
            'seed': seed,
            'lgo': {
                'test_probabilities': results.get('lgo_results', {}).get('test_probabilities', np.array([])),
                'test_predictions': results.get('lgo_results', {}).get('test_predictions', np.array([])),
                'test_metrics': results.get('lgo_results', {}).get('test_metrics', {}),
                'formula': results.get('lgo_results', {}).get('formula', ''),
                'train_time_sec': results.get('lgo_results', {}).get('train_time_sec', 0),
                'is_calibrated': results.get('lgo_results', {}).get('is_calibrated', False),
                'calibration_method': results.get('lgo_results', {}).get('calibration_method', 'none'),
            },
            'autoscore': {
                'test_probabilities': results.get('autoscore_results', {}).get('test_probabilities', np.array([])),
                'test_predictions': results.get('autoscore_results', {}).get('test_predictions', np.array([])),
                'test_scores': results.get('autoscore_results', {}).get('test_scores', np.array([])),
                'test_metrics': results.get('autoscore_results', {}).get('test_metrics', {}),
                'train_time_sec': results.get('autoscore_results', {}).get('train_time_sec', 0),
            },
            'y_test': results.get('y_test', np.array([])),  # ground-truth labels on the test set
        }
        all_detailed.append(seed_detail)

        # Collect per-method scalar metrics for summary CSV
        for method, res in [
            ('LGO', results.get('lgo_results', {})),
            ('AutoScore', results.get('autoscore_results', {})),
        ]:
            if res:
                metrics = res.get('test_metrics', {})
                row = {
                    'seed': seed,
                    'method': method,
                    'AUROC': metrics.get('AUROC', np.nan),
                    'AUPRC': metrics.get('AUPRC', np.nan),
                    'Brier': metrics.get('Brier', np.nan),
                    'F1': metrics.get('F1', np.nan),
                    'Accuracy': metrics.get('Accuracy', np.nan),
                    'train_time': res.get('train_time_sec', np.nan),
                }
                all_results.append(row)

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("Multi-Seed Summary")
    print("=" * 60)

    numeric_cols = ['AUROC', 'AUPRC', 'Brier', 'F1', 'Accuracy', 'train_time']
    summary = results_df.groupby('method')[numeric_cols].agg(['mean', 'std']).round(4)
    print(summary)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_path = Path(output_dir) / f'multiseed_results_{timestamp}.csv'
    summary_path = Path(output_dir) / f'multiseed_summary_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    summary.to_csv(summary_path)

    # Save detailed data for later visualization with run_comparison_plot_v1.py
    import pickle
    detailed_path = Path(output_dir) / f'multiseed_detailed_{timestamp}.pkl'
    with open(detailed_path, 'wb') as f:
        pickle.dump(all_detailed, f)

    print(f"[SAVE] Multi-seed results: {results_path}")
    print(f"[SAVE] Multi-seed summary: {summary_path}")
    print(f"[SAVE] Detailed data for plotting: {detailed_path}")
    print(f"[TIP] Use run_comparison_plot_v1.py --png_seed <seed> to generate detailed visualizations")

    return results_df


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='LGO vs AutoScore Fair Comparison (v2.5) - Core Engine'
    )
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the input CSV file.')
    parser.add_argument('--outcome', type=str, default='composite_risk_score',
                        help='Name of the outcome column.')
    parser.add_argument('--task', type=str, default='auto',
                        choices=['auto', 'binary', 'multiclass', 'regression'],
                        help='Task type; "auto" will infer from the outcome.')
    parser.add_argument('--binarize_threshold', type=str, default=None,
                        help='Threshold to binarize outcome (float, "median" or "mean").')
    parser.add_argument('--n_variables', type=int, default=6,
                        help='Number of top variables used by AutoScore (and LGO when --fair_features).')
    parser.add_argument('--gate_type', type=str, default='hard',
                        choices=['hard', 'soft'],
                        help='Gate type for LGO.')
    parser.add_argument('--n_generations', type=int, default=100,
                        help='Number of LGO generations (controls budget).')
    parser.add_argument('--population_size', type=int, default=300,
                        help='LGO population size per generation (controls budget).')
    parser.add_argument('--max_height', type=int, default=10,
                        help='Maximum expression tree height for LGO.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for single-run experiment.')
    parser.add_argument('--seeds', type=str, default='',
                        help='Comma-separated list of seeds for multi-seed experiment.')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples for AUROC CI.')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for CSV/JSON files.')
    parser.add_argument('--fair_features', action='store_true',
                        help='If set, LGO is also restricted to top-n variables.')
    parser.add_argument('--calibration_method', type=str, default='platt',
                        choices=['platt', 'isotonic', 'none'],
                        help='Calibration method for LGO probabilities.')

    args = parser.parse_args()

    # Parse binarization threshold
    binarize_threshold = None
    if args.binarize_threshold:
        if args.binarize_threshold in ['median', 'mean']:
            binarize_threshold = args.binarize_threshold
        else:
            try:
                binarize_threshold = float(args.binarize_threshold)
            except ValueError:
                print(f"[ERROR] Invalid binarize_threshold: {args.binarize_threshold}")
                sys.exit(1)

    # Multi-seed mode
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        run_multi_seed_experiment(
            data_path=args.data_path,
            outcome_col=args.outcome,
            seeds=seeds,
            output_dir=args.output_dir,
            task=args.task,
            binarize_threshold=binarize_threshold,
            calibration_method=args.calibration_method,
            n_bootstrap=args.n_bootstrap,
            fair_features=args.fair_features,
            n_variables=args.n_variables,
            gate_type=args.gate_type,
            n_generations=args.n_generations,
            population_size=args.population_size,
            max_height=args.max_height,
        )
    # Single-run mode
    else:
        exp = FairComparisonExperiment(
            random_state=args.seed,
            verbose=True,
            n_bootstrap=args.n_bootstrap,
            task=args.task,
            binarize_threshold=binarize_threshold,
            fair_features=args.fair_features,
            calibration_method=args.calibration_method,
        )

        exp.run(
            data_path=args.data_path,
            outcome_col=args.outcome,
            n_variables=args.n_variables,
            gate_type=args.gate_type,
            n_generations=args.n_generations,
            population_size=args.population_size,
            max_height=args.max_height,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()