#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LGO vs InterpretML (EBM) Fair Comparison - Core Engine
========================================================
Version: 1.0.0
Date: Dec 7, 2025

对比LGO（符号回归）与EBM（可解释提升机）在临床风险预测任务上的性能。
两者都是高度可解释的机器学习方法，但采用不同的建模范式：

- LGO: 基于遗传规划的符号回归，生成包含阈值门控的显式数学公式
- EBM: 基于广义加性模型(GAM)和梯度提升的方法，提供特征形状函数

Usage:
------
# 单种子运行
python run_lgo_interpret_comparison.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --seed 42 \
  --output_dir results_single

# 多种子运行 ICU
python run_lgo_interpret_comparison.py \
  --data_path ../data/ICU/ICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --calibration_method platt \
  --output_dir ICU_results_30k

  # 多种子运行 eICU
python run_lgo_interpret_comparison.py \
  --data_path ../data/eICU/eICU_composite_risk_score.csv \
  --outcome composite_risk_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --calibration_method platt \
  --output_dir eICU_results_30k

  # 多种子运行 NHANES
python run_lgo_interpret_comparison.py \
  --data_path ../data/NHANES/NHANES_metabolic_score.csv \
  --outcome metabolic_score \
  --binarize_threshold 5 \
  --gate_type hard \
  --n_generations 100 \
  --population_size 300 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --calibration_method platt \
  --output_dir NHANES_results_30k


# 使用合成数据（无需数据文件）
python run_lgo_interpret_comparison.py \
  --seeds 1,2,3 \
  --output_dir results_synthetic

"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle

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
# Import LGO v2.2
# =============================================================================
try:
    from LGO_v2_2 import run_lgo_sr_v3, ZScaler
    LGO_AVAILABLE = True
    print("[INFO] Using LGO v2.2 (run_lgo_sr_v3 is alias of run_lgo_sr_v2)")
except ImportError:
    try:
        # Try importing from parent directory if run from different location
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'LGO_AutoScore_v3_7_1207PM'))
        from LGO_v2_2 import run_lgo_sr_v3, ZScaler
        LGO_AVAILABLE = True
        print("[INFO] Using LGO v2.2 (from parent directory)")
    except ImportError:
        LGO_AVAILABLE = False
        print("[WARNING] LGO not found. LGO experiments will be disabled.")

# =============================================================================
# Import InterpretML (EBM)
# =============================================================================
try:
    from InterpretML_v1 import EBMWrapper, EBMConfig, get_ebm, INTERPRET_AVAILABLE
    if not INTERPRET_AVAILABLE:
        print("[WARNING] InterpretML package not installed. EBM experiments will be disabled.")
except ImportError:
    INTERPRET_AVAILABLE = False
    print("[WARNING] InterpretML_v1.py not found. EBM experiments will be disabled.")


# =============================================================================
# LGO Wrapper for Fair Comparison
# =============================================================================
class LGOFairComparison:
    """LGO wrapper for fair comparison with EBM.
    
    Features:
    - Optional feature selection (top-n via RF importance)
    - Probability calibration (Platt or Isotonic)
    - sklearn-like interface (fit/predict_proba/predict)
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
        """Fit LGO model with optional feature selection."""
        if not LGO_AVAILABLE:
            raise RuntimeError("LGO is not available. Please check imports.")
        
        if self.verbose:
            print(f"\n[LGO] Training with gate_type={self.gate_type}")
        
        self._full_feature_names = list(X.columns)
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        # Feature selection
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
        
        # Run symbolic regression
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
        
        if self.verbose:
            print("[LGO] Training completed.")
        
        return self
    
    def get_formula(self) -> str:
        """Return the best LGO formula expression."""
        if self._result_df is None or 'expr' not in self._result_df.columns:
            return ""
        best = self._result_df.iloc[0]
        return str(best['expr'])
    
    def get_complexity(self) -> int:
        """Return model complexity (expression size)."""
        if self._result_df is None or 'size' not in self._result_df.columns:
            return 0
        return int(self._result_df.iloc[0]['size'])
    
    def _evaluate_raw_expression(self, X: pd.DataFrame) -> np.ndarray:
        """Evaluate the raw symbolic expression on new data."""
        if self._result_df is None:
            raise RuntimeError("LGO has not been fit yet.")
        
        expr_str = self.get_formula()
        if not expr_str:
            return np.zeros(len(X))
        
        # Import primitives from LGO engine
        from LGO_v2_2 import (
            sdiv, ssqrt, slog, spow,
            lgo, lgo_thre, lgo_and2, lgo_or2, lgo_and3, gate_expr,
            as_pos_func, as_thr_func,
        )
        
        # Build evaluation namespace
        ns = {}
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
        
        # Add feature columns
        for col in self.selected_features:
            ns[col] = X[col].values
        
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
                print("[LGO] Calibration disabled")
            return self
        
        y_raw = self._evaluate_raw_expression(X_val)
        
        if self.calibration_method == 'platt':
            self._calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
            self._calibrator.fit(y_raw.reshape(-1, 1), y_val)
        elif self.calibration_method == 'isotonic':
            self._calibrator = IsotonicRegression(out_of_bounds='clip')
            self._calibrator.fit(y_raw, y_val)
        else:
            raise ValueError(f"Unknown calibration_method={self.calibration_method}")
        
        self._is_calibrated = True
        if self.verbose:
            print(f"[LGO] Calibration fitted with method={self.calibration_method}")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities."""
        if self._result_df is None:
            raise RuntimeError("LGO has not been fit yet.")
        
        X = X[self.selected_features]
        y_raw = self._evaluate_raw_expression(X)
        
        if self._calibrator is None or not self._is_calibrated:
            probs = 1.0 / (1.0 + np.exp(-y_raw))
        else:
            if self.calibration_method == 'platt':
                probs = self._calibrator.predict_proba(y_raw.reshape(-1, 1))[:, 1]
            else:
                probs = self._calibrator.predict(y_raw)
        return probs
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


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
    """Convert continuous target to binary."""
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
    """Compute AUROC, AUPRC, Brier, F1, Accuracy."""
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
    """Compute bootstrap CI for AUROC."""
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
    """Split data into train/validation/test sets."""
    y = data[outcome_col].values
    inferred_task = detect_task_type(y) if task == 'auto' else task
    print(f"[DATA] Inferred task type: {inferred_task}")
    
    # 如果提供了binarize_threshold，强制执行二值化（无论原始任务类型）
    if binarize_threshold is not None:
        data = data.copy()
        data[outcome_col] = binarize_target(y, threshold=binarize_threshold)
        y = data[outcome_col].values
        inferred_task = 'binary'  # 强制设为二分类
        print(f"[DATA] Binarized with threshold={binarize_threshold}, class distribution: {np.bincount(y.astype(int))}")
    
    if inferred_task == 'binary':
        stratify = y
        if len(np.unique(y)) < 2:
            print("[WARN] Only one class present. Stratified split disabled.")
            stratify = None
    else:
        stratify = None
    
    # Train+val vs test
    train_val_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=stratify,
    )
    
    # Train vs val
    if val_size > 0:
        relative_val_size = val_size / (1.0 - test_size)
        train_data, val_data = train_test_split(
            train_val_data, test_size=relative_val_size, random_state=random_state,
            stratify=train_val_data[outcome_col] if stratify is not None else None,
        )
    else:
        train_data = train_val_data
        val_data = pd.DataFrame(columns=data.columns)
    
    print(f"[DATA] Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data, inferred_task


def load_data(data_path: Optional[str] = None, n_samples: int = 2000) -> pd.DataFrame:
    """Load real data or generate synthetic data."""
    if data_path is None:
        print("[DATA] No data_path provided. Generating synthetic dataset...")
        return generate_synthetic_data(n_samples=n_samples)
    
    # 将相对路径转换为绝对路径
    path = Path(data_path).resolve()
    if not path.exists():
        # 也尝试原始路径（以防resolve有问题）
        path_orig = Path(data_path)
        if path_orig.exists():
            path = path_orig
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"  Resolved path: {path}\n"
                f"  Current working directory: {Path.cwd()}"
            )
    
    print(f"[DATA] Loading data from {path}")
    df = pd.read_csv(path)
    print(f"[DATA] Loaded shape: {df.shape}")
    return df


def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic ICU-like dataset."""
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
class LGOvsEBMExperiment:
    """Fair LGO vs EBM comparison experiment."""
    
    def __init__(
        self,
        random_state: int = 42,
        verbose: bool = True,
        n_bootstrap: int = 1000,
        task: str = 'auto',
        binarize_threshold: Optional[Union[float, str]] = None,
        calibration_method: str = 'platt'
    ):
        self.random_state = random_state
        self.verbose = verbose
        self.n_bootstrap = n_bootstrap
        self.task = task
        self.binarize_threshold = binarize_threshold
        self.calibration_method = calibration_method
        
        self.results: Dict[str, Any] = {}
        self._detected_task: Optional[str] = None
    
    def _empty_result(self, method: str) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'model': None,
            'method': method,
            'train_metrics': {},
            'test_metrics': {},
            'test_probabilities': np.array([]),
            'test_predictions': np.array([]),
            'auroc_ci': (np.nan, np.nan, np.nan),
            'train_time_sec': 0.0,
        }
    
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
        """Run LGO experiment."""
        
        if not LGO_AVAILABLE:
            print("[ERROR] LGO not available")
            return self._empty_result("LGO")
        
        task = self._detected_task or 'binary'
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"LGO (gate={gate_type}, gen={n_generations}, pop={population_size})")
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
            n_variables=n_variables,
            max_height=max_height,
            random_state=self.random_state,
            verbose=self.verbose,
            calibration_method=self.calibration_method,
        )
        
        lgo.fit(X_train, y_train)
        
        if self.calibration_method != 'none':
            lgo.calibrate(X_val, y_val)
        
        train_time = time.time() - t0
        
        y_prob_train = lgo.predict_proba(X_train)
        y_prob_test = lgo.predict_proba(X_test)
        
        if task == 'binary':
            train_metrics = compute_all_metrics(y_train.values, y_prob_train)
            test_metrics = compute_all_metrics(y_test.values, y_prob_test)
            auroc_ci = bootstrap_auroc_ci(
                y_true=y_test.values, y_prob=y_prob_test,
                n_bootstrap=self.n_bootstrap, random_state=self.random_state,
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
            print(f"[LGO] Test AUROC: {test_metrics.get('AUROC', np.nan):.4f}")
            print(f"[LGO] Test Brier: {test_metrics.get('Brier', np.nan):.4f}")
        
        return results
    
    def run_ebm(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        outcome_col: str,
        interactions: int = 10,
        learning_rate: float = 0.01,
        max_bins: int = 256,
    ) -> Dict[str, Any]:
        """Run EBM experiment."""
        
        if not INTERPRET_AVAILABLE:
            print("[ERROR] InterpretML not available")
            return self._empty_result("EBM")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"EBM (interactions={interactions}, lr={learning_rate})")
            print("=" * 60)
        
        task = self._detected_task or 'binary'
        
        # Use train+val combined for training (same as LGO)
        train_val = pd.concat([train_data, val_data], ignore_index=True)
        X_train = train_val.drop(columns=[outcome_col])
        y_train = train_val[outcome_col]
        X_test = test_data.drop(columns=[outcome_col])
        y_test = test_data[outcome_col]
        
        t0 = time.time()
        
        ebm = get_ebm(
            verbose=self.verbose,
            random_state=self.random_state,
            interactions=interactions,
            learning_rate=learning_rate,
            max_bins=max_bins,
        )
        
        ebm_task = 'classification' if task == 'binary' else 'regression'
        ebm.fit(X_train, y_train, task=ebm_task)
        
        train_time = time.time() - t0
        
        y_prob_train = ebm.predict_proba(X_train)
        y_prob_test = ebm.predict_proba(X_test)
        
        if task == 'binary':
            train_metrics = compute_all_metrics(y_train.values, y_prob_train)
            test_metrics = compute_all_metrics(y_test.values, y_prob_test)
            auroc_ci = bootstrap_auroc_ci(
                y_true=y_test.values, y_prob=y_prob_test,
                n_bootstrap=self.n_bootstrap, random_state=self.random_state,
            )
        else:
            train_metrics = {}
            test_metrics = {}
            auroc_ci = (np.nan, np.nan, np.nan)
        
        y_pred_test = (y_prob_test >= 0.5).astype(int)
        
        # Get EBM summary
        summary = ebm.get_model_summary()
        importances = ebm.get_feature_importances()
        
        results = {
            'model': ebm,
            'method': 'EBM',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'test_probabilities': y_prob_test,
            'test_predictions': y_pred_test,
            'auroc_ci': auroc_ci,
            'train_time_sec': train_time,
            'task': task,
            'n_terms': summary['n_terms'],
            'n_main_effects': summary['n_main_effects'],
            'n_interactions': summary['n_interactions'],
            'feature_importances': importances.to_dict(),
        }
        
        if self.verbose:
            print(f"\n[EBM] Training time: {train_time:.1f}s")
            print(f"[EBM] Terms: {summary['n_terms']} ({summary['n_main_effects']} main + {summary['n_interactions']} interactions)")
            print(f"[EBM] Test AUROC: {test_metrics.get('AUROC', np.nan):.4f}")
            print(f"[EBM] Test Brier: {test_metrics.get('Brier', np.nan):.4f}")
        
        return results
    
    def run(
        self,
        data_path: Optional[str] = None,
        outcome_col: str = 'composite_risk_score',
        gate_type: str = 'hard',
        n_generations: int = 100,
        population_size: int = 300,
        n_variables: Optional[int] = None,
        max_height: int = 10,
        ebm_interactions: int = 10,
        ebm_learning_rate: float = 0.01,
        output_dir: str = './results',
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Run full comparison experiment."""
        
        print("\n" + "=" * 70)
        print("LGO vs EBM Fair Comparison")
        print(f"Calibration: {self.calibration_method}")
        print("=" * 70)
        
        # Load data
        data = load_data(data_path)
        
        # Split data
        train_data, val_data, test_data, inferred_task = split_data(
            data=data,
            outcome_col=outcome_col,
            random_state=self.random_state,
            task=self.task,
            binarize_threshold=self.binarize_threshold,
        )
        self._detected_task = inferred_task
        
        # Run LGO
        lgo_results = self.run_lgo(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            outcome_col=outcome_col,
            gate_type=gate_type,
            n_generations=n_generations,
            population_size=population_size,
            n_variables=n_variables,
            max_height=max_height,
        )
        
        # Run EBM
        ebm_results = self.run_ebm(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            outcome_col=outcome_col,
            interactions=ebm_interactions,
            learning_rate=ebm_learning_rate,
        )
        
        # Aggregate results
        self.results = {
            'lgo_results': lgo_results,
            'ebm_results': ebm_results,
            'y_test': test_data[outcome_col].values,
            'random_state': self.random_state,
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'Metric':<15} {'LGO':>12} {'EBM':>12}")
        print("-" * 40)
        
        for metric in ['AUROC', 'AUPRC', 'Brier', 'F1', 'Accuracy']:
            lgo_val = lgo_results['test_metrics'].get(metric, np.nan)
            ebm_val = ebm_results['test_metrics'].get(metric, np.nan)
            print(f"{metric:<15} {lgo_val:>12.4f} {ebm_val:>12.4f}")
        
        print("-" * 40)
        print(f"{'Train Time (s)':<15} {lgo_results['train_time_sec']:>12.1f} {ebm_results['train_time_sec']:>12.1f}")
        
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save CSV summary
            summary_df = pd.DataFrame([
                {'method': 'LGO', **lgo_results['test_metrics'], 'train_time': lgo_results['train_time_sec']},
                {'method': 'EBM', **ebm_results['test_metrics'], 'train_time': ebm_results['train_time_sec']},
            ])
            summary_path = Path(output_dir) / f'comparison_summary_{timestamp}.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"\n[SAVE] Summary: {summary_path}")
        
        return self.results


# =============================================================================
# Multi-Seed Experiment
# =============================================================================
def run_multi_seed_experiment(
    data_path: Optional[str],
    outcome_col: str,
    seeds: List[int],
    output_dir: str,
    task: str = 'auto',
    binarize_threshold: Optional[Union[float, str]] = None,
    calibration_method: str = 'platt',
    n_bootstrap: int = 1000,
    **kwargs
) -> pd.DataFrame:
    """Run multi-seed experiment."""
    
    print("\n" + "=" * 70)
    print(f"Multi-Seed Experiment ({len(seeds)} seeds)")
    print(f"Calibration: {calibration_method}")
    print("=" * 70)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results: List[Dict[str, Any]] = []
    all_detailed: List[Dict[str, Any]] = []
    
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running seed={seed}")
        
        exp = LGOvsEBMExperiment(
            random_state=seed,
            verbose=False,
            n_bootstrap=n_bootstrap,
            task=task,
            binarize_threshold=binarize_threshold,
            calibration_method=calibration_method,
        )
        
        results = exp.run(
            data_path=data_path,
            outcome_col=outcome_col,
            save_results=False,
            **kwargs,
        )
        
        # Collect detailed outputs
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
            'ebm': {
                'test_probabilities': results.get('ebm_results', {}).get('test_probabilities', np.array([])),
                'test_predictions': results.get('ebm_results', {}).get('test_predictions', np.array([])),
                'test_metrics': results.get('ebm_results', {}).get('test_metrics', {}),
                'train_time_sec': results.get('ebm_results', {}).get('train_time_sec', 0),
                'n_terms': results.get('ebm_results', {}).get('n_terms', 0),
                'feature_importances': results.get('ebm_results', {}).get('feature_importances', {}),
            },
            'y_test': results.get('y_test', np.array([])),
        }
        all_detailed.append(seed_detail)
        
        # Collect per-method metrics
        for method, res in [
            ('LGO', results.get('lgo_results', {})),
            ('EBM', results.get('ebm_results', {})),
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
    
    # Save detailed data for visualization
    detailed_path = Path(output_dir) / f'multiseed_detailed_{timestamp}.pkl'
    with open(detailed_path, 'wb') as f:
        pickle.dump(all_detailed, f)
    
    print(f"\n[SAVE] Multi-seed results: {results_path}")
    print(f"[SAVE] Multi-seed summary: {summary_path}")
    print(f"[SAVE] Detailed data for plotting: {detailed_path}")
    print(f"[TIP] Use run_lgo_interpret_plot.py to generate visualizations")
    
    return results_df


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='LGO vs InterpretML (EBM) Fair Comparison'
    )
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to CSV data file (default: generate synthetic data)')
    parser.add_argument('--outcome', type=str, default='composite_risk_score',
                        help='Target variable column name')
    parser.add_argument('--task', type=str, default='auto', 
                        choices=['auto', 'binary', 'multiclass', 'regression'])
    parser.add_argument('--binarize_threshold', type=str, default=None,
                        help='Threshold for binarization (number, "median", or "mean")')
    parser.add_argument('--n_variables', type=int, default=None,
                        help='Number of variables for LGO feature selection')
    
    # LGO parameters
    parser.add_argument('--gate_type', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--n_generations', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=300)
    parser.add_argument('--max_height', type=int, default=10)
    parser.add_argument('--calibration_method', type=str, default='platt',
                        choices=['platt', 'isotonic', 'none'])
    
    # EBM parameters
    parser.add_argument('--ebm_interactions', type=int, default=10,
                        help='Number of automatic interactions for EBM')
    parser.add_argument('--ebm_learning_rate', type=float, default=0.01,
                        help='Learning rate for EBM')
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default='',
                        help='Comma-separated seeds for multi-seed experiment')
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
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
            n_variables=args.n_variables,
            gate_type=args.gate_type,
            n_generations=args.n_generations,
            population_size=args.population_size,
            max_height=args.max_height,
            ebm_interactions=args.ebm_interactions,
            ebm_learning_rate=args.ebm_learning_rate,
        )
    else:
        exp = LGOvsEBMExperiment(
            random_state=args.seed,
            verbose=True,
            n_bootstrap=args.n_bootstrap,
            task=args.task,
            binarize_threshold=binarize_threshold,
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
            ebm_interactions=args.ebm_interactions,
            ebm_learning_rate=args.ebm_learning_rate,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
