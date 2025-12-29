#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InterpretML_v1.py - Explainable Boosting Machine (EBM) Wrapper
==============================================================
Version: 1.0.0
Date: Dec 7, 2025

InterpretML的Python封装，用于与LGO进行公平比较。
EBM是一种高度可解释的机器学习模型，结合了GAM和boosting的优点。

Reference: https://github.com/interpretml/interpret
Paper: InterpretML: A Unified Framework for Machine Learning Interpretability
       https://arxiv.org/abs/1909.09223
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import time

from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    brier_score_loss, 
    average_precision_score
)
from sklearn.preprocessing import StandardScaler

# InterpretML import
try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from interpret import show
    INTERPRET_AVAILABLE = True
except ImportError:
    INTERPRET_AVAILABLE = False
    ExplainableBoostingClassifier = None
    ExplainableBoostingRegressor = None
    print("[WARNING] InterpretML not installed. Please run: pip install interpret")


@dataclass
class EBMConfig:
    """EBM配置参数
    
    Parameters matching R/Python interpret defaults and AutoScore comparison settings.
    """
    # 模型参数
    max_bins: int = 256                    # 特征分箱数
    max_interaction_bins: int = 32         # 交互项分箱数
    interactions: int = 10                 # 自动检测的交互项数
    outer_bags: int = 8                    # 外层bagging次数
    inner_bags: int = 0                    # 内层bagging次数
    learning_rate: float = 0.01            # 学习率
    min_samples_leaf: int = 2              # 叶节点最小样本数
    max_leaves: int = 3                    # 每棵树最大叶节点数
    early_stopping_rounds: int = 50        # 早停轮数
    early_stopping_tolerance: float = 1e-4 # 早停容忍度
    
    # 训练参数
    n_jobs: int = -1                       # 并行数 (-1表示全部)
    random_state: int = 42                 # 随机种子
    
    # 其他
    verbose: bool = True


class EBMWrapper:
    """
    Explainable Boosting Machine (EBM) 封装类
    
    提供与LGO类似的接口用于公平比较。
    EBM是一种基于广义加性模型(GAM)的可解释机器学习方法。
    
    特点:
    - 高精度: 与随机森林、梯度提升相当
    - 高可解释性: 可视化每个特征的贡献
    - 自动特征交互检测
    """
    
    def __init__(self, config: Optional[EBMConfig] = None):
        """
        初始化EBM
        
        Parameters:
            config: EBM配置对象
        """
        if not INTERPRET_AVAILABLE:
            raise ImportError("InterpretML not installed. Please run: pip install interpret")
            
        self.config = config or EBMConfig()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        self.task_type = 'classification'  # or 'regression'
        
        if self.config.verbose:
            print("InterpretML EBM Wrapper v1.0 initialized")
    
    def _get_term_names(self):
        """安全获取term_names（兼容新旧版本InterpretML）"""
        if not self.is_fitted:
            return []
        # 尝试带下划线的属性（旧版本）
        result = getattr(self.model, 'term_names_', None)
        if result is not None:
            return result
        # 尝试不带下划线的属性或方法（新版本）
        result = getattr(self.model, 'term_names', None)
        if result is None:
            return []
        # 如果是方法则调用它
        if callable(result):
            return result()
        return result
    
    def _get_term_importances(self):
        """安全获取term_importances（兼容新旧版本InterpretML）"""
        if not self.is_fitted:
            return []
        # 尝试带下划线的属性（旧版本）
        result = getattr(self.model, 'term_importances_', None)
        if result is not None:
            return result
        # 尝试不带下划线的属性或方法（新版本）
        result = getattr(self.model, 'term_importances', None)
        if result is None:
            return []
        # 如果是方法则调用它
        if callable(result):
            return result()
        return result
    
    def _get_bins(self):
        """安全获取bins（兼容新旧版本InterpretML）"""
        if not self.is_fitted:
            return []
        result = getattr(self.model, 'bins_', None)
        if result is not None:
            return result
        result = getattr(self.model, 'bins', None)
        if result is None:
            return []
        if callable(result):
            return result()
        return result
    
    def _get_term_scores(self):
        """安全获取term_scores（兼容新旧版本InterpretML）"""
        if not self.is_fitted:
            return []
        result = getattr(self.model, 'term_scores_', None)
        if result is not None:
            return result
        result = getattr(self.model, 'term_scores', None)
        if result is None:
            return []
        if callable(result):
            return result()
        return result
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        task: str = 'classification'
    ) -> 'EBMWrapper':
        """
        训练EBM模型
        
        Parameters:
            X: 特征矩阵
            y: 目标变量
            task: 'classification' 或 'regression'
            
        Returns:
            self
        """
        if self.config.verbose:
            print("\n[EBM] Training Explainable Boosting Machine")
            print("-" * 50)
        
        # 转换为numpy array
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        if isinstance(y, pd.Series):
            y = y.values
        
        self.task_type = task
        
        # 创建EBM模型
        if task == 'classification':
            self.model = ExplainableBoostingClassifier(
                max_bins=self.config.max_bins,
                max_interaction_bins=self.config.max_interaction_bins,
                interactions=self.config.interactions,
                outer_bags=self.config.outer_bags,
                inner_bags=self.config.inner_bags,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
                max_leaves=self.config.max_leaves,
                early_stopping_rounds=self.config.early_stopping_rounds,
                early_stopping_tolerance=self.config.early_stopping_tolerance,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                feature_names=self.feature_names,
            )
        else:
            self.model = ExplainableBoostingRegressor(
                max_bins=self.config.max_bins,
                max_interaction_bins=self.config.max_interaction_bins,
                interactions=self.config.interactions,
                outer_bags=self.config.outer_bags,
                inner_bags=self.config.inner_bags,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
                max_leaves=self.config.max_leaves,
                early_stopping_rounds=self.config.early_stopping_rounds,
                early_stopping_tolerance=self.config.early_stopping_tolerance,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                feature_names=self.feature_names,
            )
        
        # 训练模型
        t0 = time.time()
        self.model.fit(X, y)
        train_time = time.time() - t0
        
        self.is_fitted = True
        
        if self.config.verbose:
            print(f"[EBM] Training completed in {train_time:.2f}s")
            print(f"[EBM] Number of features: {len(self.feature_names)}")
            print(f"[EBM] Number of terms: {len(self._get_term_names())}")
        
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测概率
        
        Parameters:
            X: 特征矩阵
            
        Returns:
            预测概率数组 (二分类返回正类概率)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if self.task_type == 'classification':
            proba = self.model.predict_proba(X)
            return proba[:, 1]  # 返回正类概率
        else:
            # 回归任务直接返回预测值
            return self.model.predict(X)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        预测类别
        
        Parameters:
            X: 特征矩阵
            threshold: 分类阈值
            
        Returns:
            预测类别数组
        """
        if self.task_type == 'classification':
            proba = self.predict_proba(X)
            return (proba >= threshold).astype(int)
        else:
            return self.model.predict(X)
    
    def get_feature_importances(self) -> pd.Series:
        """
        获取特征重要性
        
        Returns:
            特征重要性 (pd.Series)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        
        # EBM提供term_importances，包括主效应和交互项
        importances = {}
        for name, importance in zip(self._get_term_names(), self._get_term_importances()):
            importances[name] = importance
        
        return pd.Series(importances).sort_values(ascending=False)
    
    def get_global_explanation(self):
        """
        获取全局解释对象
        
        Returns:
            InterpretML全局解释对象
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        
        from interpret import show
        return self.model.explain_global()
    
    def get_local_explanation(self, X: Union[pd.DataFrame, np.ndarray], idx: int = 0):
        """
        获取局部解释对象（单样本）
        
        Parameters:
            X: 特征矩阵
            idx: 样本索引
            
        Returns:
            InterpretML局部解释对象
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.explain_local(X[idx:idx+1])
    
    def get_shape_functions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        获取形状函数（每个特征对预测的贡献）
        
        Returns:
            Dict[特征名, (特征值bins, 对应贡献值)]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        
        shape_functions = {}
        term_names = self._get_term_names()
        bins = self._get_bins()
        term_scores = self._get_term_scores()
        
        for i, term_name in enumerate(term_names):
            if " x " not in term_name:  # 只获取主效应，排除交互项
                try:
                    # bins和scores
                    bin_vals = bins[i][0] if len(bins) > i else np.array([])
                    scores = term_scores[i] if len(term_scores) > i else np.array([])
                    shape_functions[term_name] = (bin_vals, scores)
                except (IndexError, KeyError):
                    continue
        
        return shape_functions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        if not self.is_fitted:
            return {'status': 'not fitted'}
        
        # 统计主效应和交互项
        term_names = self._get_term_names()
        main_effects = [t for t in term_names if " x " not in t]
        interactions = [t for t in term_names if " x " in t]
        
        # 兼容新旧版本的intercept属性
        intercept = getattr(self.model, 'intercept_', None)
        if intercept is None:
            intercept = getattr(self.model, 'intercept', None)
        
        return {
            'n_features': len(self.feature_names),
            'n_terms': len(term_names),
            'n_main_effects': len(main_effects),
            'n_interactions': len(interactions),
            'main_effects': main_effects,
            'interactions': interactions,
            'intercept': float(intercept) if intercept is not None else None,
            'feature_names': self.feature_names,
        }
    
    def evaluate(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        评估模型性能
        
        Parameters:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            评估指标字典
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = {}
        
        try:
            metrics['AUROC'] = roc_auc_score(y, y_prob)
        except Exception:
            metrics['AUROC'] = np.nan
            
        try:
            metrics['AUPRC'] = average_precision_score(y, y_prob)
        except Exception:
            metrics['AUPRC'] = np.nan
            
        try:
            metrics['Brier'] = brier_score_loss(y, y_prob)
        except Exception:
            metrics['Brier'] = np.nan
            
        try:
            metrics['F1'] = f1_score(y, y_pred)
        except Exception:
            metrics['F1'] = np.nan
            
        try:
            metrics['Accuracy'] = accuracy_score(y, y_pred)
        except Exception:
            metrics['Accuracy'] = np.nan
        
        return metrics


def get_ebm(
    verbose: bool = True, 
    random_state: int = 42,
    **kwargs
) -> EBMWrapper:
    """
    工厂函数：创建EBM实例
    
    Parameters:
        verbose: 是否输出详细信息
        random_state: 随机种子
        **kwargs: 其他EBMConfig参数
        
    Returns:
        EBMWrapper实例
    """
    config = EBMConfig(
        verbose=verbose,
        random_state=random_state,
        **kwargs
    )
    return EBMWrapper(config)


# =============================================================================
# 示例用法
# =============================================================================
if __name__ == "__main__":
    print("InterpretML EBM Wrapper - Demo")
    print("=" * 50)
    
    if not INTERPRET_AVAILABLE:
        print("InterpretML not installed. Please run: pip install interpret")
        exit(1)
    
    # 生成合成数据
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'age': np.random.normal(65, 10, n_samples),
        'heart_rate': np.random.normal(90, 15, n_samples),
        'sbp': np.random.normal(110, 20, n_samples),
        'creatinine': np.random.lognormal(-0.2, 0.5, n_samples),
    })
    
    # 生成二分类目标
    logits = 0.3 * (X['age'] - 65) + 0.2 * (X['heart_rate'] - 90) - 0.1 * (X['sbp'] - 110)
    prob = 1 / (1 + np.exp(-logits))
    y = (np.random.random(n_samples) < prob).astype(int)
    
    # 分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练EBM
    ebm = get_ebm(verbose=True, random_state=42)
    ebm.fit(X_train, y_train, task='classification')
    
    # 评估
    metrics = ebm.evaluate(X_test, y_test)
    print("\n[Test Metrics]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 特征重要性
    print("\n[Feature Importances]")
    importances = ebm.get_feature_importances()
    for name, imp in importances.head(10).items():
        print(f"  {name}: {imp:.4f}")
    
    # 模型摘要
    print("\n[Model Summary]")
    summary = ebm.get_model_summary()
    print(f"  Terms: {summary['n_terms']} ({summary['n_main_effects']} main + {summary['n_interactions']} interactions)")
    
    print("\nDemo completed!")
