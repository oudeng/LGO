"""
AutoScore_v2.py - Pure Python Implementation (R-Compatible)
============================================================
AutoScore的纯Python实现，与R版本保持一致

Version: 2.0.0
Date: Dec 4, 2025

主要改进（相比v1）：
1. 分箱策略：使用与R版本相同的非均匀分位数 [0, 0.05, 0.2, 0.8, 0.95, 1]
2. 评分推导：实现两步逻辑回归 + change_reference + round(coef/min(coef))
3. 评分计算：基于离散评分表，而非直接使用LogisticRegression.predict_proba

AutoScore是一个自动化的临床评分模型开发框架，包含6个模块：
1. 变量排序（随机森林特征重要性）
2. 变量选择（简约性分析）
3. 变量转换（分箱）
4. 评分推导（逻辑回归系数映射）
5. 评分微调
6. 性能评估

Reference: https://github.com/nliulab/AutoScore
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, brier_score_loss, average_precision_score


@dataclass
class AutoScoreConfig:
    """AutoScore配置"""
    # 变量排序
    n_trees: int = 100
    
    # 变量选择
    max_variables: int = 20
    
    # 分箱参数（与R版本一致）
    quantiles: List[float] = field(default_factory=lambda: [0, 0.05, 0.2, 0.8, 0.95, 1.0])
    
    # 评分模型
    max_score: int = 100
    
    # 其他
    random_state: int = 42
    verbose: bool = True


class AutoScore:
    """
    AutoScore纯Python实现（与R版本一致）
    
    自动化临床评分模型开发框架
    """
    
    def __init__(self, config: Optional[AutoScoreConfig] = None):
        """
        初始化AutoScore
        
        Parameters:
            config: 配置对象
        """
        self.config = config or AutoScoreConfig()
        
        # 状态变量
        self.ranking = None
        self.selected_variables = None
        self.cut_points = {}
        self.score_table = {}
        self.model = None
        self.intercept = None
        self.min_coef_normalizer = None
        
        # 训练数据统计（用于评分到概率的映射）
        self._score_prob_mapping = None
        
        if self.config.verbose:
            print("AutoScore v2.0 (R-Compatible Python) initialized")
    
    # =========================================================================
    # Module 1: Variable Ranking
    # =========================================================================
    
    def rank_variables(
        self, 
        train_data: pd.DataFrame, 
        outcome_col: str = 'label',
        method: str = 'rf'
    ) -> pd.Series:
        """
        Step 1: 使用随机森林对变量进行重要性排序
        
        Parameters:
            train_data: 训练数据
            outcome_col: 结果变量列名
            method: 'rf' (随机森林) 或 'auc' (单变量AUC)
            
        Returns:
            变量重要性排序 (pd.Series)
        """
        if self.config.verbose:
            print("\n[Module 1] Variable Ranking (Random Forest)")
            print("-" * 50)
        
        X = train_data.drop(columns=[outcome_col])
        y = train_data[outcome_col]
        
        # 处理分类变量
        X_encoded = self._encode_categoricals(X)
        
        if method == 'rf':
            # 训练随机森林
            rf = RandomForestClassifier(
                n_estimators=self.config.n_trees,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            rf.fit(X_encoded, y)
            
            # 获取特征重要性
            importance = pd.Series(
                rf.feature_importances_,
                index=X_encoded.columns
            ).sort_values(ascending=False)
        else:
            # 单变量AUC排序
            importance = {}
            for col in X_encoded.columns:
                try:
                    auc = roc_auc_score(y, X_encoded[col])
                    importance[col] = max(auc, 1 - auc)  # 处理负相关
                except:
                    importance[col] = 0.5
            importance = pd.Series(importance).sort_values(ascending=False)
        
        self.ranking = importance
        
        if self.config.verbose:
            print("Variable importance ranking:")
            for i, (var, imp) in enumerate(importance.head(15).items()):
                print(f"  {i+1:2d}. {var:<25} {imp:.4f}")
        
        return self.ranking
    
    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """编码分类变量"""
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X_encoded[col] = X[col].astype('category').cat.codes
        
        return X_encoded
    
    # =========================================================================
    # Module 2: Variable Selection (Parsimony)
    # =========================================================================
    
    def select_variables_parsimony(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        outcome_col: str = 'label',
        max_vars: Optional[int] = None,
        n_min: int = 1,
        n_max: int = 20
    ) -> Tuple[int, List[float]]:
        """
        Step 2: 简约性分析 - 选择最佳变量数量
        
        评估不同变量数量下的验证集AUC，找到性价比最高的变量数
        
        Parameters:
            train_data: 训练数据
            val_data: 验证数据
            outcome_col: 结果变量列名
            max_vars: 最大变量数 (deprecated, use n_max)
            n_min: 最小变量数
            n_max: 最大变量数
            
        Returns:
            (最佳变量数, AUC列表)
        """
        if self.config.verbose:
            print("\n[Module 2] Parsimony Analysis")
            print("-" * 50)
        
        if self.ranking is None:
            self.rank_variables(train_data, outcome_col)
        
        if max_vars is not None:
            n_max = max_vars
        n_max = min(n_max, self.config.max_variables, len(self.ranking))
        
        auc_list = []
        n_vars_list = list(range(n_min, n_max + 1))
        
        if self.config.verbose:
            print(f"{'n_vars':>8} | {'Val AUC':>10}")
            print("-" * 25)
        
        for n in n_vars_list:
            # 选择前n个变量
            selected = self.ranking.head(n).index.tolist()
            
            # 临时构建模型并评估
            try:
                temp_autoscore = AutoScore(AutoScoreConfig(
                    quantiles=self.config.quantiles,
                    random_state=self.config.random_state,
                    verbose=False
                ))
                temp_autoscore.ranking = self.ranking
                temp_autoscore.derive_scores(train_data, outcome_col, n)
                
                y_prob = temp_autoscore.predict_proba(val_data)
                y_true = val_data[outcome_col]
                auc = roc_auc_score(y_true, y_prob)
            except Exception as e:
                auc = 0.5
            
            auc_list.append(auc)
            
            if self.config.verbose:
                print(f"{n:>8} | {auc:>10.4f}")
        
        # 找到最佳变量数（考虑简约性）
        best_idx = self._find_parsimonious_n(auc_list)
        best_n = n_vars_list[best_idx]
        
        if self.config.verbose:
            print("-" * 25)
            print(f"Recommended: {best_n} variables (AUC={auc_list[best_idx]:.4f})")
        
        return best_n, auc_list
    
    def _find_parsimonious_n(self, auc_list: List[float], tolerance: float = 0.01) -> int:
        """
        找到简约的变量数
        
        选择AUC在最大值tolerance范围内的最小变量数
        """
        max_auc = max(auc_list)
        threshold = max_auc - tolerance
        
        for i, auc in enumerate(auc_list):
            if auc >= threshold:
                return i
        
        return np.argmax(auc_list)
    
    # =========================================================================
    # Module 3: Variable Transformation (Binning) - R-Compatible
    # =========================================================================
    
    def _quantile_binning(
        self,
        x: np.ndarray,
        quantiles: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        与原版AutoScore一致的分位数分箱
        
        原版R代码使用：categorize = "quantile", quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1)
        
        Parameters:
            x: 数值型数组
            quantiles: 分位数列表
        
        Returns:
            (binned_values, edges)
        """
        if quantiles is None:
            quantiles = self.config.quantiles
        
        x = np.asarray(x).ravel()
        valid_x = x[~np.isnan(x)]
        
        if len(valid_x) == 0:
            return np.zeros_like(x), np.array([0, 1])
        
        # 计算分位数边界
        edges = np.quantile(valid_x, quantiles)
        
        # 处理重复边界（当数据有大量相同值时）
        edges = np.unique(edges)
        
        if len(edges) < 2:
            edges = np.array([np.min(valid_x), np.max(valid_x)])
        
        # 执行分箱 - 使用内部边界
        # np.digitize with right=False: bins[i-1] <= x < bins[i]
        if len(edges) > 2:
            binned = np.digitize(x, edges[1:-1], right=False)
        else:
            binned = np.zeros(len(x), dtype=int)
        
        # 处理NaN
        binned = np.where(np.isnan(x), 0, binned)
        
        return binned.astype(int), edges
    
    def transform_variables(
        self,
        train_data: pd.DataFrame,
        outcome_col: str = 'label',
        n_variables: int = 6
    ) -> pd.DataFrame:
        """
        Step 3: 变量转换（分箱）- 使用与R版本一致的分位数
        
        将连续变量转换为有序分类变量
        
        Parameters:
            train_data: 训练数据
            outcome_col: 结果变量列名
            n_variables: 选择的变量数
            
        Returns:
            转换后的数据
        """
        if self.config.verbose:
            print("\n[Module 3] Variable Transformation (R-compatible binning)")
            print("-" * 50)
            print(f"Quantiles: {self.config.quantiles}")
        
        if self.ranking is None:
            self.rank_variables(train_data, outcome_col)
        
        # 选择变量
        self.selected_variables = self.ranking.head(n_variables).index.tolist()
        
        X = train_data[self.selected_variables].copy()
        X_binned = pd.DataFrame(index=X.index)
        
        if self.config.verbose:
            print(f"\nSelected {n_variables} variables: {self.selected_variables}")
            print("\nBinning results:")
        
        for var in self.selected_variables:
            if X[var].dtype in ['float64', 'float32', 'int64', 'int32', 'float', 'int']:
                # 数值变量：使用R版本的分位数分箱
                binned, edges = self._quantile_binning(X[var].values)
                X_binned[var] = binned
                self.cut_points[var] = edges.tolist()
                
                if self.config.verbose:
                    n_bins = len(edges) - 1
                    print(f"  {var:<25} {n_bins} bins, edges: {[f'{e:.2f}' for e in edges]}")
            else:
                # 分类变量
                categories = X[var].astype('category').cat.categories.tolist()
                X_binned[var] = X[var].astype('category').cat.codes
                self.cut_points[var] = categories
                
                if self.config.verbose:
                    print(f"  {var:<25} categorical, {len(categories)} levels")
        
        return X_binned
    
    # =========================================================================
    # Module 4: Score Derivation - R-Compatible (Two-step Logistic Regression)
    # =========================================================================
    
    def _change_reference(
        self,
        X_binned: pd.DataFrame,
        coef_vec: np.ndarray
    ) -> pd.DataFrame:
        """
        实现原版AutoScore的change_reference函数
        
        对于每个变量，根据系数方向调整参考类别
        原版R逻辑：使系数为正，即高bin值=高风险
        """
        X_adjusted = X_binned.copy()
        
        for i, var in enumerate(self.selected_variables):
            col_data = X_binned[var].values
            unique_bins = np.unique(col_data[~np.isnan(col_data)])
            
            if len(unique_bins) > 1 and coef_vec[i] < 0:
                # 负系数：翻转编码使得高bin值=高风险
                max_bin = int(np.max(unique_bins))
                X_adjusted[var] = max_bin - col_data
        
        return X_adjusted
    
    def derive_scores(
        self,
        train_data: pd.DataFrame,
        outcome_col: str = 'label',
        n_variables: int = 6
    ) -> Dict[str, Dict]:
        """
        Step 4: 评分推导（与R版本一致的两步逻辑回归）
        
        流程：
        1. 第一步逻辑回归，获取初始系数
        2. change_reference: 调整各变量的参考类别（确保系数为正）
        3. 第二步逻辑回归，获取最终系数
        4. 评分计算：round(coef / min(abs(coef)))
        
        Parameters:
            train_data: 训练数据
            outcome_col: 结果变量列名
            n_variables: 变量数
            
        Returns:
            评分表字典
        """
        if self.config.verbose:
            print("\n[Module 4] Score Derivation (R-compatible two-step LR)")
            print("-" * 50)
        
        # 变量转换
        X_binned = self.transform_variables(train_data, outcome_col, n_variables)
        y = train_data[outcome_col].values
        
        # ===== 第一步逻辑回归 =====
        model1 = LogisticRegression(
            max_iter=2000,
            random_state=self.config.random_state,
            solver='lbfgs',
            penalty=None,  # 原版AutoScore不使用正则化
            fit_intercept=True
        )
        model1.fit(X_binned, y)
        coef_vec1 = model1.coef_[0]
        
        if self.config.verbose:
            print("\nStep 1 - Initial coefficients:")
            for var, coef in zip(self.selected_variables, coef_vec1):
                print(f"  {var:<25} {coef:>10.4f}")
        
        # ===== 改变参考类别 =====
        X_adjusted = self._change_reference(X_binned, coef_vec1)
        
        # ===== 第二步逻辑回归 =====
        model2 = LogisticRegression(
            max_iter=2000,
            random_state=self.config.random_state,
            solver='lbfgs',
            penalty=None,
            fit_intercept=True
        )
        model2.fit(X_adjusted, y)
        coef_vec2 = model2.coef_[0]
        self.intercept = model2.intercept_[0]
        self.model = model2
        
        if self.config.verbose:
            print("\nStep 2 - Adjusted coefficients:")
            for var, coef in zip(self.selected_variables, coef_vec2):
                print(f"  {var:<25} {coef:>10.4f}")
        
        # ===== 评分计算（原版方法）=====
        # round(coef_vec / min(abs(coef_vec)))
        nonzero_coefs = coef_vec2[np.abs(coef_vec2) > 1e-10]
        if len(nonzero_coefs) > 0:
            self.min_coef_normalizer = np.min(np.abs(nonzero_coefs))
        else:
            self.min_coef_normalizer = 1.0
        
        self.score_table = {}
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Scoring Table")
            print("=" * 60)
            print(f"{'Variable':<25} {'Coef':>10} {'Score/bin':>12} {'Max Score':>12}")
            print("-" * 60)
        
        total_max_score = 0
        for i, var in enumerate(self.selected_variables):
            coef = coef_vec2[i]
            
            # 原版评分公式：每个bin增加的分数
            score_per_bin = int(round(coef / self.min_coef_normalizer)) if self.min_coef_normalizer > 0 else 0
            
            # 最大bin数
            n_bins = len(self.cut_points.get(var, [])) - 1
            if n_bins < 1:
                n_bins = 1
            
            # 该变量的最大可能分数
            max_score = score_per_bin * (n_bins - 1)  # bin从0开始，最大bin是n_bins-1
            
            # 检查是否需要翻转（第一步系数为负的变量已翻转）
            was_flipped = (coef_vec1[i] < 0)
            
            self.score_table[var] = {
                'coefficient': float(coef),
                'score_per_bin': score_per_bin,
                'max_score': max_score,
                'n_bins': n_bins,
                'cut_points': self.cut_points.get(var, []),
                'was_flipped': was_flipped
            }
            
            total_max_score += max(0, max_score)
            
            if self.config.verbose:
                print(f"{var:<25} {coef:>10.4f} {score_per_bin:>12} {max_score:>12}")
        
        if self.config.verbose:
            print("-" * 60)
            print(f"{'Intercept':<25} {self.intercept:>10.4f}")
            print(f"{'Min coef (normalizer)':<25} {self.min_coef_normalizer:>10.4f}")
            print(f"{'Total max score':<25} {total_max_score:>10}")
            print("=" * 60)
        
        # 建立评分到概率的映射（使用训练数据）
        self._build_score_probability_mapping(train_data, outcome_col)
        
        return self.score_table
    
    def _build_score_probability_mapping(
        self,
        train_data: pd.DataFrame,
        outcome_col: str
    ):
        """建立评分到概率的映射关系"""
        try:
            scores = self.compute_score(train_data)
            y = train_data[outcome_col].values
            
            # 按分数分组计算经验概率
            score_probs = {}
            for s in np.unique(scores):
                mask = scores == s
                if mask.sum() > 0:
                    score_probs[s] = np.mean(y[mask])
            
            self._score_prob_mapping = score_probs
            
            # 同时拟合一个logistic回归用于插值
            if len(np.unique(scores)) > 1:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=1000)
                lr.fit(scores.reshape(-1, 1), y)
                self._score_lr_model = lr
            else:
                self._score_lr_model = None
        except:
            self._score_prob_mapping = None
            self._score_lr_model = None
    
    # =========================================================================
    # Module 5: Score Fine-tuning
    # =========================================================================
    
    def fine_tune_scores(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        outcome_col: str = 'label',
        cut_vec: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Dict]:
        """
        Step 5: 评分微调
        
        允许手动调整分箱边界
        
        Parameters:
            train_data: 训练数据
            val_data: 验证数据
            outcome_col: 结果变量列名
            cut_vec: 手动指定的分箱边界 {变量名: [边界列表]}
        """
        if self.config.verbose:
            print("\n[Module 5] Score Fine-tuning")
            print("-" * 50)
        
        if cut_vec is not None:
            # 使用手动指定的分箱边界重新计算
            for var, edges in cut_vec.items():
                if var in self.cut_points:
                    self.cut_points[var] = edges
                    if self.config.verbose:
                        print(f"  Updated {var}: {edges}")
            
            # 重新推导评分
            self.derive_scores(train_data, outcome_col, len(self.selected_variables))
        else:
            if self.config.verbose:
                print("  No manual adjustments provided. Using default cut points.")
        
        return self.score_table
    
    # =========================================================================
    # Module 6: Performance Evaluation
    # =========================================================================
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        outcome_col: str = 'label'
    ) -> Dict[str, float]:
        """
        Step 6: 模型评估
        
        Parameters:
            test_data: 测试数据
            outcome_col: 结果变量列名
            
        Returns:
            性能指标字典
        """
        if self.config.verbose:
            print("\n[Module 6] Model Evaluation")
            print("-" * 50)
        
        y_true = test_data[outcome_col].values
        y_prob = self.predict_proba(test_data)
        y_pred = self.predict(test_data)
        scores = self.compute_score(test_data)
        
        metrics = {}
        
        # Discrimination
        try:
            metrics['AUROC'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['AUROC'] = np.nan
        
        try:
            metrics['AUPRC'] = average_precision_score(y_true, y_prob)
        except:
            metrics['AUPRC'] = np.nan
        
        # Calibration
        try:
            metrics['Brier'] = brier_score_loss(y_true, y_prob)
        except:
            metrics['Brier'] = np.nan
        
        # Classification
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Score statistics
        metrics['Score_mean'] = np.mean(scores)
        metrics['Score_std'] = np.std(scores)
        metrics['Score_min'] = np.min(scores)
        metrics['Score_max'] = np.max(scores)
        
        if self.config.verbose:
            print("Performance Metrics:")
            print(f"  {'AUROC':<15}: {metrics['AUROC']:.4f}")
            print(f"  {'AUPRC':<15}: {metrics['AUPRC']:.4f}")
            print(f"  {'Brier Score':<15}: {metrics['Brier']:.4f}")
            print(f"  {'Accuracy':<15}: {metrics['Accuracy']:.4f}")
            print(f"  {'F1':<15}: {metrics['F1']:.4f}")
            print(f"\nScore Distribution:")
            print(f"  Mean: {metrics['Score_mean']:.1f}, Std: {metrics['Score_std']:.1f}")
            print(f"  Range: [{metrics['Score_min']:.0f}, {metrics['Score_max']:.0f}]")
        
        return metrics
    
    # =========================================================================
    # Prediction Methods
    # =========================================================================
    
    def _transform_new_data(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        转换新数据并计算分数
        
        Returns:
            (binned_data, scores)
        """
        X_binned = pd.DataFrame(index=X.index)
        scores = np.zeros(len(X))
        
        for var in self.selected_variables:
            if var not in X.columns:
                raise ValueError(f"Variable '{var}' not found in data")
            
            info = self.score_table[var]
            col_data = X[var].values
            
            if var in self.cut_points and len(self.cut_points[var]) > 1:
                edges = np.array(self.cut_points[var])
                
                if isinstance(edges[0], (int, float, np.integer, np.floating)):
                    # 数值变量：重新分箱
                    if len(edges) > 2:
                        binned = np.digitize(col_data, edges[1:-1], right=False)
                    else:
                        binned = np.zeros(len(col_data), dtype=int)
                    
                    # 如果该变量在训练时被翻转，也需要翻转预测数据
                    if info.get('was_flipped', False):
                        max_bin = len(edges) - 2  # n_bins - 1
                        binned = max_bin - binned
                else:
                    # 分类变量
                    try:
                        cat_to_code = {cat: i for i, cat in enumerate(edges)}
                        binned = np.array([cat_to_code.get(v, 0) for v in col_data])
                    except:
                        binned = np.zeros(len(col_data), dtype=int)
            else:
                binned = np.zeros(len(col_data), dtype=int)
            
            X_binned[var] = binned
            
            # 计算该变量的分数贡献
            var_score = binned * info['score_per_bin']
            scores += np.nan_to_num(var_score, nan=0)
        
        return X_binned, scores
    
    def compute_score(self, data: pd.DataFrame) -> np.ndarray:
        """
        计算整数评分
        
        Parameters:
            data: 输入数据
            
        Returns:
            每个样本的评分数组
        """
        if not self.score_table:
            raise ValueError("Score table not generated. Run derive_scores first.")
        
        X = data[self.selected_variables]
        _, scores = self._transform_new_data(X)
        
        return np.round(scores).astype(int)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        预测概率
        
        使用评分到概率的映射，而非直接使用逻辑回归
        """
        scores = self.compute_score(data)
        
        # 方法1：使用训练时拟合的score->probability模型
        if hasattr(self, '_score_lr_model') and self._score_lr_model is not None:
            try:
                probs = self._score_lr_model.predict_proba(scores.reshape(-1, 1))[:, 1]
                return np.clip(probs, 0, 1)
            except:
                pass
        
        # 方法2：使用逻辑变换
        # 将分数映射到概率：sigmoid((score - mean) / scale)
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        if score_std < 1e-10:
            score_std = 1.0
        
        z = (scores - score_mean) / score_std
        probs = 1 / (1 + np.exp(-z))
        
        return np.clip(probs, 0, 1)
    
    def predict(self, data: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """预测类别"""
        return (self.predict_proba(data) >= threshold).astype(int)
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    
    def run_pipeline(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        outcome_col: str = 'label',
        n_variables: Optional[int] = None,
        use_combined_training: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整AutoScore流程
        
        Parameters:
            train_data: 训练数据
            val_data: 验证数据
            test_data: 测试数据
            outcome_col: 结果变量列名
            n_variables: 变量数（如果None则自动选择）
            use_combined_training: 是否使用train+val合并训练（用于公平比较）
            
        Returns:
            完整结果字典
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("AutoScore Pipeline v2.0 (R-Compatible)")
            print("=" * 60)
        
        results = {}
        
        # Module 1: Variable Ranking
        results['ranking'] = self.rank_variables(train_data, outcome_col)
        
        # Module 2: Parsimony Analysis
        if n_variables is None:
            best_n, auc_list = self.select_variables_parsimony(
                train_data, val_data, outcome_col
            )
            n_variables = best_n
            results['parsimony_auc'] = auc_list
        
        results['n_variables'] = n_variables
        
        # 选择训练数据
        if use_combined_training:
            combined_train = pd.concat([train_data, val_data], ignore_index=True)
            if self.config.verbose:
                print(f"\n[INFO] Using combined train+val data for final model ({len(combined_train)} samples)")
        else:
            combined_train = train_data
        
        # Module 3 & 4: Transform and Derive Scores
        results['score_table'] = self.derive_scores(
            combined_train, outcome_col, n_variables
        )
        
        # Module 5: Fine-tuning (optional)
        self.fine_tune_scores(combined_train, val_data, outcome_col)
        
        # Module 6: Evaluation
        results['train_metrics'] = self.evaluate(combined_train, outcome_col)
        results['val_metrics'] = self.evaluate(val_data, outcome_col)
        results['test_metrics'] = self.evaluate(test_data, outcome_col)
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Pipeline Complete")
            print("=" * 60)
        
        return results
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_scoring_table_df(self) -> pd.DataFrame:
        """获取评分表的DataFrame格式"""
        if not self.score_table:
            return pd.DataFrame()
        
        rows = []
        for var, info in self.score_table.items():
            n_bins = info['n_bins']
            score_per_bin = info['score_per_bin']
            
            for bin_idx in range(n_bins):
                bin_score = bin_idx * score_per_bin
                
                # 获取bin的范围描述
                edges = info.get('cut_points', [])
                if len(edges) > bin_idx + 1:
                    if isinstance(edges[0], (int, float, np.integer, np.floating)):
                        if bin_idx == 0:
                            range_str = f"< {edges[1]:.2f}"
                        elif bin_idx == n_bins - 1:
                            range_str = f">= {edges[-2]:.2f}"
                        else:
                            range_str = f"[{edges[bin_idx]:.2f}, {edges[bin_idx+1]:.2f})"
                    else:
                        range_str = str(edges[bin_idx]) if bin_idx < len(edges) else "N/A"
                else:
                    range_str = f"bin_{bin_idx}"
                
                rows.append({
                    'Variable': var,
                    'Bin': bin_idx,
                    'Range': range_str,
                    'Score': bin_score
                })
        
        return pd.DataFrame(rows)
    
    def to_dict(self) -> Dict[str, Any]:
        """导出模型配置"""
        return {
            'selected_variables': self.selected_variables,
            'score_table': self.score_table,
            'cut_points': self.cut_points,
            'intercept': self.intercept,
            'min_coef_normalizer': self.min_coef_normalizer,
            'config': {
                'quantiles': self.config.quantiles,
                'max_score': self.config.max_score,
                'random_state': self.config.random_state
            }
        }
    
    def to_json(self, filepath: str):
        """保存模型到JSON文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_autoscore(verbose: bool = True, **kwargs) -> AutoScore:
    """
    获取AutoScore实例
    
    Parameters:
        verbose: 是否输出详细信息
        **kwargs: 传递给配置的参数
        
    Returns:
        AutoScore实例
    """
    config = AutoScoreConfig(verbose=verbose, **kwargs)
    return AutoScore(config)


# =============================================================================
# Version Info
# =============================================================================
__version__ = "2.0.0"
__changelog__ = """
v2.0.0 (Dec 4, 2025):
- R-compatible implementation
- Fixed binning: uses quantiles=[0, 0.05, 0.2, 0.8, 0.95, 1] like R version
- Fixed score derivation: two-step logistic regression with change_reference
- Fixed score calculation: round(coef / min(abs(coef)))
- Added use_combined_training option for fair comparison with LGO
- Added score-to-probability mapping
- Added comprehensive scoring table export

v1.0.0:
- Initial pure Python implementation
"""

if __name__ == "__main__":
    print(f"AutoScore v{__version__} (R-Compatible Python Implementation)")
    print("Key improvements:")
    print("  - R-compatible quantile binning: [0, 0.05, 0.2, 0.8, 0.95, 1]")
    print("  - Two-step logistic regression with change_reference")
    print("  - Score calculation: round(coef / min(abs(coef)))")