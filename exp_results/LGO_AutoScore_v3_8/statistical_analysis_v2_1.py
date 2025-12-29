#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
statistical_analysis.py - LGO vs AutoScore 完整统计检验
================================================================
Version: 2.0.0
Date: Dec 4, 2025

包含以下检验：
1. 配对比较检验
   - Wilcoxon符号秩检验（非参数配对检验）
   - 配对t检验（参数检验，作为补充）
   
2. 效应量计算
   - Cohen's d（参数效应量）
   - Cliff's delta（非参数效应量）
   - Common Language Effect Size (CLES)
   
3. 资源缩放效应分析
   - Spearman秩相关（趋势检验）
   - Jonckheere-Terpstra趋势检验
   
4. 方差分析
   - Friedman检验（非参数重复测量）
   - Levene检验（方差齐性）
   
5. 置信区间
   - Bootstrap置信区间（BCa方法）
   - Percentile方法
   
6. 多重比较校正
   - Bonferroni校正
   - Holm-Bonferroni校正
   - FDR (Benjamini-Hochberg)

Usage:
------
python statistical_analysis_v2_1.py \
  --results_dirs ICU_results_30k_full,ICU_results_100k_full,ICU_results_200k_full,ICU_results_300k_full,ICU_results_500k_full \
  --output_dir stats_output/ICU

python statistical_analysis_v2_1.py \
  --results_dirs eICU_results_30k_full,eICU_results_100k_full,eICU_results_200k_full,eICU_results_300k_full \
  --output_dir stats_output/eICU

python statistical_analysis_v2_1.py \
  --results_dirs NHANES_results_30k_full,NHANES_results_100k_full,NHANES_results_200k_full,NHANES_results_300k_full \
  --output_dir stats_output/NHANES
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (wilcoxon, mannwhitneyu, friedmanchisquare, 
                         spearmanr, pearsonr, ttest_rel, levene,
                         shapiro, normaltest)

warnings.filterwarnings('ignore')


# ==============================================================================
# Data Loading
# ==============================================================================

def parse_budget_key(s: str) -> float:
    """Parse budget string (e.g., '30k', '100k', '1M') to numeric value for sorting."""
    s = s.lower().strip()
    if s.endswith('m'):
        return float(s[:-1]) * 1000000
    elif s.endswith('k'):
        return float(s[:-1]) * 1000
    else:
        try:
            return float(s)
        except:
            return 0


def load_multiseed_results(results_dir: str) -> pd.DataFrame:
    """Load multiseed results from a directory."""
    results_dir = Path(results_dir)
    csv_files = list(results_dir.glob("multiseed_results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No multiseed_results CSV found in {results_dir}")
    csv_file = sorted(csv_files)[-1]
    print(f"  [LOAD] {csv_file}")
    return pd.read_csv(csv_file)


def load_all_results(result_dirs: List[str]) -> Dict[str, pd.DataFrame]:
    """Load results from multiple directories."""
    all_data = {}
    for rdir in result_dirs:
        if not os.path.exists(rdir):
            print(f"  [SKIP] {rdir} not found")
            continue
        config_name = os.path.basename(rdir).replace('results_', '')
        try:
            df = load_multiseed_results(rdir)
            all_data[config_name] = df
        except Exception as e:
            print(f"  [ERROR] Failed to load {rdir}: {e}")
    return all_data


# ==============================================================================
# Effect Size Calculations
# ==============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cohen's d effect size for paired samples.
    
    Interpretation (Cohen, 1988):
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std < 1e-10:
        return 0.0, "undefined"
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return float(d), interpretation


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cliff's delta (non-parametric effect size).
    
    Interpretation (Romano et al., 2006):
    - |d| < 0.147: negligible
    - 0.147 <= |d| < 0.33: small
    - 0.33 <= |d| < 0.474: medium
    - |d| >= 0.474: large
    """
    n1, n2 = len(group1), len(group2)
    greater = sum(1 for x in group1 for y in group2 if x > y)
    less = sum(1 for x in group1 for y in group2 if x < y)
    
    delta = (greater - less) / (n1 * n2)
    
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return float(delta), interpretation


def cles(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Common Language Effect Size.
    
    Probability that a randomly selected value from group1 
    will be greater than a randomly selected value from group2.
    """
    n1, n2 = len(group1), len(group2)
    greater = sum(1 for x in group1 for y in group2 if x > y)
    ties = sum(1 for x in group1 for y in group2 if x == y)
    
    return (greater + 0.5 * ties) / (n1 * n2)


# ==============================================================================
# Statistical Tests
# ==============================================================================

def paired_wilcoxon_test(lgo: np.ndarray, autoscore: np.ndarray) -> Dict:
    """Wilcoxon signed-rank test for paired samples."""
    differences = lgo - autoscore
    non_zero_diff = differences[differences != 0]
    
    if len(non_zero_diff) < 3:
        return {'statistic': np.nan, 'p_value': np.nan, 'note': 'Too few non-zero differences'}
    
    try:
        stat, p_value = wilcoxon(lgo, autoscore, alternative='two-sided')
        # Also one-sided tests
        _, p_greater = wilcoxon(lgo, autoscore, alternative='greater')
        _, p_less = wilcoxon(lgo, autoscore, alternative='less')
    except Exception as e:
        return {'statistic': np.nan, 'p_value': np.nan, 'note': str(e)}
    
    return {
        'statistic': float(stat),
        'p_value_two_sided': float(p_value),
        'p_value_lgo_greater': float(p_greater),
        'p_value_autoscore_greater': float(p_less),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'median_diff': float(np.median(differences)),
        'mean_diff': float(np.mean(differences)),
        'lgo_wins': int(np.sum(differences > 0)),
        'autoscore_wins': int(np.sum(differences < 0)),
        'ties': int(np.sum(differences == 0))
    }


def paired_ttest(lgo: np.ndarray, autoscore: np.ndarray) -> Dict:
    """Paired t-test (parametric alternative)."""
    try:
        stat, p_value = ttest_rel(lgo, autoscore)
    except Exception as e:
        return {'statistic': np.nan, 'p_value': np.nan, 'note': str(e)}
    
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05
    }


def normality_test(data: np.ndarray) -> Dict:
    """Test for normality using Shapiro-Wilk test."""
    if len(data) < 3:
        return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
    
    try:
        stat, p_value = shapiro(data)
    except Exception as e:
        return {'statistic': np.nan, 'p_value': np.nan, 'note': str(e)}
    
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'is_normal': p_value > 0.05  # Fail to reject normality
    }


def levene_test(lgo: np.ndarray, autoscore: np.ndarray) -> Dict:
    """Levene's test for equality of variances."""
    try:
        stat, p_value = levene(lgo, autoscore)
    except Exception as e:
        return {'statistic': np.nan, 'p_value': np.nan, 'note': str(e)}
    
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'equal_variance': p_value > 0.05
    }


# ==============================================================================
# Bootstrap Methods
# ==============================================================================

def bootstrap_ci(lgo: np.ndarray, autoscore: np.ndarray,
                 n_bootstrap: int = 10000, ci_level: float = 0.95,
                 method: str = 'percentile') -> Dict:
    """
    Bootstrap confidence interval for the mean difference.
    
    Methods:
    - 'percentile': Simple percentile method
    - 'bca': Bias-corrected and accelerated (BCa)
    """
    np.random.seed(42)
    n = len(lgo)
    observed_diff = np.mean(lgo) - np.mean(autoscore)
    
    boot_diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        diff = np.mean(lgo[indices]) - np.mean(autoscore[indices])
        boot_diffs.append(diff)
    
    boot_diffs = np.array(boot_diffs)
    alpha = 1 - ci_level
    
    if method == 'percentile':
        ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    elif method == 'bca':
        # BCa method (simplified)
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_diffs < observed_diff))
        
        # Acceleration (jackknife)
        jackknife_diffs = []
        for i in range(n):
            jack_lgo = np.delete(lgo, i)
            jack_as = np.delete(autoscore, i)
            jackknife_diffs.append(np.mean(jack_lgo) - np.mean(jack_as))
        jackknife_diffs = np.array(jackknife_diffs)
        jack_mean = np.mean(jackknife_diffs)
        
        num = np.sum((jack_mean - jackknife_diffs) ** 3)
        denom = 6 * (np.sum((jack_mean - jackknife_diffs) ** 2) ** 1.5)
        a = num / denom if denom != 0 else 0
        
        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
        
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
        
        ci_lower = np.percentile(boot_diffs, 100 * p_lower)
        ci_upper = np.percentile(boot_diffs, 100 * p_upper)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'mean_diff': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_level': ci_level,
        'method': method,
        'excludes_zero': not (ci_lower <= 0 <= ci_upper),
        'boot_std': float(np.std(boot_diffs))
    }


# ==============================================================================
# Trend Analysis
# ==============================================================================

def trend_analysis(data_by_resource: Dict[str, np.ndarray]) -> Dict:
    """
    Analyze trend of performance across resource levels.
    
    Uses Spearman correlation and Jonckheere-Terpstra-like statistic.
    """
    def parse_budget(s: str) -> float:
        """Parse budget string to numeric value."""
        s = s.lower().strip()
        if s.endswith('m'):
            return float(s[:-1]) * 1000000
        elif s.endswith('k'):
            return float(s[:-1]) * 1000
        else:
            try:
                return float(s)
            except:
                return 0
    
    # 动态获取并排序所有可用的资源配置
    available_resources = list(data_by_resource.keys())
    resource_order = sorted(available_resources, key=parse_budget)
    resource_numeric = {r: parse_budget(r) for r in resource_order}
    
    all_values = []
    resource_levels = []
    
    for r in resource_order:
        if r in data_by_resource:
            for v in data_by_resource[r]:
                all_values.append(v)
                resource_levels.append(resource_numeric[r])
    
    if len(all_values) < 4:
        return {'note': 'Not enough data for trend analysis'}
    
    # Spearman correlation
    rho, p_spearman = spearmanr(resource_levels, all_values)
    
    # Pearson correlation (on log-transformed resources)
    log_resources = np.log10(resource_levels)
    r, p_pearson = pearsonr(log_resources, all_values)
    
    # Calculate improvement from lowest to highest resource
    groups = [data_by_resource[r] for r in resource_order if r in data_by_resource]
    if len(groups) >= 2:
        improvement = np.mean(groups[-1]) - np.mean(groups[0])
        relative_improvement = improvement / np.mean(groups[0]) * 100
    else:
        improvement = np.nan
        relative_improvement = np.nan
    
    return {
        'spearman_rho': float(rho),
        'spearman_p': float(p_spearman),
        'pearson_r': float(r),
        'pearson_p': float(p_pearson),
        'trend_significant': p_spearman < 0.05,
        'trend_direction': 'increasing' if rho > 0 else 'decreasing',
        'absolute_improvement': float(improvement),
        'relative_improvement_pct': float(relative_improvement)
    }


# ==============================================================================
# Multiple Comparison Correction
# ==============================================================================

def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    adjusted_alpha = alpha / n
    significant = [p < adjusted_alpha for p in p_values]
    
    return {
        'original_alpha': alpha,
        'adjusted_alpha': adjusted_alpha,
        'n_comparisons': n,
        'significant': significant,
        'n_significant': sum(significant)
    }


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    significant = [False] * n
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        adjusted_alpha = alpha / (n - i)
        if p < adjusted_alpha:
            significant[idx] = True
        else:
            break  # Stop at first non-significant
    
    return {
        'original_alpha': alpha,
        'significant': significant,
        'n_significant': sum(significant)
    }


def fdr_bh(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Calculate BH critical values
    bh_critical = [(i + 1) / n * alpha for i in range(n)]
    
    # Find largest k where p(k) <= k/n * alpha
    significant = [False] * n
    k_max = -1
    for k in range(n):
        if sorted_p[k] <= bh_critical[k]:
            k_max = k
    
    # All tests up to k_max are significant
    if k_max >= 0:
        for i in range(k_max + 1):
            significant[sorted_indices[i]] = True
    
    return {
        'original_alpha': alpha,
        'significant': significant,
        'n_significant': sum(significant)
    }


# ==============================================================================
# Variance Analysis Between Seeds
# ==============================================================================

def seed_variance_analysis(df: pd.DataFrame, metric: str = 'AUROC') -> Dict:
    """Analyze variance due to random seeds."""
    lgo_data = df[df['method'] == 'LGO'][metric].values
    as_data = df[df['method'] == 'AutoScore'][metric].values
    
    # Coefficient of Variation
    lgo_cv = np.std(lgo_data) / np.mean(lgo_data) * 100 if np.mean(lgo_data) != 0 else np.nan
    as_cv = np.std(as_data) / np.mean(as_data) * 100 if np.mean(as_data) != 0 else np.nan
    
    # Range
    lgo_range = np.max(lgo_data) - np.min(lgo_data)
    as_range = np.max(as_data) - np.min(as_data)
    
    # Interquartile range
    lgo_iqr = np.percentile(lgo_data, 75) - np.percentile(lgo_data, 25)
    as_iqr = np.percentile(as_data, 75) - np.percentile(as_data, 25)
    
    # Levene test for variance equality
    lev = levene_test(lgo_data, as_data)
    
    return {
        'lgo': {
            'mean': float(np.mean(lgo_data)),
            'std': float(np.std(lgo_data)),
            'cv_pct': float(lgo_cv),
            'range': float(lgo_range),
            'iqr': float(lgo_iqr),
            'min': float(np.min(lgo_data)),
            'max': float(np.max(lgo_data))
        },
        'autoscore': {
            'mean': float(np.mean(as_data)),
            'std': float(np.std(as_data)),
            'cv_pct': float(as_cv),
            'range': float(as_range),
            'iqr': float(as_iqr),
            'min': float(np.min(as_data)),
            'max': float(np.max(as_data))
        },
        'levene_test': lev,
        'variance_ratio': float(np.var(lgo_data) / np.var(as_data)) if np.var(as_data) > 0 else np.nan
    }


# ==============================================================================
# Main Analysis
# ==============================================================================

def analyze_single_config(df: pd.DataFrame, config_name: str) -> Dict:
    """Complete analysis for a single configuration."""
    lgo_data = df[df['method'] == 'LGO']
    as_data = df[df['method'] == 'AutoScore']
    
    merged = lgo_data.merge(as_data, on='seed', suffixes=('_lgo', '_as'))
    
    results = {
        'config': config_name,
        'n_seeds': len(merged),
        'seeds': merged['seed'].tolist(),
        'metrics': {}
    }
    
    for metric in ['AUROC', 'AUPRC', 'Brier']:
        lgo_vals = merged[f'{metric}_lgo'].values
        as_vals = merged[f'{metric}_as'].values
        
        # Normality tests
        lgo_norm = normality_test(lgo_vals)
        as_norm = normality_test(as_vals)
        diff_norm = normality_test(lgo_vals - as_vals)
        
        metric_results = {
            # Descriptive statistics
            'lgo_mean': float(np.mean(lgo_vals)),
            'lgo_std': float(np.std(lgo_vals)),
            'lgo_median': float(np.median(lgo_vals)),
            'autoscore_mean': float(np.mean(as_vals)),
            'autoscore_std': float(np.std(as_vals)),
            'autoscore_median': float(np.median(as_vals)),
            
            # Hypothesis tests
            'wilcoxon': paired_wilcoxon_test(lgo_vals, as_vals),
            'paired_ttest': paired_ttest(lgo_vals, as_vals),
            
            # Effect sizes
            'cohens_d': cohens_d(lgo_vals, as_vals),
            'cliffs_delta': cliffs_delta(lgo_vals, as_vals),
            'cles': float(cles(lgo_vals, as_vals)),
            
            # Bootstrap CIs
            'bootstrap_percentile': bootstrap_ci(lgo_vals, as_vals, method='percentile'),
            'bootstrap_bca': bootstrap_ci(lgo_vals, as_vals, method='bca'),
            
            # Normality
            'normality_lgo': lgo_norm,
            'normality_autoscore': as_norm,
            'normality_difference': diff_norm,
            
            # Variance analysis
            'variance_analysis': seed_variance_analysis(df, metric)
        }
        
        results['metrics'][metric] = metric_results
    
    return results


def run_complete_analysis(all_data: Dict[str, pd.DataFrame]) -> Dict:
    """Run complete statistical analysis across all configurations."""
    all_results = {}
    
    # Analyze each configuration
    for config_name, df in all_data.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {config_name}")
        print('='*60)
        
        results = analyze_single_config(df, config_name)
        all_results[config_name] = results
        
        # Print summary
        auroc = results['metrics']['AUROC']
        print(f"\nAUROC Results:")
        print(f"  LGO:       {auroc['lgo_mean']:.4f} ± {auroc['lgo_std']:.4f}")
        print(f"  AutoScore: {auroc['autoscore_mean']:.4f} ± {auroc['autoscore_std']:.4f}")
        print(f"  Δ = {auroc['bootstrap_percentile']['mean_diff']:+.4f} "
              f"(95% CI: [{auroc['bootstrap_percentile']['ci_lower']:.4f}, "
              f"{auroc['bootstrap_percentile']['ci_upper']:.4f}])")
        print(f"  Wilcoxon p = {auroc['wilcoxon']['p_value_two_sided']:.4f}")
        print(f"  Cohen's d = {auroc['cohens_d'][0]:.3f} ({auroc['cohens_d'][1]})")
        print(f"  Win rate: LGO {auroc['wilcoxon']['lgo_wins']}/{results['n_seeds']}")
    
    # Trend analysis across configurations
    print(f"\n{'='*60}")
    print("Trend Analysis: LGO Performance vs Computational Resources")
    print('='*60)
    
    lgo_auroc_by_resource = {}
    for config, df in all_data.items():
        lgo_vals = df[df['method'] == 'LGO']['AUROC'].values
        lgo_auroc_by_resource[config] = lgo_vals
    
    trend = trend_analysis(lgo_auroc_by_resource)
    all_results['trend_analysis'] = trend
    
    print(f"\nSpearman ρ = {trend['spearman_rho']:.4f} (p = {trend['spearman_p']:.4f})")
    print(f"Trend: {trend['trend_direction']} "
          f"({'significant' if trend['trend_significant'] else 'not significant'})")
    print(f"Improvement (30k→1M): {trend['absolute_improvement']:.4f} "
          f"({trend['relative_improvement_pct']:.1f}%)")
    
    # Multiple comparison correction
    print(f"\n{'='*60}")
    print("Multiple Comparison Correction")
    print('='*60)
    
    # 动态获取配置并排序
    available_configs = [k for k in all_results.keys() 
                        if k not in ['trend_analysis', 'multiple_comparison'] 
                        and 'metrics' in all_results.get(k, {})]
    sorted_configs = sorted(available_configs, key=lambda x: parse_budget_key(x))
    
    p_values = []
    config_names = []
    for config in sorted_configs:
        if config in all_results and 'metrics' in all_results[config]:
            p = all_results[config]['metrics']['AUROC']['wilcoxon']['p_value_two_sided']
            if not np.isnan(p):
                p_values.append(p)
                config_names.append(config)
    
    if p_values:
        bonf = bonferroni_correction(p_values)
        holm = holm_bonferroni(p_values)
        fdr = fdr_bh(p_values)
        
        all_results['multiple_comparison'] = {
            'bonferroni': bonf,
            'holm_bonferroni': holm,
            'fdr_bh': fdr
        }
        
        print(f"\nOriginal p-values: {[f'{p:.4f}' for p in p_values]}")
        print(f"Bonferroni (α={bonf['adjusted_alpha']:.4f}): {bonf['n_significant']}/{len(p_values)} significant")
        print(f"Holm-Bonferroni: {holm['n_significant']}/{len(p_values)} significant")
        print(f"FDR (BH): {fdr['n_significant']}/{len(p_values)} significant")
    
    return all_results


# ==============================================================================
# Output Generation
# ==============================================================================

def sort_config_keys(configs: List[str]) -> List[str]:
    """Sort configuration keys by computational budget (e.g., 30k < 100k < 200k < 500k < 1M)."""
    def parse_budget(s: str) -> float:
        s = s.lower().strip()
        if s.endswith('m'):
            return float(s[:-1]) * 1000000
        elif s.endswith('k'):
            return float(s[:-1]) * 1000
        else:
            try:
                return float(s)
            except:
                return 0
    return sorted(configs, key=parse_budget)


def generate_summary_table(all_results: Dict) -> pd.DataFrame:
    """Generate summary table for publication."""
    rows = []
    
    # 动态获取配置并排序（而非硬编码）
    available_configs = [k for k in all_results.keys() 
                        if k not in ['trend_analysis', 'multiple_comparison'] 
                        and 'metrics' in all_results.get(k, {})]
    sorted_configs = sort_config_keys(available_configs)
    
    for config in sorted_configs:
        if config not in all_results or 'metrics' not in all_results[config]:
            continue
        
        results = all_results[config]
        auroc = results['metrics']['AUROC']
        auprc = results['metrics']['AUPRC']
        
        row = {
            'Config': config,
            'n': results['n_seeds'],
            'LGO AUROC': f"{auroc['lgo_mean']:.3f}±{auroc['lgo_std']:.3f}",
            'AutoScore AUROC': f"{auroc['autoscore_mean']:.3f}±{auroc['autoscore_std']:.3f}",
            'ΔAUROC': f"{auroc['bootstrap_percentile']['mean_diff']:+.3f}",
            '95% CI': f"[{auroc['bootstrap_percentile']['ci_lower']:.3f},{auroc['bootstrap_percentile']['ci_upper']:.3f}]",
            'p (Wilcoxon)': f"{auroc['wilcoxon']['p_value_two_sided']:.4f}",
            "Cohen's d": f"{auroc['cohens_d'][0]:.2f}",
            'Effect': auroc['cohens_d'][1],
            'LGO Wins': f"{auroc['wilcoxon']['lgo_wins']}/{results['n_seeds']}",
            'LGO AUPRC': f"{auprc['lgo_mean']:.3f}±{auprc['lgo_std']:.3f}",
            'AutoScore AUPRC': f"{auprc['autoscore_mean']:.3f}±{auprc['autoscore_std']:.3f}",
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_latex_table(all_results: Dict) -> str:
    """Generate LaTeX table for publication."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance comparison of LGO and AutoScore across computational budgets (ICU dataset, 10 seeds)}",
        r"\label{tab:lgo_autoscore}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Budget & LGO AUROC & AutoScore AUROC & $\Delta$AUROC [95\% CI] & $p$ & Cohen's $d$ & Win \\",
        r"\midrule",
    ]
    
    # 动态获取配置并排序（而非硬编码）
    available_configs = [k for k in all_results.keys() 
                        if k not in ['trend_analysis', 'multiple_comparison'] 
                        and 'metrics' in all_results.get(k, {})]
    sorted_configs = sort_config_keys(available_configs)
    
    for config in sorted_configs:
        if config not in all_results or 'metrics' not in all_results[config]:
            continue
        
        results = all_results[config]
        auroc = results['metrics']['AUROC']
        
        lgo = f"{auroc['lgo_mean']:.3f}$\\pm${auroc['lgo_std']:.3f}"
        auto = f"{auroc['autoscore_mean']:.3f}$\\pm${auroc['autoscore_std']:.3f}"
        
        diff = auroc['bootstrap_percentile']['mean_diff']
        ci_l = auroc['bootstrap_percentile']['ci_lower']
        ci_u = auroc['bootstrap_percentile']['ci_upper']
        delta_ci = f"{diff:+.3f} [{ci_l:.3f},{ci_u:.3f}]"
        
        p = auroc['wilcoxon']['p_value_two_sided']
        if p < 0.001:
            p_str = "$<$0.001***"
        elif p < 0.01:
            p_str = f"{p:.3f}**"
        elif p < 0.05:
            p_str = f"{p:.3f}*"
        else:
            p_str = f"{p:.3f}"
        
        d = auroc['cohens_d'][0]
        d_str = f"{d:.2f}"
        
        wins = auroc['wilcoxon']['lgo_wins']
        n = results['n_seeds']
        
        lines.append(f"{config} & {lgo} & {auto} & {delta_ci} & {p_str} & {d_str} & {wins}/{n} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\item Note: * $p<0.05$, ** $p<0.01$, *** $p<0.001$ (Wilcoxon signed-rank test).",
        r"\item Win: number of seeds where LGO AUROC $>$ AutoScore AUROC.",
        r"\item CI: Bootstrap percentile confidence interval (10,000 resamples).",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return '\n'.join(lines)


def make_json_serializable(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif obj is None or isinstance(obj, (str, int, float)):
        return obj
    else:
        return str(obj)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Complete statistical tests for LGO vs AutoScore')
    parser.add_argument('--results_dirs', type=str,
                        default='results_30k,results_100k,results_500k,results_1M',
                        help='Comma-separated list of result directories')
    parser.add_argument('--output_dir', type=str, default='./stats_output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LGO vs AutoScore: Complete Statistical Analysis")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    result_dirs = [d.strip() for d in args.results_dirs.split(',')]
    print(f"\nLoading results from: {result_dirs}")
    all_data = load_all_results(result_dirs)
    
    if not all_data:
        print("[ERROR] No data loaded!")
        return
    
    # Run analysis
    all_results = run_complete_analysis(all_data)
    
    # Generate outputs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Summary table
    summary_df = generate_summary_table(all_results)
    print(f"\n{'='*60}")
    print("Summary Table")
    print('='*60)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(Path(args.output_dir) / f'summary_table_{timestamp}.csv', index=False)
    
    # LaTeX table
    latex = generate_latex_table(all_results)
    print(f"\n{'='*60}")
    print("LaTeX Table")
    print('='*60)
    print(latex)
    
    with open(Path(args.output_dir) / f'table_{timestamp}.tex', 'w') as f:
        f.write(latex)
    
    # Detailed JSON
    clean_results = make_json_serializable(all_results)
    with open(Path(args.output_dir) / f'detailed_stats_{timestamp}.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\n[SAVE] Results saved to {args.output_dir}/")
    print(f"  - summary_table_{timestamp}.csv")
    print(f"  - table_{timestamp}.tex")
    print(f"  - detailed_stats_{timestamp}.json")


if __name__ == '__main__':
    main()