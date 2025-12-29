#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LGO vs EBM Statistical Analysis
================================
Version: 1.0.0
Date: Dec 7, 2025

统计分析脚本，用于对LGO vs EBM多种子实验结果进行严格的统计检验。

包含的统计测试:
1. Wilcoxon signed-rank test (非参数配对检验)
2. Paired t-test (参数配对检验)
3. 效应量: Cohen's d, Cliff's delta, CLES
4. Bootstrap置信区间 (percentile和BCa方法)
5. 多重比较校正: Bonferroni, Holm-Bonferroni, FDR

Usage:
------
python statistical_analysis.py \
  --results_dirs results_30k,results_100k,results_200k,results_500k \
  --output_dir stats_output

python statistical_analysis.py \
  --results_csv results/multiseed_results_*.csv \
  --output_dir stats_output

"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')


# =============================================================================
# Effect Size Functions
# =============================================================================
def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d for paired samples."""
    diff = x - y
    if np.std(diff, ddof=1) < 1e-10:
        return 0.0 if np.abs(np.mean(diff)) < 1e-10 else np.inf * np.sign(np.mean(diff))
    return np.mean(diff) / np.std(diff, ddof=1)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    more = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1
    return (more - less) / (n_x * n_y)


def cles(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Common Language Effect Size (probability that X > Y)."""
    return (1 + cliffs_delta(x, y)) / 2


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Bootstrap Functions
# =============================================================================
def bootstrap_mean_ci(
    x: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for mean."""
    rng = np.random.RandomState(random_state)
    n = len(x)
    
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_means.append(np.mean(x[idx]))
    
    boot_means = np.array(boot_means)
    mean_val = np.mean(x)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    
    return mean_val, lower, upper


def bootstrap_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for difference of means (paired)."""
    rng = np.random.RandomState(random_state)
    diff = x - y
    n = len(diff)
    
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_diffs.append(np.mean(diff[idx]))
    
    boot_diffs = np.array(boot_diffs)
    mean_diff = np.mean(diff)
    lower = np.percentile(boot_diffs, 100 * alpha / 2)
    upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    
    return mean_diff, lower, upper


# =============================================================================
# Multiple Comparison Corrections
# =============================================================================
def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Bonferroni correction."""
    m = len(p_values)
    return [(p * m, p * m < alpha) for p in p_values]


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    
    results = [None] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        reject = p < adjusted_alpha
        results[orig_idx] = (p, reject)
    
    return results


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    
    results = [None] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = (rank + 1) * alpha / m
        reject = p < threshold
        results[orig_idx] = (p, reject)
    
    return results


# =============================================================================
# Statistical Analysis
# =============================================================================
class StatisticalAnalysis:
    """Comprehensive statistical analysis for LGO vs EBM comparison."""
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize with results dataframe.
        
        Parameters:
            results_df: DataFrame with columns [seed, method, AUROC, AUPRC, Brier, F1, Accuracy, train_time]
        """
        self.df = results_df
        self.lgo_data = results_df[results_df['method'] == 'LGO']
        self.ebm_data = results_df[results_df['method'] == 'EBM']
        self.metrics = ['AUROC', 'AUPRC', 'Brier', 'F1', 'Accuracy']
        self.results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all statistical tests."""
        print("\n" + "=" * 70)
        print("Statistical Analysis: LGO vs EBM")
        print("=" * 70)
        
        for metric in self.metrics:
            self.results[metric] = self._analyze_metric(metric)
        
        # Win rate analysis
        self.results['win_rate'] = self._calculate_win_rate()
        
        # Multiple comparison corrections
        self.results['corrections'] = self._apply_corrections()
        
        return self.results
    
    def _analyze_metric(self, metric: str) -> Dict[str, Any]:
        """Analyze a single metric."""
        lgo_vals = self.lgo_data[metric].values
        ebm_vals = self.ebm_data[metric].values
        
        # Make sure they're paired by seed
        lgo_by_seed = self.lgo_data.set_index('seed')[metric]
        ebm_by_seed = self.ebm_data.set_index('seed')[metric]
        common_seeds = sorted(set(lgo_by_seed.index) & set(ebm_by_seed.index))
        
        lgo_paired = np.array([lgo_by_seed[s] for s in common_seeds])
        ebm_paired = np.array([ebm_by_seed[s] for s in common_seeds])
        
        result = {
            'n_samples': len(common_seeds),
            'lgo_mean': float(np.mean(lgo_paired)),
            'lgo_std': float(np.std(lgo_paired, ddof=1)),
            'ebm_mean': float(np.mean(ebm_paired)),
            'ebm_std': float(np.std(ebm_paired, ddof=1)),
            'diff_mean': float(np.mean(lgo_paired - ebm_paired)),
            'diff_std': float(np.std(lgo_paired - ebm_paired, ddof=1)),
        }
        
        # Wilcoxon signed-rank test
        try:
            stat, p_wilcoxon = scipy_stats.wilcoxon(lgo_paired, ebm_paired)
            result['wilcoxon_stat'] = float(stat)
            result['wilcoxon_p'] = float(p_wilcoxon)
        except Exception:
            result['wilcoxon_stat'] = np.nan
            result['wilcoxon_p'] = np.nan
        
        # Paired t-test
        try:
            stat, p_ttest = scipy_stats.ttest_rel(lgo_paired, ebm_paired)
            result['ttest_stat'] = float(stat)
            result['ttest_p'] = float(p_ttest)
        except Exception:
            result['ttest_stat'] = np.nan
            result['ttest_p'] = np.nan
        
        # Effect sizes
        result['cohens_d'] = float(cohens_d_paired(lgo_paired, ebm_paired))
        result['cohens_d_interpretation'] = interpret_cohens_d(result['cohens_d'])
        result['cliffs_delta'] = float(cliffs_delta(lgo_paired, ebm_paired))
        result['cles'] = float(cles(lgo_paired, ebm_paired))
        
        # Bootstrap CI for difference
        mean_diff, ci_lower, ci_upper = bootstrap_diff_ci(lgo_paired, ebm_paired)
        result['bootstrap_diff_mean'] = float(mean_diff)
        result['bootstrap_ci_lower'] = float(ci_lower)
        result['bootstrap_ci_upper'] = float(ci_upper)
        
        # Win/tie/loss
        result['lgo_wins'] = int(np.sum(lgo_paired > ebm_paired))
        result['ties'] = int(np.sum(lgo_paired == ebm_paired))
        result['ebm_wins'] = int(np.sum(lgo_paired < ebm_paired))
        
        print(f"\n[{metric}]")
        print(f"  LGO: {result['lgo_mean']:.4f} ± {result['lgo_std']:.4f}")
        print(f"  EBM: {result['ebm_mean']:.4f} ± {result['ebm_std']:.4f}")
        print(f"  Diff: {result['diff_mean']:.4f} ± {result['diff_std']:.4f}")
        print(f"  Wilcoxon p={result['wilcoxon_p']:.4f}, t-test p={result['ttest_p']:.4f}")
        print(f"  Cohen's d={result['cohens_d']:.2f} ({result['cohens_d_interpretation']})")
        print(f"  95% CI: [{result['bootstrap_ci_lower']:.4f}, {result['bootstrap_ci_upper']:.4f}]")
        print(f"  Win/Tie/Loss: {result['lgo_wins']}/{result['ties']}/{result['ebm_wins']}")
        
        return result
    
    def _calculate_win_rate(self) -> Dict[str, Any]:
        """Calculate overall win rate."""
        total_wins = sum(self.results[m]['lgo_wins'] for m in self.metrics)
        total_ties = sum(self.results[m]['ties'] for m in self.metrics)
        total_losses = sum(self.results[m]['ebm_wins'] for m in self.metrics)
        total = total_wins + total_ties + total_losses
        
        return {
            'lgo_total_wins': total_wins,
            'ties': total_ties,
            'ebm_total_wins': total_losses,
            'lgo_win_rate': total_wins / total if total > 0 else 0.0,
        }
    
    def _apply_corrections(self) -> Dict[str, Any]:
        """Apply multiple comparison corrections."""
        p_values = [self.results[m]['wilcoxon_p'] for m in self.metrics]
        
        bonf = bonferroni_correction(p_values)
        holm = holm_bonferroni_correction(p_values)
        fdr = fdr_correction(p_values)
        
        corrections = {}
        for i, metric in enumerate(self.metrics):
            corrections[metric] = {
                'original_p': p_values[i],
                'bonferroni_adjusted': bonf[i][0],
                'bonferroni_significant': bonf[i][1],
                'holm_significant': holm[i][1],
                'fdr_significant': fdr[i][1],
            }
        
        return corrections
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{LGO vs EBM Statistical Comparison}",
            r"\label{tab:lgo_ebm_comparison}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Metric & LGO & EBM & $\Delta$ & 95\% CI & p-value \\",
            r"\midrule",
        ]
        
        for metric in self.metrics:
            r = self.results[metric]
            lgo_str = f"{r['lgo_mean']:.3f}±{r['lgo_std']:.3f}"
            ebm_str = f"{r['ebm_mean']:.3f}±{r['ebm_std']:.3f}"
            diff_str = f"{r['diff_mean']:+.3f}"
            ci_str = f"[{r['bootstrap_ci_lower']:.3f}, {r['bootstrap_ci_upper']:.3f}]"
            p_str = f"{r['wilcoxon_p']:.3f}"
            if r['wilcoxon_p'] < 0.001:
                p_str = r"$<$0.001"
            lines.append(f"{metric} & {lgo_str} & {ebm_str} & {diff_str} & {ci_str} & {p_str} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_summary_csv(self) -> pd.DataFrame:
        """Generate summary CSV."""
        rows = []
        for metric in self.metrics:
            r = self.results[metric]
            rows.append({
                'Metric': metric,
                'LGO_mean': r['lgo_mean'],
                'LGO_std': r['lgo_std'],
                'EBM_mean': r['ebm_mean'],
                'EBM_std': r['ebm_std'],
                'Diff': r['diff_mean'],
                'CI_lower': r['bootstrap_ci_lower'],
                'CI_upper': r['bootstrap_ci_upper'],
                'Wilcoxon_p': r['wilcoxon_p'],
                'Cohens_d': r['cohens_d'],
                'LGO_wins': r['lgo_wins'],
                'EBM_wins': r['ebm_wins'],
            })
        return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='LGO vs EBM Statistical Analysis'
    )
    parser.add_argument('--results_csv', type=str, default=None,
                        help='Path to results CSV file')
    parser.add_argument('--results_dirs', type=str, default=None,
                        help='Comma-separated results directories')
    parser.add_argument('--output_dir', type=str, default='./stats_output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load data
    all_dfs = []
    
    if args.results_csv:
        csv_path = Path(args.results_csv)
        if csv_path.exists():
            all_dfs.append(pd.read_csv(csv_path))
        else:
            # Try glob
            for f in Path('.').glob(args.results_csv):
                all_dfs.append(pd.read_csv(f))
    
    if args.results_dirs:
        for dir_path in args.results_dirs.split(','):
            dir_path = Path(dir_path.strip())
            csv_files = list(dir_path.glob('multiseed_results_*.csv'))
            for f in csv_files:
                all_dfs.append(pd.read_csv(f))
    
    if not all_dfs:
        print("[ERROR] No results files found")
        return
    
    results_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[INFO] Loaded {len(results_df)} rows from {len(all_dfs)} file(s)")
    
    # Run analysis
    analysis = StatisticalAnalysis(results_df)
    results = analysis.run_all_tests()
    
    # Save outputs
    summary_csv = analysis.generate_summary_csv()
    summary_path = Path(args.output_dir) / f'summary_table_{timestamp}.csv'
    summary_csv.to_csv(summary_path, index=False)
    print(f"\n[SAVE] Summary table: {summary_path}")
    
    latex_table = analysis.generate_latex_table()
    latex_path = Path(args.output_dir) / f'table_{timestamp}.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"[SAVE] LaTeX table: {latex_path}")
    
    # Save detailed JSON
    json_path = Path(args.output_dir) / f'detailed_stats_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[SAVE] Detailed stats: {json_path}")
    
    print("\n[DONE] Statistical analysis completed!")


if __name__ == '__main__':
    main()
