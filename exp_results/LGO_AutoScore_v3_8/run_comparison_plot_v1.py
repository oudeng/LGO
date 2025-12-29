#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_comparison_plot_v1.py — Visualization for LGO vs AutoScore Comparison
=========================================================================
Version: 1.2.0
Date: Dec 5, 2025

独立的可视化脚本，从pkl文件读取详细数据后生成图表。
完全忠实于run_comparison_v2_1.py的plot_comprehensive_comparison方法。

Usage:
------

# 同时生成两种图
python run_comparison_plot_v1.py \
  --results_csv ICU_results_30k_full/multiseed_results_*.csv \
  --detailed_pkl ICU_results_30k_full/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_30k_full

python run_comparison_plot_v1.py \
  --results_csv ICU_results_100k_full/multiseed_results_*.csv \
  --detailed_pkl ICU_results_100k_full/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_100k_full

python run_comparison_plot_v1.py \
  --results_csv ICU_results_200k_full/multiseed_results_*.csv \
  --detailed_pkl ICU_results_200k_full/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_200k_full

python run_comparison_plot_v1.py \
  --results_csv ICU_results_300k_full/multiseed_results_*.csv \
  --detailed_pkl ICU_results_300k_full/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_300k_full

python run_comparison_plot_v1.py \
  --results_csv ICU_results_500k_full/multiseed_results_*.csv \
  --detailed_pkl ICU_results_500k_full/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_500k_full

2)Fair
python run_comparison_plot_v1.py \
  --results_csv ICU_results_30k_fair/multiseed_results_*.csv \
  --detailed_pkl ICU_results_30k_fair/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_30k_fair

3)Raw
python run_comparison_plot_v1.py \
  --results_csv ICU_results_30k_raw/multiseed_results_*.csv \
  --detailed_pkl ICU_results_30k_raw/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_30k_raw

python run_comparison_plot_v1.py \
  --results_csv ICU_results_100k_raw/multiseed_results_*.csv \
  --detailed_pkl ICU_results_100k_raw/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_100k_raw

python run_comparison_plot_v1.py \
  --results_csv ICU_results_200k_raw/multiseed_results_*.csv \
  --detailed_pkl ICU_results_200k_raw/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_200k_raw

python run_comparison_plot_v1.py \
  --results_csv ICU_results_300k_raw/multiseed_results_*.csv \
  --detailed_pkl ICU_results_300k_raw/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_300k_raw

python run_comparison_plot_v1.py \
  --results_csv ICU_results_500k_raw/multiseed_results_*.csv \
  --detailed_pkl ICU_results_500k_raw/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --plot_all \
  --output_dir ICU_results_500k_raw
  

"""

import argparse
import pickle
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

# 设置matplotlib参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2


# =============================================================================
# Multi-Seed Summary Plot
# =============================================================================
def plot_multiseed_summary(
    results_df: pd.DataFrame, 
    output_path: str, 
    title_suffix: str = ''
) -> plt.Figure:
    """
    Generate summary visualization for multi-seed experiment.
    
    Creates a 2x2 figure:
    - a) AUROC comparison (mean ± std)
    - b) AUPRC comparison
    - c) Brier Score comparison (lower is better)
    - d) Win rate across seeds
    """
    lgo_df = results_df[results_df['method'] == 'LGO']
    auto_df = results_df[results_df['method'] == 'AutoScore']
    merged = lgo_df.merge(auto_df, on='seed', suffixes=('_lgo', '_as'))
    
    metrics = ['AUROC', 'AUPRC', 'Brier', 'F1']
    
    # ------------------------------------------------------------------
    # NEW: handle the corner case where all metric values are NaN / Inf.
    #      This typically happens when the core script failed to compute
    #      metrics (e.g. wrong task type / missing binarization).
    # ------------------------------------------------------------------
    metric_cols: List[str] = []
    for m in metrics:
        metric_cols.extend([f'{m}_lgo', f'{m}_as'])
    
    # Replace Inf with NaN, then check if there is any valid number
    finite_mask = merged[metric_cols].replace([np.inf, -np.inf], np.nan).notna()
    if finite_mask.sum().sum() == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5,
            "No valid metric values found in multiseed results.\n"
            "Please check run_comparison_v2_5.py configuration\n"
            "(e.g., task type / binarize_threshold).",
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes, wrap=True,
        )
        ax.axis('off')
        plt.suptitle(f'Multi-Seed Summary{title_suffix}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[WARN] No valid metrics. Placeholder plot saved to: {output_path}")
        plt.close()
        return fig
    
    # ---- 以下为原有逻辑，保持不变 -----------------------------------
    stats = {}
    for m in metrics:
        stats[m] = {
            'lgo_mean': merged[f'{m}_lgo'].mean(),
            'lgo_std': merged[f'{m}_lgo'].std(),
            'as_mean': merged[f'{m}_as'].mean(),
            'as_std': merged[f'{m}_as'].std(),
        }
        if m == 'Brier':
            stats[m]['lgo_wins'] = int((merged[f'{m}_lgo'] < merged[f'{m}_as']).sum())
        else:
            stats[m]['lgo_wins'] = int((merged[f'{m}_lgo'] > merged[f'{m}_as']).sum())
        stats[m]['n_seeds'] = len(merged)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    C_LGO, C_AUTO = '#2ecc71', '#3498db'
    
    # Panel a): AUROC
    ax = axes[0, 0]
    x = np.arange(2)
    means = [stats['AUROC']['lgo_mean'], stats['AUROC']['as_mean']]
    stds = [stats['AUROC']['lgo_std'], stats['AUROC']['as_std']]
    ax.bar(x, means, yerr=stds, color=[C_LGO, C_AUTO], capsize=8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('a) AUROC (mean ± std)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['LGO', 'AutoScore'])
    y_max = max(means[i] + stds[i] for i in range(2))
    ax.set_ylim(0.65, min(1.05, y_max + 0.08))
    ax.grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        label_y = min(m + s + 0.015, ax.get_ylim()[1] - 0.02)
        ax.text(i, label_y, f'{m:.3f}±{s:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel b): AUPRC
    ax = axes[0, 1]
    means = [stats['AUPRC']['lgo_mean'], stats['AUPRC']['as_mean']]
    stds = [stats['AUPRC']['lgo_std'], stats['AUPRC']['as_std']]
    ax.bar(x, means, yerr=stds, color=[C_LGO, C_AUTO], capsize=8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
    ax.set_title('b) AUPRC (mean ± std)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['LGO', 'AutoScore'])
    y_max = max(means[i] + stds[i] for i in range(2))
    ax.set_ylim(0.55, min(1.05, y_max + 0.08))
    ax.grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        label_y = min(m + s + 0.015, ax.get_ylim()[1] - 0.02)
        ax.text(i, label_y, f'{m:.3f}±{s:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel c): Brier Score
    ax = axes[1, 0]
    means = [stats['Brier']['lgo_mean'], stats['Brier']['as_mean']]
    stds = [stats['Brier']['lgo_std'], stats['Brier']['as_std']]
    ax.bar(x, means, yerr=stds, color=[C_LGO, C_AUTO], capsize=8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Brier Score', fontsize=12, fontweight='bold')
    ax.set_title('c) Brier Score (lower is better)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['LGO', 'AutoScore'])
    y_max = max(means[i] + stds[i] for i in range(2))
    ax.set_ylim(0, y_max + 0.05)
    ax.grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.008, f'{m:.3f}±{s:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Panel d): Win Rates
    ax = axes[1, 1]
    n_seeds = stats['AUROC']['n_seeds']
    metrics_win = ['AUROC', 'AUPRC', 'Brier', 'F1']
    x_win = np.arange(len(metrics_win))
    lgo_wins = [stats[m]['lgo_wins'] for m in metrics_win]
    colors = [C_LGO if w >= n_seeds/2 else '#95a5a6' for w in lgo_wins]
    ax.bar(x_win, lgo_wins, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=n_seeds/2, color='gray', linestyle='--', linewidth=1.5, label='50%')
    ax.set_ylabel('LGO Win Count', fontsize=12, fontweight='bold')
    ax.set_title(f'd) LGO Win Rate (out of {n_seeds} seeds)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_win)
    ax.set_xticklabels(metrics_win)
    ax.set_ylim(0, n_seeds + 1)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(loc='lower right')
    for i, w in enumerate(lgo_wins):
        if w > 0:
            ax.text(i, w - 0.5, f'{w}/{n_seeds}', ha='center', va='top', 
                   fontsize=11, fontweight='bold', color='white')
    
    plt.suptitle(f'Multi-Seed Summary{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[SAVE] Summary plot saved to: {output_path}")
    plt.close()
    return fig


# =============================================================================
# Single Seed Detailed 6-Panel Plot (忠实于v2.1的plot_comprehensive_comparison)
# =============================================================================
def plot_comprehensive_comparison(
    lgo_results: Dict,
    autoscore_results: Dict,
    y_test: np.ndarray,
    save_path: str,
    seed: int = None,
    title_suffix: str = ''
) -> plt.Figure:
    """
    Create comprehensive comparison visualization.
    
    完全忠实于run_comparison_v2_1.py的plot_comprehensive_comparison方法。
    
    6个子图:
    1. ROC Curves
    2. Precision-Recall Curves
    3. Calibration Curves
    4. Performance Metrics Comparison (bar chart)
    5. AutoScore Distribution by Outcome
    6. Probability Distribution
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {'LGO': '#2ecc71', 'AutoScore': '#3498db'}
    
    # 1. ROC Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for name, results, color in [('LGO', lgo_results, colors['LGO']),
                                  ('AutoScore', autoscore_results, colors['AutoScore'])]:
        y_prob = results.get('test_probabilities', np.array([]))
        if len(y_prob) > 0:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = results.get('test_metrics', {}).get('AUROC', np.nan)
            ax1.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC={auc_val:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. PR Curves
    ax2 = fig.add_subplot(gs[0, 1])
    for name, results, color in [('LGO', lgo_results, colors['LGO']),
                                  ('AutoScore', autoscore_results, colors['AutoScore'])]:
        y_prob = results.get('test_probabilities', np.array([]))
        if len(y_prob) > 0:
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap = results.get('test_metrics', {}).get('AUPRC', np.nan)
            ax2.plot(recall, precision, color=color, lw=2,
                    label=f'{name} (AP={ap:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Calibration Curves
    ax3 = fig.add_subplot(gs[0, 2])
    for name, results, color in [('LGO', lgo_results, colors['LGO']),
                                  ('AutoScore', autoscore_results, colors['AutoScore'])]:
        y_prob = results.get('test_probabilities', np.array([]))
        if len(y_prob) > 0:
            try:
                prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
                ax3.plot(prob_pred, prob_true, 's-', color=color, lw=2,
                        label=name, markersize=6)
            except:
                pass
    ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax3.set_xlabel('Mean Predicted Probability')
    ax3.set_ylabel('Fraction of Positives')
    ax3.set_title('Calibration Curves')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Comparison Bar Chart
    ax4 = fig.add_subplot(gs[1, 0])
    metrics = ['AUROC', 'AUPRC', 'F1', 'Accuracy']
    x = np.arange(len(metrics))
    width = 0.35
    
    lgo_vals = [lgo_results.get('test_metrics', {}).get(m, 0) for m in metrics]
    as_vals = [autoscore_results.get('test_metrics', {}).get(m, 0) for m in metrics]
    
    ax4.bar(x - width/2, lgo_vals, width, label='LGO', color=colors['LGO'])
    ax4.bar(x + width/2, as_vals, width, label='AutoScore', color=colors['AutoScore'])
    ax4.set_ylabel('Score')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Score Distribution (AutoScore only)
    ax5 = fig.add_subplot(gs[1, 1])
    test_scores = autoscore_results.get('test_scores', np.array([]))
    if len(test_scores) > 0 and len(test_scores) == len(y_test):
        ax5.hist(test_scores[y_test == 0], bins=20, alpha=0.7,
                label='Negative', color='green', density=True)
        ax5.hist(test_scores[y_test == 1], bins=20, alpha=0.7,
                label='Positive', color='red', density=True)
        ax5.set_xlabel('AutoScore Risk Score')
        ax5.set_ylabel('Density')
        ax5.set_title('AutoScore Distribution by Outcome')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'AutoScore scores\nnot available', 
                transform=ax5.transAxes, ha='center', va='center', fontsize=12)
        ax5.set_title('AutoScore Distribution by Outcome')
    ax5.grid(True, alpha=0.3)
    
    # 6. Probability Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    for name, results, color in [('LGO', lgo_results, colors['LGO']),
                                  ('AutoScore', autoscore_results, colors['AutoScore'])]:
        y_prob = results.get('test_probabilities', np.array([]))
        if len(y_prob) > 0:
            ax6.hist(y_prob, bins=30, alpha=0.5, label=name, color=color, density=True)
    ax6.set_xlabel('Predicted Probability')
    ax6.set_ylabel('Density')
    ax6.set_title('Probability Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Title
    seed_str = f' (Seed={seed})' if seed is not None else ''
    calib_info = ""
    if lgo_results.get('is_calibrated', False):
        calib_info = f" [LGO: {lgo_results.get('calibration_method', 'platt')} calibrated]"
    
    plt.suptitle(f'LGO vs AutoScore: Fair Comparison{seed_str}{calib_info}{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[SAVE] Plot saved to: {save_path}")
    plt.close()
    
    return fig


def plot_single_seed_from_pkl(
    seed_data: Dict,
    output_path: str,
    title_suffix: str = ''
) -> Optional[plt.Figure]:
    """
    从pkl数据生成单种子6子图可视化。
    
    将pkl数据转换为plot_comprehensive_comparison需要的格式。
    """
    seed = seed_data.get('seed', '?')
    y_test = np.asarray(seed_data.get('y_test', []))
    
    lgo_data = seed_data.get('lgo', {})
    auto_data = seed_data.get('autoscore', {})
    
    # 检查数据
    lgo_prob = np.asarray(lgo_data.get('test_probabilities', []))
    auto_prob = np.asarray(auto_data.get('test_probabilities', []))
    
    if len(y_test) == 0:
        print(f"[ERROR] y_test is empty for seed {seed}")
        return None
    
    if len(lgo_prob) == 0 and len(auto_prob) == 0:
        print(f"[ERROR] Both LGO and AutoScore probabilities are empty for seed {seed}")
        return None
    
    # 构建与plot_comprehensive_comparison兼容的数据结构
    lgo_results = {
        'test_probabilities': lgo_prob,
        'test_metrics': lgo_data.get('test_metrics', {}),
        'is_calibrated': lgo_data.get('is_calibrated', False),
        'calibration_method': lgo_data.get('calibration_method', 'none'),
    }
    
    autoscore_results = {
        'test_probabilities': auto_prob,
        'test_scores': np.asarray(auto_data.get('test_scores', [])),
        'test_metrics': auto_data.get('test_metrics', {}),
    }
    
    return plot_comprehensive_comparison(
        lgo_results=lgo_results,
        autoscore_results=autoscore_results,
        y_test=y_test,
        save_path=output_path,
        seed=seed,
        title_suffix=title_suffix
    )


def load_detailed_data(pkl_path: str) -> List[Dict]:
    """Load detailed data from pickle file."""
    print(f"[LOAD] Loading detailed data from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def find_seed_data(detailed_data: List[Dict], seed: int) -> Optional[Dict]:
    """Find data for a specific seed."""
    for d in detailed_data:
        if d.get('seed') == seed:
            return d
    return None


# =============================================================================
# Scaling Analysis Plot
# =============================================================================
def plot_scaling_analysis(
    results_dirs: List[str],
    output_path: str
) -> Optional[plt.Figure]:
    """Generate scaling analysis plot comparing multiple configurations."""
    from scipy.stats import wilcoxon
    
    data = {
        'configs': [], 'lgo_auroc_mean': [], 'lgo_auroc_std': [],
        'autoscore_auroc_mean': [], 'autoscore_auroc_std': [],
        'lgo_auprc_mean': [], 'autoscore_auprc_mean': [],
        'wilcoxon_p': [], 'win_rate': [],
        'delta_auroc': [], 'ci_lower': [], 'ci_upper': []
    }
    
    for rdir in results_dirs:
        config = Path(rdir).name.replace('results_', '')
        csv_files = list(Path(rdir).glob("multiseed_results_*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV found in {rdir}, skipping...")
            continue
        
        df = pd.read_csv(sorted(csv_files)[-1])
        lgo = df[df['method'] == 'LGO']
        auto = df[df['method'] == 'AutoScore']
        merged = lgo.merge(auto, on='seed', suffixes=('_lgo', '_as'))
        
        data['configs'].append(config)
        data['lgo_auroc_mean'].append(merged['AUROC_lgo'].mean())
        data['lgo_auroc_std'].append(merged['AUROC_lgo'].std())
        data['autoscore_auroc_mean'].append(merged['AUROC_as'].mean())
        data['autoscore_auroc_std'].append(merged['AUROC_as'].std())
        data['lgo_auprc_mean'].append(merged['AUPRC_lgo'].mean())
        data['autoscore_auprc_mean'].append(merged['AUPRC_as'].mean())
        
        try:
            _, p = wilcoxon(merged['AUROC_lgo'], merged['AUROC_as'])
        except:
            p = 1.0
        data['wilcoxon_p'].append(float(p))
        
        diff = merged['AUROC_lgo'] - merged['AUROC_as']
        data['win_rate'].append(int((diff > 0).sum()))
        data['delta_auroc'].append(float(diff.mean()))
        
        np.random.seed(42)
        n = len(merged)
        boot_diffs = [merged['AUROC_lgo'].sample(n, replace=True).mean() - 
                      merged['AUROC_as'].sample(n, replace=True).mean() 
                      for _ in range(10000)]
        data['ci_lower'].append(float(np.percentile(boot_diffs, 2.5)))
        data['ci_upper'].append(float(np.percentile(boot_diffs, 97.5)))
    
    if not data['configs']:
        print("[ERROR] No data loaded!")
        return None
    
    configs = data['configs']
    x = np.arange(len(configs))
    C_LGO, C_AUTO = '#2ecc71', '#3498db'
    C_SIG, C_NSIG = '#2ecc71', '#95a5a6'
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('LGO vs AutoScore: Performance Scaling Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    ax1 = axes[0, 0]
    ax1.errorbar(x, data['lgo_auroc_mean'], yerr=data['lgo_auroc_std'], 
                 fmt='o-', color=C_LGO, lw=2.5, ms=10, capsize=6, label='LGO')
    ax1.errorbar(x, data['autoscore_auroc_mean'], yerr=data['autoscore_auroc_std'],
                 fmt='s--', color=C_AUTO, lw=2.5, ms=10, capsize=6, label='AutoScore')
    ax1.set_xticks(x); ax1.set_xticklabels(configs)
    ax1.set_xlabel('Computational Budget', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax1.set_title('A. AUROC vs Resources', fontsize=13, fontweight='bold')
    ax1.legend(); ax1.set_ylim(0.80, 1.0); ax1.grid(alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(x, data['lgo_auprc_mean'], 'o-', color=C_LGO, lw=2.5, ms=10, label='LGO')
    ax2.plot(x, data['autoscore_auprc_mean'], 's--', color=C_AUTO, lw=2.5, ms=10, label='AutoScore')
    ax2.set_xticks(x); ax2.set_xticklabels(configs)
    ax2.set_xlabel('Computational Budget', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
    ax2.set_title('B. AUPRC vs Resources', fontsize=13, fontweight='bold')
    ax2.legend(); ax2.set_ylim(0.70, 1.0); ax2.grid(alpha=0.3)
    
    ax3 = axes[1, 0]
    n_seeds = 10
    colors = [C_SIG if w >= n_seeds*0.8 else C_NSIG for w in data['win_rate']]
    ax3.bar(x, data['win_rate'], color=colors, edgecolor='black', width=0.6)
    ax3.axhline(y=n_seeds/2, color='gray', ls='--', lw=1.5)
    ax3.set_xticks(x); ax3.set_xticklabels(configs)
    ax3.set_xlabel('Computational Budget', fontsize=12, fontweight='bold')
    ax3.set_ylabel('LGO Win Count', fontsize=12, fontweight='bold')
    ax3.set_title('C. LGO Win Rate', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, n_seeds + 1); ax3.grid(alpha=0.3, axis='y')
    
    ax4 = axes[1, 1]
    yerr_lo = [d - l for d, l in zip(data['delta_auroc'], data['ci_lower'])]
    yerr_hi = [u - d for d, u in zip(data['delta_auroc'], data['ci_upper'])]
    bar_colors = [C_NSIG if l < 0 else C_SIG for l in data['ci_lower']]
    ax4.bar(x, data['delta_auroc'], yerr=[yerr_lo, yerr_hi], 
            color=bar_colors, edgecolor='black', capsize=6, width=0.6)
    ax4.axhline(y=0, color='#e74c3c', lw=2)
    ax4.set_xticks(x); ax4.set_xticklabels(configs)
    ax4.set_xlabel('Computational Budget', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Δ AUROC', fontsize=12, fontweight='bold')
    ax4.set_title('D. AUROC Difference with 95% CI', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[SAVE] Scaling analysis plot saved to: {output_path}")
    plt.close()
    return fig


# =============================================================================
# Main CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Visualization for LGO vs AutoScore Comparison (v1.2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Generate multi-seed summary plot
python run_comparison_plot_v1.py \\
  --results_csv results_500k/multiseed_results_*.csv \\
  --output_dir results_500k

# Generate detailed 6-panel plot for specific seed (requires pkl file)
python run_comparison_plot_v1.py \\
  --detailed_pkl results_500k/multiseed_detailed_*.pkl \\
  --png_seed 1 \\
  --output_dir results_500k

# Generate both summary and detailed plots
python run_comparison_plot_v1.py \\
  --results_csv results_500k/multiseed_results_*.csv \\
  --detailed_pkl results_500k/multiseed_detailed_*.pkl \\
  --png_seed 1 \\
  --plot_all \\
  --output_dir results_500k
        """
    )
    
    parser.add_argument('--results_csv', type=str, default=None,
                        help='Path to multiseed_results_*.csv file')
    parser.add_argument('--detailed_pkl', type=str, default=None,
                        help='Path to multiseed_detailed_*.pkl file (for 6-panel plots)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for plots')
    parser.add_argument('--png_seed', type=int, default=None,
                        help='Generate detailed 6-panel comparison for specific seed')
    parser.add_argument('--plot_all', action='store_true',
                        help='Generate both summary and single-seed plots')
    parser.add_argument('--scaling_dirs', type=str, default=None,
                        help='Comma-separated result directories for scaling analysis')
    parser.add_argument('--title_suffix', type=str, default='',
                        help='Additional text for plot titles')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load CSV results
    results_df = None
    if args.results_csv:
        csv_files = glob(args.results_csv)
        if csv_files:
            csv_path = sorted(csv_files)[-1]
            print(f"[LOAD] Loading results from: {csv_path}")
            results_df = pd.read_csv(csv_path)
        else:
            print(f"[ERROR] No CSV files found matching: {args.results_csv}")
    
    # Load detailed data
    detailed_data = None
    if args.detailed_pkl:
        pkl_files = glob(args.detailed_pkl)
        if pkl_files:
            pkl_path = sorted(pkl_files)[-1]
            detailed_data = load_detailed_data(pkl_path)
        else:
            print(f"[WARN] No PKL files found matching: {args.detailed_pkl}")
    
    # Generate multi-seed summary
    if results_df is not None:
        if args.png_seed is None or args.plot_all:
            summary_path = Path(args.output_dir) / f'multiseed_summary_{timestamp}.png'
            plot_multiseed_summary(results_df, str(summary_path), args.title_suffix)
    
    # Generate single seed detailed plot
    if args.png_seed is not None:
        if detailed_data is not None:
            seed_data = find_seed_data(detailed_data, args.png_seed)
            if seed_data:
                detail_path = Path(args.output_dir) / f'comparison_seed{args.png_seed}_{timestamp}.png'
                plot_single_seed_from_pkl(seed_data, str(detail_path), args.title_suffix)
            else:
                print(f"[ERROR] Seed {args.png_seed} not found in detailed data")
                print(f"[INFO] Available seeds: {[d['seed'] for d in detailed_data]}")
        else:
            print("[ERROR] Need --detailed_pkl to generate 6-panel seed plot")
            print("[INFO] Run run_comparison_v2_3.py with --seeds to generate pkl file")
    
    # Scaling analysis
    if args.scaling_dirs:
        dirs = [d.strip() for d in args.scaling_dirs.split(',')]
        scaling_path = Path(args.output_dir) / f'scaling_analysis_{timestamp}.png'
        plot_scaling_analysis(dirs, str(scaling_path))
    
    if results_df is None and detailed_data is None and not args.scaling_dirs:
        print("[ERROR] No input specified. Use --results_csv, --detailed_pkl, or --scaling_dirs")
        parser.print_help()
        sys.exit(1)
    
    print("\n[DONE] Visualization complete!")


if __name__ == '__main__':
    main()