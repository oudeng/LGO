#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LGO vs InterpretML (EBM) Visualization
=======================================
Version: 1.0.0
Date: Dec 7, 2025

可视化脚本，用于生成LGO与EBM对比实验的各种图表：
1. ROC曲线对比
2. 校准曲线
3. 特征重要性对比
4. 多种子实验汇总
5. EBM形状函数可视化

Usage:
------
# ICU
# 从详细数据文件生成可视化
python run_lgo_interpret_plot.py \
  --detailed_pkl ICU_results_30k/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --output_dir ICU_results_30k

# 生成所有种子的图
python run_lgo_interpret_plot.py \
  --detailed_pkl ICU_results_30k/multiseed_detailed_*.pkl \
  --plot_all \
  --output_dir ICU_results_30k

# 生成汇总图
python run_lgo_interpret_plot.py \
  --results_csv ICU_results_30k/multiseed_results_*.csv \
  --plot_summary \
  --output_dir ICU_results_30k

# eICU
# 从详细数据文件生成可视化
python run_lgo_interpret_plot.py \
  --detailed_pkl eICU_results_30k/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --output_dir eICU_results_30k

# 生成所有种子的图
python run_lgo_interpret_plot.py \
  --detailed_pkl eICU_results_30k/multiseed_detailed_*.pkl \
  --plot_all \
  --output_dir eICU_results_30k

# 生成汇总图
python run_lgo_interpret_plot.py \
  --results_csv eICU_results_30k/multiseed_results_*.csv \
  --plot_summary \
  --output_dir eICU_results_30k

# NHANES
# 从详细数据文件生成可视化
python run_lgo_interpret_plot.py \
  --detailed_pkl NHANES_results_30k/multiseed_detailed_*.pkl \
  --png_seed 1 \
  --output_dir NHANES_results_30k

# 生成所有种子的图
python run_lgo_interpret_plot.py \
  --detailed_pkl NHANES_results_30k/multiseed_detailed_*.pkl \
  --plot_all \
  --output_dir NHANES_results_30k

# 生成汇总图
python run_lgo_interpret_plot.py \
  --results_csv NHANES_results_30k/multiseed_results_*.csv \
  --plot_summary \
  --output_dir NHANES_results_30k

"""

import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

# 设置中文字体（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_roc_curves(
    y_true: np.ndarray,
    lgo_probs: np.ndarray,
    ebm_probs: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "ROC Curve Comparison"
) -> plt.Axes:
    """Plot ROC curves for LGO and EBM."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # LGO ROC
    fpr_lgo, tpr_lgo, _ = roc_curve(y_true, lgo_probs)
    auc_lgo = auc(fpr_lgo, tpr_lgo)
    ax.plot(fpr_lgo, tpr_lgo, 'b-', lw=2, label=f'LGO (AUROC = {auc_lgo:.3f})')
    
    # EBM ROC
    fpr_ebm, tpr_ebm, _ = roc_curve(y_true, ebm_probs)
    auc_ebm = auc(fpr_ebm, tpr_ebm)
    ax.plot(fpr_ebm, tpr_ebm, 'r-', lw=2, label=f'EBM (AUROC = {auc_ebm:.3f})')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_pr_curves(
    y_true: np.ndarray,
    lgo_probs: np.ndarray,
    ebm_probs: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Precision-Recall Curve"
) -> plt.Axes:
    """Plot Precision-Recall curves for LGO and EBM."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # LGO PR
    precision_lgo, recall_lgo, _ = precision_recall_curve(y_true, lgo_probs)
    ap_lgo = average_precision_score(y_true, lgo_probs)
    ax.plot(recall_lgo, precision_lgo, 'b-', lw=2, label=f'LGO (AP = {ap_lgo:.3f})')
    
    # EBM PR
    precision_ebm, recall_ebm, _ = precision_recall_curve(y_true, ebm_probs)
    ap_ebm = average_precision_score(y_true, ebm_probs)
    ax.plot(recall_ebm, precision_ebm, 'r-', lw=2, label=f'EBM (AP = {ap_ebm:.3f})')
    
    # Baseline
    baseline = np.mean(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_calibration_curves(
    y_true: np.ndarray,
    lgo_probs: np.ndarray,
    ebm_probs: np.ndarray,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 10,
    title: str = "Calibration Curve"
) -> plt.Axes:
    """Plot calibration curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # LGO calibration
    prob_true_lgo, prob_pred_lgo = calibration_curve(y_true, lgo_probs, n_bins=n_bins, strategy='uniform')
    ax.plot(prob_pred_lgo, prob_true_lgo, 'b-o', lw=2, label='LGO')
    
    # EBM calibration
    prob_true_ebm, prob_pred_ebm = calibration_curve(y_true, ebm_probs, n_bins=n_bins, strategy='uniform')
    ax.plot(prob_pred_ebm, prob_true_ebm, 'r-s', lw=2, label='EBM')
    
    # Perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_probability_histograms(
    y_true: np.ndarray,
    lgo_probs: np.ndarray,
    ebm_probs: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Probability Distribution"
) -> plt.Axes:
    """Plot probability histograms by class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    bins = np.linspace(0, 1, 21)
    
    # LGO
    ax.hist(lgo_probs[y_true == 0], bins=bins, alpha=0.4, color='blue', label='LGO (y=0)', density=True)
    ax.hist(lgo_probs[y_true == 1], bins=bins, alpha=0.4, color='cyan', label='LGO (y=1)', density=True)
    
    # EBM
    ax.hist(ebm_probs[y_true == 0], bins=bins, alpha=0.4, color='red', label='EBM (y=0)', density=True)
    ax.hist(ebm_probs[y_true == 1], bins=bins, alpha=0.4, color='orange', label='EBM (y=1)', density=True)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper center', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_metrics_comparison(
    lgo_metrics: Dict[str, float],
    ebm_metrics: Dict[str, float],
    ax: Optional[plt.Axes] = None,
    title: str = "Metrics Comparison"
) -> plt.Axes:
    """Plot bar chart comparing metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    metrics = ['AUROC', 'AUPRC', 'F1', 'Accuracy']
    x = np.arange(len(metrics))
    width = 0.35
    
    lgo_vals = [lgo_metrics.get(m, 0) for m in metrics]
    ebm_vals = [ebm_metrics.get(m, 0) for m in metrics]
    
    bars1 = ax.bar(x - width/2, lgo_vals, width, label='LGO', color='steelblue')
    bars2 = ax.bar(x + width/2, ebm_vals, width, label='EBM', color='indianred')
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    return ax


def plot_brier_comparison(
    lgo_metrics: Dict[str, float],
    ebm_metrics: Dict[str, float],
    ax: Optional[plt.Axes] = None,
    title: str = "Brier Score (lower is better)"
) -> plt.Axes:
    """Plot Brier score comparison."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    methods = ['LGO', 'EBM']
    brier_scores = [lgo_metrics.get('Brier', 0), ebm_metrics.get('Brier', 0)]
    colors = ['steelblue', 'indianred']
    
    bars = ax.bar(methods, brier_scores, color=colors)
    
    ax.set_ylabel('Brier Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    return ax


def plot_single_seed_comparison(
    seed_data: Dict[str, Any],
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Generate 6-panel comparison figure for a single seed."""
    
    y_true = seed_data['y_test']
    lgo_probs = seed_data['lgo']['test_probabilities']
    ebm_probs = seed_data['ebm']['test_probabilities']
    lgo_metrics = seed_data['lgo']['test_metrics']
    ebm_metrics = seed_data['ebm']['test_metrics']
    seed = seed_data['seed']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # ROC curves
    plot_roc_curves(y_true, lgo_probs, ebm_probs, ax=axes[0, 0], title='ROC Curve')
    
    # PR curves
    plot_pr_curves(y_true, lgo_probs, ebm_probs, ax=axes[0, 1], title='Precision-Recall Curve')
    
    # Calibration
    plot_calibration_curves(y_true, lgo_probs, ebm_probs, ax=axes[0, 2], title='Calibration Curve')
    
    # Probability histograms
    plot_probability_histograms(y_true, lgo_probs, ebm_probs, ax=axes[1, 0], title='Probability Distribution')
    
    # Metrics comparison
    plot_metrics_comparison(lgo_metrics, ebm_metrics, ax=axes[1, 1], title='Performance Metrics')
    
    # Brier score
    plot_brier_comparison(lgo_metrics, ebm_metrics, ax=axes[1, 2], title='Brier Score')
    
    plt.suptitle(f'LGO vs EBM Comparison (Seed={seed})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_multiseed_summary(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Generate multi-seed summary figure."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = ['AUROC', 'AUPRC', 'Brier', 'F1', 'Accuracy', 'train_time']
    titles = ['AUROC', 'AUPRC', 'Brier Score', 'F1 Score', 'Accuracy', 'Training Time (s)']
    
    lgo_data = results_df[results_df['method'] == 'LGO']
    ebm_data = results_df[results_df['method'] == 'EBM']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        # Box plot
        data_to_plot = [lgo_data[metric].values, ebm_data[metric].values]
        bp = ax.boxplot(data_to_plot, labels=['LGO', 'EBM'], patch_artist=True)
        
        # Colors
        colors = ['steelblue', 'indianred']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add scatter points
        for i, (d, c) in enumerate(zip(data_to_plot, colors)):
            x = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.5, color=c, s=30, edgecolors='white')
        
        # Statistics annotation
        lgo_mean, lgo_std = lgo_data[metric].mean(), lgo_data[metric].std()
        ebm_mean, ebm_std = ebm_data[metric].mean(), ebm_data[metric].std()
        
        ax.set_title(f'{title}\nLGO: {lgo_mean:.3f}±{lgo_std:.3f} | EBM: {ebm_mean:.3f}±{ebm_std:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('LGO vs EBM Multi-Seed Summary', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_scaling_analysis(
    results_dirs: List[str],
    budgets: List[int],
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot scaling analysis across different computational budgets."""
    
    lgo_means, lgo_stds = [], []
    ebm_means, ebm_stds = [], []
    
    for results_dir in results_dirs:
        # Find CSV file
        csv_files = list(Path(results_dir).glob('multiseed_results_*.csv'))
        if not csv_files:
            print(f"[WARNING] No results found in {results_dir}")
            continue
        
        df = pd.read_csv(csv_files[0])
        
        lgo_auroc = df[df['method'] == 'LGO']['AUROC']
        ebm_auroc = df[df['method'] == 'EBM']['AUROC']
        
        lgo_means.append(lgo_auroc.mean())
        lgo_stds.append(lgo_auroc.std())
        ebm_means.append(ebm_auroc.mean())
        ebm_stds.append(ebm_auroc.std())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(budgets))
    width = 0.35
    
    # LGO bars with error
    ax.bar(x - width/2, lgo_means, width, yerr=lgo_stds, label='LGO', 
           color='steelblue', capsize=5, alpha=0.8)
    
    # EBM bars with error
    ax.bar(x + width/2, ebm_means, width, yerr=ebm_stds, label='EBM', 
           color='indianred', capsize=5, alpha=0.8)
    
    ax.set_ylabel('AUROC')
    ax.set_xlabel('Computational Budget')
    ax.set_title('LGO vs EBM: Scaling Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b//1000}k' for b in budgets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_ebm_shape_functions(
    ebm_model,
    feature_names: List[str] = None,
    n_features: int = 6,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot EBM shape functions for top features."""
    
    try:
        from interpret import show as interpret_show
    except ImportError:
        print("[WARNING] InterpretML not available for shape function plotting")
        return None
    
    # Get global explanation
    ebm_global = ebm_model.model.explain_global()
    
    # Get importances
    importances = ebm_model.get_feature_importances()
    top_features = list(importances.head(n_features).index)
    
    # Filter to main effects only
    main_effects = [f for f in top_features if " x " not in f][:n_features]
    
    n_plots = min(len(main_effects), n_features)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # 辅助函数：安全获取EBM属性（兼容新旧版本）
    def safe_get_attr(model, attr_with_underscore, attr_without_underscore, default=None):
        result = getattr(model, attr_with_underscore, None)
        if result is not None:
            return result
        result = getattr(model, attr_without_underscore, None)
        if result is None:
            return default if default is not None else []
        if callable(result):
            return result()
        return result
    
    for idx, feature_name in enumerate(main_effects):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Get feature index - 兼容新旧版本InterpretML
        try:
            term_names = safe_get_attr(ebm_model.model, 'term_names_', 'term_names', [])
            bins_all = safe_get_attr(ebm_model.model, 'bins_', 'bins', [])
            term_scores = safe_get_attr(ebm_model.model, 'term_scores_', 'term_scores', [])
            
            feature_idx = list(term_names).index(feature_name)
            bins = bins_all[feature_idx][0]
            scores = term_scores[feature_idx]
            
            # Plot
            if len(bins) > 0:
                ax.step(bins, scores[:-1] if len(scores) > len(bins) else scores, 
                       where='post', color='steelblue', lw=2)
                ax.fill_between(bins, scores[:-1] if len(scores) > len(bins) else scores, 
                               step='post', alpha=0.3, color='steelblue')
            
            ax.axhline(y=0, color='k', linestyle='--', lw=1, alpha=0.5)
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Score')
            ax.set_title(f'{feature_name}')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            ax.set_title(feature_name)
    
    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('EBM Shape Functions (Top Features)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='LGO vs EBM Visualization'
    )
    parser.add_argument('--detailed_pkl', type=str, default=None,
                        help='Path to detailed pickle file')
    parser.add_argument('--results_csv', type=str, default=None,
                        help='Path to results CSV file')
    parser.add_argument('--results_dirs', type=str, default=None,
                        help='Comma-separated results directories for scaling analysis')
    parser.add_argument('--budgets', type=str, default='30000,100000,200000,500000',
                        help='Comma-separated budgets for scaling analysis')
    parser.add_argument('--png_seed', type=int, default=None,
                        help='Seed to plot (single seed)')
    parser.add_argument('--plot_all', action='store_true',
                        help='Plot all seeds')
    parser.add_argument('--plot_summary', action='store_true',
                        help='Plot multi-seed summary')
    parser.add_argument('--plot_scaling', action='store_true',
                        help='Plot scaling analysis')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for plots')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load detailed data
    if args.detailed_pkl:
        pkl_path = Path(args.detailed_pkl)
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                detailed_data = pickle.load(f)
            print(f"[LOAD] Loaded {len(detailed_data)} seed(s) from {pkl_path}")
        else:
            # Try glob
            pkl_files = list(Path(args.output_dir).glob('multiseed_detailed_*.pkl'))
            if pkl_files:
                with open(pkl_files[0], 'rb') as f:
                    detailed_data = pickle.load(f)
                print(f"[LOAD] Loaded {len(detailed_data)} seed(s) from {pkl_files[0]}")
            else:
                detailed_data = None
                print("[WARNING] No detailed data found")
    else:
        detailed_data = None
    
    # Plot single seed
    if args.png_seed is not None and detailed_data:
        seed_data = None
        for d in detailed_data:
            if d['seed'] == args.png_seed:
                seed_data = d
                break
        
        if seed_data:
            output_path = Path(args.output_dir) / f'comparison_seed{args.png_seed}_{timestamp}.png'
            plot_single_seed_comparison(seed_data, output_path=str(output_path), show=args.show)
        else:
            print(f"[ERROR] Seed {args.png_seed} not found in data")
    
    # Plot all seeds
    if args.plot_all and detailed_data:
        for seed_data in detailed_data:
            seed = seed_data['seed']
            output_path = Path(args.output_dir) / f'comparison_seed{seed}_{timestamp}.png'
            plot_single_seed_comparison(seed_data, output_path=str(output_path), show=False)
    
    # Plot summary
    if args.plot_summary:
        if args.results_csv:
            csv_path = Path(args.results_csv)
        else:
            csv_files = list(Path(args.output_dir).glob('multiseed_results_*.csv'))
            csv_path = csv_files[0] if csv_files else None
        
        if csv_path and csv_path.exists():
            results_df = pd.read_csv(csv_path)
            output_path = Path(args.output_dir) / f'multiseed_summary_{timestamp}.png'
            plot_multiseed_summary(results_df, output_path=str(output_path), show=args.show)
        else:
            print("[ERROR] No results CSV found")
    
    # Plot scaling analysis
    if args.plot_scaling and args.results_dirs:
        dirs = [d.strip() for d in args.results_dirs.split(',')]
        budgets = [int(b.strip()) for b in args.budgets.split(',')]
        output_path = Path(args.output_dir) / f'scaling_analysis_{timestamp}.png'
        plot_scaling_analysis(dirs, budgets, output_path=str(output_path), show=args.show)
    
    print("\n[DONE] Visualization completed!")


if __name__ == '__main__':
    main()
