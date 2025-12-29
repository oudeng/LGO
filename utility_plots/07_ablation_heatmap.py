#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_ablation_heatmap.py - Visualize ablation study results

Creates heatmap visualizations showing the performance impact
of different LGO components (base vs soft vs hard gating).

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Absolute performance heatmap
- Improvement from base heatmap
- Summary statistics
- Publication-quality output (PNG + PDF)

Usage:
    python 07_ablation_heatmap.py --roots overall_* --outdir figs/ablation

# 07 Ablation
python utility_plots/07_ablation_heatmap.py \
  --roots overall_ICU_* overall_eICU_* overall_NHANES_* \
          overall_UCI_CTG_* overall_UCI_Heart_* overall_UCI_Hydraulic_* \
  --outdir utility_plots/figs/ablation
      
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0

# ========================================
# CONFIGURATION
# ========================================
FIG_SUPTITLE = "Ablation Study: Impact of LGO Components"

TITLES = {
    "ICU": "MIMIC-IV ICU",
    "eICU": "eICU",
    "NHANES": "NHANES",
    "CTG": "UCI CTG",
    "Cleveland": "UCI Cleveland",
    "Hydraulic": "UCI Hydraulic"
}

NAME_MAP = { 
    "overall_NHANES_metabolic_score": "NHANES",
    "overall_ICU_composite_risk_score": "ICU",
    "overall_eICU_composite_risk_score": "eICU",
    "overall_UCI_CTG_NSPbin": "CTG",
    "overall_UCI_Heart_Cleveland_num": "Cleveland",
    "overall_UCI_HydraulicSys_fault_score": "Hydraulic"
}

EXPERIMENT_DISPLAY = {
    "base": "Base",
    "lgo_soft": "Soft Gate",
    "lgo_hard": "Hard Gate"
}

EXPERIMENT_COLORS = {
    "Base": "#fdae61",
    "Soft Gate": "#f46d43",
    "Hard Gate": "#d73027"
}

# Dataset order for 2x3 layout
DATASETS_ORDER = [
    ("overall_ICU_composite_risk_score", "ICU"),
    ("overall_eICU_composite_risk_score", "eICU"),
    ("overall_NHANES_metabolic_score", "NHANES"),
    ("overall_UCI_CTG_NSPbin", "CTG"),
    ("overall_UCI_Heart_Cleveland_num", "Cleveland"),
    ("overall_UCI_HydraulicSys_fault_score", "Hydraulic")
]


def read_ablation_data(root):
    """Read ablation_table.csv from analysis output."""
    p = Path(root) / "aggregated" / "ablation_table.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def create_performance_matrix(all_data):
    """Create performance matrix for heatmap."""
    if not all_data:
        return None
    
    rows = []
    for dataset, df in all_data:
        if df.empty:
            continue
        
        row = {'Dataset': dataset}
        
        # Get performance values
        for exp, col in [('Base', 'base_median'), 
                         ('Soft Gate', 'lgo_soft_median'),
                         ('Hard Gate', 'lgo_hard_median')]:
            if col in df.columns:
                row[exp] = df[col].iloc[0]
        
        if len(row) > 1:
            rows.append(row)
    
    if not rows:
        return None
    
    matrix_df = pd.DataFrame(rows)
    matrix_df = matrix_df.set_index('Dataset')
    
    # Reorder rows to match dataset order
    dataset_order = [d[1] for d in DATASETS_ORDER]
    matrix_df = matrix_df.reindex([d for d in dataset_order if d in matrix_df.index])
    
    return matrix_df


def create_improvement_matrix(all_data):
    """Create improvement matrix showing % change from base."""
    if not all_data:
        return None
    
    rows = []
    for dataset, df in all_data:
        if df.empty:
            continue
        
        row = {'Dataset': dataset}
        
        if 'base_median' not in df.columns:
            continue
        
        base_val = df['base_median'].iloc[0]
        if pd.isna(base_val) or base_val == 0:
            continue
        
        # Get primary metric direction (higher is better for R2/AUROC)
        metric = df.get('primary_metric', pd.Series(['R2'])).iloc[0] if 'primary_metric' in df.columns else 'R2'
        higher_is_better = metric in ['R2', 'AUROC']
        
        for exp, col in [('Soft vs Base (%)', 'lgo_soft_median'),
                         ('Hard vs Base (%)', 'lgo_hard_median')]:
            if col in df.columns:
                exp_val = df[col].iloc[0]
                if pd.notna(exp_val):
                    if higher_is_better:
                        improvement = (exp_val - base_val) / abs(base_val) * 100
                    else:
                        improvement = (base_val - exp_val) / abs(base_val) * 100
                    row[exp] = improvement
        
        if len(row) > 1:
            rows.append(row)
    
    if not rows:
        return None
    
    matrix_df = pd.DataFrame(rows).set_index('Dataset')
    
    # Reorder rows to match dataset order
    dataset_order = [d[1] for d in DATASETS_ORDER]
    matrix_df = matrix_df.reindex([d for d in dataset_order if d in matrix_df.index])
    
    return matrix_df


def create_ablation_bars(ax, df, dataset_name):
    """Create bar plot for ablation comparison."""
    if df.empty:
        ax.text(0.5, 0.5, f"{TITLES.get(dataset_name, dataset_name)}\nNo Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    experiments = []
    values = []
    colors = []
    
    for exp, col in [('Base', 'base_median'), 
                     ('Soft Gate', 'lgo_soft_median'),
                     ('Hard Gate', 'lgo_hard_median')]:
        if col in df.columns and pd.notna(df[col].iloc[0]):
            experiments.append(exp)
            values.append(df[col].iloc[0])
            colors.append(EXPERIMENT_COLORS.get(exp, 'gray'))
    
    if not experiments:
        ax.text(0.5, 0.5, f"{TITLES.get(dataset_name, dataset_name)}\nNo Ablation Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    x = np.arange(len(experiments))
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_title(TITLES.get(dataset_name, dataset_name), fontsize=11, fontweight='bold')
    ax.set_ylabel("Performance", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=0, ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)


def main():
    ap = argparse.ArgumentParser(description="Visualize ablation study results")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Dataset directories")
    ap.add_argument("--outdir", default="figs/ablation",
                    help="Output directory")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating ablation heatmap visualizations...")
    
    all_data = []
    for root_name, dset_key in DATASETS_ORDER:
        root = [r for r in args.roots if Path(r).name == root_name]
        if root:
            df = read_ablation_data(root[0])
            all_data.append((dset_key, df))
        else:
            all_data.append((dset_key, pd.DataFrame()))
    
    # Create figure with 2x3 bar plots + heatmaps
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.30,
                          top=0.92, bottom=0.08)
    
    # 2x3 bar plots for individual datasets
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for (dset_key, df), (row, col) in zip(all_data, positions):
        ax = fig.add_subplot(gs[row, col])
        create_ablation_bars(ax, df, dset_key)
    
    # Performance heatmap (bottom left)
    ax1 = fig.add_subplot(gs[2, :2])
    perf_matrix = create_performance_matrix(all_data)
    if perf_matrix is not None and not perf_matrix.empty:
        vmin = perf_matrix.min().min()
        vmax = perf_matrix.max().max()
        
        sns.heatmap(perf_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax1, cbar_kws={'label': 'Performance (R²/AUROC)'},
                   vmin=vmin, vmax=vmax)
        ax1.set_title("Absolute Performance by Experiment", fontsize=12, fontweight='bold')
        ax1.set_ylabel("")
    else:
        ax1.set_axis_off()
        ax1.text(0.5, 0.5, "No performance data available",
                ha='center', va='center', transform=ax1.transAxes, fontsize=11)
    
    # Summary statistics (bottom right)
    ax2 = fig.add_subplot(gs[2, 2])
    ax2.set_axis_off()
    
    imp_matrix = create_improvement_matrix(all_data)
    if imp_matrix is not None and not imp_matrix.empty:
        soft_col = 'Soft vs Base (%)' if 'Soft vs Base (%)' in imp_matrix.columns else None
        hard_col = 'Hard vs Base (%)' if 'Hard vs Base (%)' in imp_matrix.columns else None
        
        soft_avg = imp_matrix[soft_col].mean() if soft_col else 0
        hard_avg = imp_matrix[hard_col].mean() if hard_col else 0
        
        soft_wins = (imp_matrix[soft_col] > 0).sum() if soft_col else 0
        hard_wins = (imp_matrix[hard_col] > 0).sum() if hard_col else 0
        
        n_datasets = len(imp_matrix)
        
        summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
        summary_text += "Average Improvement:\n"
        summary_text += f"  Soft Gating: {soft_avg:+.1f}%\n"
        summary_text += f"  Hard Gating: {hard_avg:+.1f}%\n\n"
        summary_text += "Win Rate (vs Base):\n"
        summary_text += f"  Soft Gating: {soft_wins}/{n_datasets}\n"
        summary_text += f"  Hard Gating: {hard_wins}/{n_datasets}\n\n"
        
        summary_text += "Best Performer:\n"
        for idx, row in imp_matrix.iterrows():
            vals = row.to_dict()
            if vals:
                best = max(vals, key=vals.get)
                summary_text += f"  {idx}: {best.split(' vs')[0]}\n"
        
        ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.97, fontweight='bold')
    
    # Add legend at bottom
    handles = [mpatches.Patch(color=EXPERIMENT_COLORS[e], label=e, alpha=0.8)
               for e in ['Base', 'Soft Gate', 'Hard Gate']]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01),
              ncol=3, fontsize=10, frameon=True)
    
    output_path = Path(args.outdir) / "07_ablation_heatmap.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = str(output_path).replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    main()