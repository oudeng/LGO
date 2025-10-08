#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_ablation_heatmap.py - Visualize ablation study results as heatmap

This script creates heatmap visualizations showing the performance impact
of different LGU components (base vs soft vs hard gating).
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CUSTOMIZABLE TITLES
# ========================================
FIG_SUPTITLE = "Ablation Study: Impact of LGU Components"

TITLES = {
    "ICU": "ICU Risk",
    "NHANES": "NHANES Metabolic",
    "CTG": "CTG Fetal",
    "Cleveland": "Cleveland Heart",
    "Hydraulic": "Hydraulic Fault"
}

NAME_MAP = { 
    "overall_NHANES_metabolic_score": "NHANES",
    "overall_ICU_composite_risk_score": "ICU",
    "overall_UCI_CTG_NSPbin": "CTG",
    "overall_UCI_Heart_Cleveland_num": "Cleveland",
    "overall_UCI_HydraulicSys_fault_score": "Hydraulic"
}


def read_ablation_data(root):
    """Read ablation table from analysis output."""
    p = Path(root) / "aggregated" / "ablation_table.csv"
    if p.exists():
        return pd.read_csv(p)
    
    # Fallback: compute from overall_metrics
    p_metrics = Path(root) / "aggregated" / "overall_metrics.csv"
    if p_metrics.exists():
        return compute_ablation_from_metrics(p_metrics)
    
    return pd.DataFrame()


def compute_ablation_from_metrics(metrics_path):
    """Compute ablation metrics from overall_metrics.csv."""
    df = pd.read_csv(metrics_path)
    
    # Filter for LGU method only
    if 'method' in df.columns:
        df = df[df['method'].str.lower() == 'lgu']
    
    if df.empty:
        return pd.DataFrame()
    
    # Determine primary metric
    if 'metric' in df.columns:
        # Long format
        available_metrics = df['metric'].unique()
        primary_metric = None
        for m in ['AUROC', 'R2', 'RMSE']:
            if m in available_metrics:
                primary_metric = m
                break
        
        if primary_metric:
            df = df[df['metric'] == primary_metric]
            value_col = 'value'
        else:
            return pd.DataFrame()
    else:
        # Wide format
        value_col = None
        for m in ['AUROC', 'R2', 'RMSE']:
            if m in df.columns:
                value_col = m
                primary_metric = m
                break
    
    if not value_col:
        return pd.DataFrame()
    
    # Calculate medians for each experiment
    result = {}
    if 'experiment' in df.columns:
        for exp in df['experiment'].unique():
            exp_data = df[df['experiment'] == exp][value_col]
            result[f'{exp}_median'] = float(exp_data.median()) if len(exp_data) > 0 else np.nan
    
    result['primary_metric'] = primary_metric
    
    return pd.DataFrame([result])


def create_performance_matrix(all_data):
    """Create performance matrix for heatmap."""
    if not all_data:
        return None
    
    # Combine all ablation tables
    rows = []
    for dataset, df in all_data:
        if df.empty:
            continue
        
        row = {'Dataset': dataset}
        
        # Get performance values for each experiment
        for exp in ['base', 'lgu_soft', 'lgu_hard']:
            col = f'{exp}_median'
            if col in df.columns:
                row[exp.replace('lgu_', '')] = df[col].iloc[0]
        
        if len(row) > 1:  # Has data beyond dataset name
            rows.append(row)
    
    if not rows:
        return None
    
    matrix_df = pd.DataFrame(rows)
    matrix_df = matrix_df.set_index('Dataset')
    
    # Rename columns for display
    rename_map = {
        'base': 'Base',
        'soft': 'Soft Gate', 
        'hard': 'Hard Gate'
    }
    matrix_df = matrix_df.rename(columns=rename_map)
    
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
        
        # Get base performance
        if 'base_median' not in df.columns:
            continue
        
        base_val = df['base_median'].iloc[0]
        if pd.isna(base_val) or base_val == 0:
            continue
        
        # Calculate % improvement for soft and hard
        for exp in ['lgu_soft', 'lgu_hard']:
            col = f'{exp}_median'
            if col in df.columns:
                exp_val = df[col].iloc[0]
                if pd.notna(exp_val):
                    # For metrics where higher is better (AUROC, R2)
                    if 'primary_metric' in df.columns:
                        metric = df['primary_metric'].iloc[0]
                        if metric in ['AUROC', 'R2']:
                            improvement = (exp_val - base_val) / base_val * 100
                        else:  # Lower is better (RMSE, MAE)
                            improvement = (base_val - exp_val) / base_val * 100
                    else:
                        improvement = (exp_val - base_val) / base_val * 100
                    
                    row[exp.replace('lgu_', '')] = improvement
        
        if len(row) > 1:
            rows.append(row)
    
    if not rows:
        return None
    
    matrix_df = pd.DataFrame(rows)
    matrix_df = matrix_df.set_index('Dataset')
    
    # Rename columns
    matrix_df = matrix_df.rename(columns={
        'soft': 'Soft vs Base (%)',
        'hard': 'Hard vs Base (%)'
    })
    
    return matrix_df


def main():
    ap = argparse.ArgumentParser(description="Visualize ablation study results")
    ap.add_argument("--roots", nargs="+", required=True, 
                    help="List of dataset output roots")
    ap.add_argument("--outdir", default="utility_plots/figs",
                    help="Output directory for figures")
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating ablation heatmap visualizations...")
    
    # Collect data from all datasets
    datasets_order = [
        ("overall_ICU_composite_risk_score", "ICU"),
        ("overall_NHANES_metabolic_score", "NHANES"),
        ("overall_UCI_CTG_NSPbin", "CTG"),
        ("overall_UCI_Heart_Cleveland_num", "Cleveland"),
        ("overall_UCI_HydraulicSys_fault_score", "Hydraulic")
    ]
    
    all_data = []
    for root_name, dset_key in datasets_order:
        root = [r for r in args.roots if Path(r).name == root_name]
        if root:
            df = read_ablation_data(root[0])
            all_data.append((dset_key, df))
    
    # Create figure with multiple heatmaps
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Performance heatmap
    ax1 = fig.add_subplot(gs[0, :])
    perf_matrix = create_performance_matrix(all_data)
    if perf_matrix is not None:
        # Normalize values for better color mapping
        vmin = perf_matrix.min().min()
        vmax = perf_matrix.max().max()
        
        sns.heatmap(perf_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax1, cbar_kws={'label': 'Performance'},
                   vmin=vmin, vmax=vmax)
        ax1.set_title("Absolute Performance by Experiment")
        ax1.set_ylabel("")
    else:
        ax1.set_axis_off()
        ax1.text(0.5, 0.5, "No performance data available", 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Improvement heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    imp_matrix = create_improvement_matrix(all_data)
    if imp_matrix is not None:
        # Use diverging colormap centered at 0
        vmax = max(abs(imp_matrix.min().min()), abs(imp_matrix.max().max()))
        
        sns.heatmap(imp_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=ax2, cbar_kws={'label': 'Improvement (%)'},
                   center=0, vmin=-vmax, vmax=vmax)
        ax2.set_title("Performance Change from Base (%)")
        ax2.set_ylabel("")
    else:
        ax2.set_axis_off()
    
    # Summary statistics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_axis_off()
    
    if imp_matrix is not None:
        # Calculate average improvements
        soft_avg = imp_matrix['Soft vs Base (%)'].mean() if 'Soft vs Base (%)' in imp_matrix.columns else 0
        hard_avg = imp_matrix['Hard vs Base (%)'].mean() if 'Hard vs Base (%)' in imp_matrix.columns else 0
        
        # Count wins (positive improvement)
        soft_wins = (imp_matrix['Soft vs Base (%)'] > 0).sum() if 'Soft vs Base (%)' in imp_matrix.columns else 0
        hard_wins = (imp_matrix['Hard vs Base (%)'] > 0).sum() if 'Hard vs Base (%)' in imp_matrix.columns else 0
        
        n_datasets = len(imp_matrix)
        
        summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
        summary_text += "Average Improvement:\n"
        summary_text += f"  Soft Gating: {soft_avg:+.1f}%\n"
        summary_text += f"  Hard Gating: {hard_avg:+.1f}%\n\n"
        summary_text += "Win Rate (vs Base):\n"
        summary_text += f"  Soft Gating: {soft_wins}/{n_datasets}\n"
        summary_text += f"  Hard Gating: {hard_wins}/{n_datasets}\n\n"
        
        # Best performer per dataset
        if not imp_matrix.empty:
            summary_text += "Best Performer:\n"
            for idx, row in imp_matrix.iterrows():
                vals = row.to_dict()
                if vals:
                    best = max(vals, key=vals.get)
                    summary_text += f"  {idx}: {best.split(' vs')[0]}\n"
        
        ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.98)
    
    output_path = Path(args.outdir) / "07_ablation_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()