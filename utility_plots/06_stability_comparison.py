#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_stability_comparison.py - Visualize model stability across seeds

This script creates visualizations showing the stability (variance) of different
methods across multiple random seeds, highlighting LGU's consistency.
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
FIG_SUPTITLE = "Model Stability Analysis (Lower IQR = More Stable)"

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


def read_stability_data(root):
    """Read stability summary from analysis output."""
    p = Path(root) / "aggregated" / "stability_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    
    # Fallback to overall metrics and compute stability
    p_metrics = Path(root) / "aggregated" / "overall_metrics.csv"
    if p_metrics.exists():
        df = pd.read_csv(p_metrics)
        return compute_stability_from_metrics(df)
    
    return pd.DataFrame()


def compute_stability_from_metrics(df):
    """Compute stability metrics from overall_metrics.csv."""
    # Group by method/experiment and compute variance metrics
    grp_cols = ['method']
    if 'experiment' in df.columns:
        grp_cols.append('experiment')
    
    stability = []
    for name, group in df.groupby(grp_cols):
        # Get primary metric
        if 'metric' in group.columns:
            # Long format
            for metric in group['metric'].unique():
                metric_data = group[group['metric'] == metric]['value']
                if len(metric_data) > 1:
                    row = {
                        'method': name if isinstance(name, str) else name[0],
                        f'{metric}_IQR': float(np.percentile(metric_data, 75) - np.percentile(metric_data, 25)),
                        f'{metric}_std': float(metric_data.std()),
                        f'{metric}_mean': float(metric_data.mean())
                    }
                    if len(grp_cols) > 1:
                        row['experiment'] = name[1] if len(name) > 1 else ''
                    stability.append(row)
        else:
            # Wide format - look for metric columns
            for col in ['AUROC', 'R2', 'RMSE', 'MAE', 'Brier']:
                if col in group.columns:
                    metric_data = group[col]
                    if len(metric_data) > 1:
                        row = {
                            'method': name if isinstance(name, str) else name[0],
                            f'{col}_IQR': float(np.percentile(metric_data, 75) - np.percentile(metric_data, 25)),
                            f'{col}_std': float(metric_data.std()),
                            f'{col}_mean': float(metric_data.mean())
                        }
                        if len(grp_cols) > 1:
                            row['experiment'] = name[1] if len(name) > 1 else ''
                        stability.append(row)
                        break  # Only use first metric found
    
    return pd.DataFrame(stability)


def plot_stability_bars(ax, df, dataset_name, metric='auto'):
    """Create bar plot showing IQR (stability) for each method."""
    if df.empty:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No data")
        return
    
    # Auto-detect metric
    if metric == 'auto':
        for m in ['AUROC_IQR', 'R2_IQR', 'RMSE_IQR', 'MAE_IQR', 'Brier_IQR']:
            if m in df.columns:
                metric = m
                break
    
    if metric not in df.columns:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No stability metrics")
        return
    
    # Prepare data for plotting
    plot_data = []
    
    # Separate LGU experiments from other methods
    if 'method' in df.columns:
        for _, row in df.iterrows():
            method = row['method']
            if method.lower() == 'lgu' and 'experiment' in row:
                label = row['experiment']
            else:
                label = method.upper()
            
            if pd.notna(row[metric]):
                plot_data.append({
                    'Method': label,
                    'IQR': row[metric]
                })
    
    if not plot_data:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No IQR data")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define order
    order = ['base', 'lgu_soft', 'lgu_hard', 'PYSR', 'OPERON', 'PSTREE', 'RILS_ROLS']
    existing_order = [m for m in order if m in plot_df['Method'].unique()]
    
    # Create bar plot
    sns.barplot(data=plot_df, x='Method', y='IQR', ax=ax, order=existing_order)
    
    ax.set_title(TITLES.get(dataset_name, dataset_name))
    ax.set_xlabel("")
    ax.set_ylabel(f"IQR of {metric.replace('_IQR', '')}")
    ax.tick_params(axis='x', rotation=45)
    
    # Add horizontal line at median IQR
    ax.axhline(y=plot_df['IQR'].median(), color='red', 
               linestyle='--', alpha=0.3, label='Median')


def plot_stability_heatmap(ax, all_data):
    """Create heatmap showing relative stability across datasets."""
    if not all_data:
        ax.set_axis_off()
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Find the primary metric
    metric_col = None
    for m in ['AUROC_IQR', 'R2_IQR', 'RMSE_IQR']:
        if m in combined.columns:
            metric_col = m
            break
    
    if not metric_col:
        ax.set_axis_off()
        return
    
    # Create pivot table
    if 'experiment' in combined.columns:
        # Focus on LGU experiments
        lgu_df = combined[combined['method'].str.lower() == 'lgu']
        if not lgu_df.empty:
            pivot = lgu_df.pivot_table(
                values=metric_col,
                index='Dataset',
                columns='experiment',
                aggfunc='first'
            )
            
            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                       ax=ax, cbar_kws={'label': f'IQR'})
            ax.set_title("Stability Comparison (LGU Experiments)")
            ax.set_xlabel("Experiment")
            ax.set_ylabel("Dataset")
    else:
        # All methods
        pivot = combined.pivot_table(
            values=metric_col,
            index='Dataset',
            columns='method',
            aggfunc='first'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   ax=ax, cbar_kws={'label': f'IQR'})
        ax.set_title("Stability Comparison (All Methods)")


def main():
    ap = argparse.ArgumentParser(description="Visualize model stability analysis")
    ap.add_argument("--roots", nargs="+", required=True, 
                    help="List of dataset output roots")
    ap.add_argument("--outdir", default="utility_plots/figs",
                    help="Output directory for figures")
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating stability comparison visualizations...")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Dataset order
    datasets_order = [
        ("overall_ICU_composite_risk_score", "ICU"),
        ("overall_NHANES_metabolic_score", "NHANES"),
        ("overall_UCI_CTG_NSPbin", "CTG"),
        ("overall_UCI_Heart_Cleveland_num", "Cleveland"),
        ("overall_UCI_HydraulicSys_fault_score", "Hydraulic")
    ]
    
    # Individual dataset plots
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    all_data = []
    
    for (root_name, dset_key), (row, col) in zip(datasets_order, positions):
        ax = fig.add_subplot(gs[row, col])
        
        root = [r for r in args.roots if Path(r).name == root_name]
        if root:
            df = read_stability_data(root[0])
            if not df.empty:
                df['Dataset'] = dset_key
                all_data.append(df)
            plot_stability_bars(ax, df, dset_key)
    
    # Heatmap comparison
    ax_heatmap = fig.add_subplot(gs[2, :])
    plot_stability_heatmap(ax_heatmap, all_data)
    
    plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.98)
    
    output_path = Path(args.outdir) / "06_stability_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()