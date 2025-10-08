#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_runtime_efficiency.py - Visualize runtime efficiency across methods

This script creates visualizations comparing the computational efficiency
(runtime) of different methods, highlighting LGU's speed advantage.
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
FIG_SUPTITLE = "Computational Efficiency Comparison"

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


def read_runtime_data(root):
    """Read runtime profile from analysis output."""
    p = Path(root) / "aggregated" / "runtime_profile.csv"
    if p.exists():
        df = pd.read_csv(p)
        # Filter for fit phase
        if 'phase' in df.columns:
            df = df[df['phase'] == 'fit']
        return df
    
    # Try to extract from hyperparams (sometimes has runtime info)
    p_hyper = Path(root) / "aggregated" / "hyperparams.csv"
    if p_hyper.exists():
        df = pd.read_csv(p_hyper)
        if 'runtime_sec' in df.columns or 'duration_s' in df.columns:
            return df
    
    return pd.DataFrame()


def read_performance_data(root):
    """Read performance metrics for efficiency analysis."""
    p = Path(root) / "aggregated" / "overall_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(p)
    
    # Calculate median performance per method
    if 'metric' in df.columns:
        # Long format
        primary_metrics = ['AUROC', 'R2']
        for m in primary_metrics:
            metric_df = df[df['metric'] == m]
            if not metric_df.empty:
                return metric_df.groupby(['method', 'experiment'] if 'experiment' in metric_df.columns else ['method'])['value'].median().reset_index()
    else:
        # Wide format
        for m in ['AUROC', 'R2']:
            if m in df.columns:
                return df.groupby(['method', 'experiment'] if 'experiment' in df.columns else ['method'])[m].median().reset_index()
    
    return pd.DataFrame()


def plot_runtime_bars(ax, df, dataset_name):
    """Create bar plot for runtime comparison."""
    if df.empty:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No runtime data")
        return
    
    # Aggregate runtime by method/experiment
    plot_data = []
    
    if 'duration_s' in df.columns:
        time_col = 'duration_s'
    elif 'runtime_sec' in df.columns:
        time_col = 'runtime_sec'
    else:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No timing data")
        return
    
    # Group by method and experiment if available
    grp_cols = ['method']
    if 'experiment' in df.columns:
        grp_cols.append('experiment')
    
    for name, group in df.groupby(grp_cols):
        # Handle both single column and multi-column groupby
        if isinstance(name, tuple):
            if len(name) >= 2:
                method, exp = name[0], name[1]
            else:
                method = name[0]
                exp = 'base'
        else:
            method = name
            exp = 'base'
        
        # Ensure method is a string
        method = str(method) if method is not None else 'unknown'
        
        # Calculate median runtime
        runtime = group[time_col].median()
        
        if method.lower() == 'lgu':
            label = exp
        else:
            label = method.upper()
        
        plot_data.append({
            'Method': label,
            'Runtime (s)': runtime
        })
    
    if not plot_data:
        ax.set_axis_off()
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create bar plot with log scale if needed
    sns.barplot(data=plot_df, x='Method', y='Runtime (s)', ax=ax)
    
    if plot_df['Runtime (s)'].max() > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Runtime (s) [log scale]')
    else:
        ax.set_ylabel('Runtime (s)')
    
    ax.set_title(TITLES.get(dataset_name, dataset_name))
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=8)


def plot_efficiency_scatter(ax, runtime_df, perf_df, dataset_name):
    """Create scatter plot of performance vs runtime (Pareto efficiency)."""
    if runtime_df.empty or perf_df.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{dataset_name}: Insufficient data", 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Merge runtime and performance data
    merge_cols = ['method']
    if 'experiment' in runtime_df.columns and 'experiment' in perf_df.columns:
        merge_cols.append('experiment')
    
    # Aggregate runtime
    if 'duration_s' in runtime_df.columns:
        time_col = 'duration_s'
    elif 'runtime_sec' in runtime_df.columns:
        time_col = 'runtime_sec'
    else:
        ax.set_axis_off()
        return
    
    runtime_agg = runtime_df.groupby(merge_cols)[time_col].median().reset_index()
    runtime_agg = runtime_agg.rename(columns={time_col: 'runtime'})
    
    # Merge with performance
    if 'value' in perf_df.columns:
        perf_col = 'value'
    else:
        perf_col = perf_df.columns[-1]  # Last column is likely the metric
    
    merged = pd.merge(runtime_agg, perf_df, on=merge_cols, how='inner')
    
    if merged.empty:
        ax.set_axis_off()
        return
    
    # Create labels
    merged['Label'] = merged.apply(
        lambda x: x.get('experiment', x['method']) if x['method'].lower() == 'lgu' else x['method'].upper(),
        axis=1
    )
    
    # Create scatter plot
    sns.scatterplot(data=merged, x='runtime', y=perf_col, hue='Label', 
                    s=100, ax=ax, alpha=0.7)
    
    # Add labels for each point
    for _, row in merged.iterrows():
        ax.annotate(row['Label'], (row['runtime'], row[perf_col]),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Runtime (s)')
    ax.set_ylabel('Performance')
    ax.set_title(f"{TITLES.get(dataset_name, dataset_name)} - Efficiency Frontier")
    
    # Log scale for runtime if needed
    if merged['runtime'].max() > 100:
        ax.set_xscale('log')
        ax.set_xlabel('Runtime (s) [log scale]')
    
    # Remove legend (labels are on points)
    if ax.get_legend():
        ax.get_legend().remove()


def main():
    ap = argparse.ArgumentParser(description="Visualize runtime efficiency")
    ap.add_argument("--roots", nargs="+", required=True, 
                    help="List of dataset output roots")
    ap.add_argument("--outdir", default="utility_plots/figs",
                    help="Output directory for figures")
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating runtime efficiency visualizations...")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Dataset order
    datasets_order = [
        ("overall_ICU_composite_risk_score", "ICU"),
        ("overall_NHANES_metabolic_score", "NHANES"),
        ("overall_UCI_CTG_NSPbin", "CTG"),
        ("overall_UCI_Heart_Cleveland_num", "Cleveland"),
        ("overall_UCI_HydraulicSys_fault_score", "Hydraulic")
    ]
    
    # Runtime bar plots (top row)
    positions_bars = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    
    all_runtime_data = []
    all_perf_data = []
    
    for (root_name, dset_key), (row, col) in zip(datasets_order, positions_bars):
        ax = fig.add_subplot(gs[row, col])
        
        root = [r for r in args.roots if Path(r).name == root_name]
        if root:
            runtime_df = read_runtime_data(root[0])
            perf_df = read_performance_data(root[0])
            
            if not runtime_df.empty:
                runtime_df['Dataset'] = dset_key
                all_runtime_data.append(runtime_df)
            
            if not perf_df.empty:
                perf_df['Dataset'] = dset_key
                all_perf_data.append(perf_df)
            
            plot_runtime_bars(ax, runtime_df, dset_key)
    
    # Summary comparison (bottom row)
    ax_summary = fig.add_subplot(gs[2, :2])
    
    if all_runtime_data:
        combined_runtime = pd.concat(all_runtime_data, ignore_index=True)
        
        # Calculate average runtime by method
        if 'duration_s' in combined_runtime.columns:
            time_col = 'duration_s'
        elif 'runtime_sec' in combined_runtime.columns:
            time_col = 'runtime_sec'
        else:
            time_col = None
        
        if time_col:
            # Focus on LGU experiments
            lgu_runtime = combined_runtime[combined_runtime['method'].str.lower() == 'lgu']
            
            if not lgu_runtime.empty and 'experiment' in lgu_runtime.columns:
                summary = lgu_runtime.groupby(['Dataset', 'experiment'])[time_col].median().reset_index()
                summary_pivot = summary.pivot(index='Dataset', columns='experiment', values=time_col)
                
                summary_pivot.plot(kind='bar', ax=ax_summary, width=0.7, logy=True)
                ax_summary.set_title("LGU Runtime Comparison Across Datasets")
                ax_summary.set_ylabel("Runtime (s) [log scale]")
                ax_summary.set_xlabel("")
                ax_summary.legend(title="Experiment")
                ax_summary.tick_params(axis='x', rotation=0)
                ax_summary.grid(axis='y', alpha=0.3)
            else:
                # All methods comparison
                summary = combined_runtime.groupby(['Dataset', 'method'])[time_col].median().reset_index()
                summary_pivot = summary.pivot(index='Dataset', columns='method', values=time_col)
                
                summary_pivot.plot(kind='bar', ax=ax_summary, width=0.7, logy=True)
                ax_summary.set_title("Runtime Comparison Across Methods")
                ax_summary.set_ylabel("Runtime (s) [log scale]")
                ax_summary.tick_params(axis='x', rotation=0)
        else:
            ax_summary.set_axis_off()
    else:
        ax_summary.set_axis_off()
    
    # Speedup statistics
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.set_axis_off()
    
    if all_runtime_data:
        combined = pd.concat(all_runtime_data, ignore_index=True)
        
        time_col = 'duration_s' if 'duration_s' in combined.columns else 'runtime_sec'
        
        if time_col in combined.columns:
            stats_text = "Runtime Statistics\n" + "="*25 + "\n\n"
            
            # Calculate speedup of LGU vs others
            lgu_time = combined[combined['method'].str.lower() == 'lgu'][time_col].median()
            
            for method in ['pysr', 'operon', 'pstree', 'rils_rols']:
                method_df = combined[combined['method'].str.lower() == method]
                if not method_df.empty:
                    method_time = method_df[time_col].median()
                    if pd.notna(method_time) and pd.notna(lgu_time) and lgu_time > 0:
                        speedup = method_time / lgu_time
                        stats_text += f"{method.upper()}:\n"
                        stats_text += f"  {speedup:.1f}x slower\n"
            
            stats_text += "\n"
            
            # LGU experiment comparison
            lgu_df = combined[combined['method'].str.lower() == 'lgu']
            if 'experiment' in lgu_df.columns:
                stats_text += "LGU Variants:\n"
                for exp in ['base', 'lgu_soft', 'lgu_hard']:
                    exp_time = lgu_df[lgu_df['experiment'] == exp][time_col].median()
                    if pd.notna(exp_time):
                        stats_text += f"  {exp}: {exp_time:.1f}s\n"
            
            ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                         fontsize=10, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.98)
    
    output_path = Path(args.outdir) / "08_runtime_efficiency.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()