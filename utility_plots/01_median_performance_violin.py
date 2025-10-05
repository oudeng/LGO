#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_median_performance_violin.py - Enhanced violin plots with box plot overlay

This script generates violin plots with optional box plot overlay
showing quartiles, median, and outliers for better statistical visualization.

Author: Ou Deng on Sep 18, 2025
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# ========================================
# CUSTOMIZABLE TITLES - MODIFY HERE
# ========================================
FIG_SUPTITLE = "Performance Comparison"  # Main figure title

TITLES = {
    "ICU": "ICU",
    "NHANES": "NHANES",
    "CTG": "CTG",
    "Cleveland": "Cleveland",
    "Hydraulic": "Hydraulic"
}

# Dataset name mapping
NAME_MAP = { 
    "overall_NHANES_metabolic_score": "NHANES",
    "overall_ICU_composite_risk_score": "ICU",
    "overall_UCI_CTG_NSPbin": "CTG",
    "overall_UCI_Heart_Cleveland_num": "Cleveland",
    "overall_UCI_HydraulicSys_fault_score": "Hydraulic"
}

# Method display names
METHOD_DISPLAY = {
    "pysr": "PySR",
    "pstree": "PSTree", 
    "rils_rols": "RILS-ROLS",
    "operon": "Operon",
    "lgo_base": r"LGO$_\mathrm{base}$",
    "lgo_soft": r"LGO$_\mathrm{soft}$",
    "lgo_hard": r"LGO$_\mathrm{hard}$"
}

# Method order for consistent display - all 7 methods
METHOD_ORDER = ["pysr", "pstree", "rils_rols", "operon", "lgo_base", "lgo_soft", "lgo_hard"]

# Color palette - using colorbrewer2 colors for better distinction
METHOD_COLORS = {
    "pysr": "#2166ac",      # blue
    "pstree": "#762a83",    # purple
    "rils_rols": "#92c5de", # light blue
    "operon": "#5aae61",    # green
    "lgo_base": "#f4a582",  # light orange
    "lgo_soft": "#d6604d",  # red-orange
    "lgo_hard": "#b2182b"   # dark red
}

# Mapping for various possible experiment names to our standard names
EXPERIMENT_MAPPING = {
    'base': 'base',
    'lgo': 'base',
    'soft': 'soft',
    'lgo_soft': 'soft',
    'hard': 'hard',
    'lgo_hard': 'hard',
    'lgosoft': 'soft',
    'lgohard': 'hard',
}


def read_all_data(root):
    """Read all data points (not just median) from aggregated results."""
    p = Path(root) / "aggregated" / "overall_metrics.csv"
    if not p.exists():
        print(f"Warning: {p} does not exist")
        return pd.DataFrame()
    
    df = pd.read_csv(p)
    df["metric"] = df["metric"].str.upper()
    
    # Restructure data: combine method and experiment for lgo
    if "experiment" in df.columns:
        # Normalize experiment names for lgo
        lgo_data = df[df['method'] == 'lgo'].copy()
        if not lgo_data.empty:
            # Map experiment names to standard names
            lgo_data['experiment_normalized'] = lgo_data['experiment'].map(
                lambda x: EXPERIMENT_MAPPING.get(str(x).lower().strip(), x)
            )
            
            # Create combined method names
            lgo_data['method_combined'] = lgo_data.apply(
                lambda row: f"lgo_{row['experiment_normalized']}", axis=1
            )
            
            # For non-lgo methods
            non_lgo_data = df[df['method'] != 'lgo'].copy()
            non_lgo_data['method_combined'] = non_lgo_data['method']
            
            # Combine
            df = pd.concat([lgo_data, non_lgo_data])
        else:
            # No lgo data, just use method as is
            df['method_combined'] = df['method']
    else:
        # No experiment column, just use method as is
        df['method_combined'] = df['method']
    
    return df


def is_classification(df):
    """Check if the metrics are for classification tasks."""
    metrics = df["metric"].unique()
    return "AUROC" in metrics or "AUPRC" in metrics


def violinplot_perf_enhanced(ax, df, dataset_name, custom_title=None, show_box=True):
    """Create enhanced performance violin plot with box plot overlay."""
    if df.empty:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No data available")
        return None
    
    # Choose metric
    if is_classification(df):
        metric = "AUROC" if "AUROC" in df["metric"].values else "AUPRC"
        y_label = metric
        y_lim = (0, 1.05)
    else:
        metric = "R2"
        if metric not in df["metric"].values:
            ax.set_axis_off()
            ax.set_title(f"{dataset_name}: R2 metric not found")
            return None
        y_label = "R$^2$"
        y_lim = None
    
    # Filter for the chosen metric
    df_metric = df[df["metric"] == metric].copy()
    
    # Prepare data for plotting
    plot_data = []
    positions = []
    colors = []
    labels = []
    
    for i, method in enumerate(METHOD_ORDER):
        method_data = df_metric[df_metric["method_combined"] == method]["value"].values
        if len(method_data) > 0:
            plot_data.append(method_data)
            positions.append(i)
            colors.append(METHOD_COLORS[method])
            labels.append(METHOD_DISPLAY[method])
        else:
            # Add empty data for missing methods
            plot_data.append([np.nan])
            positions.append(i)
            colors.append(METHOD_COLORS[method])
            labels.append(METHOD_DISPLAY[method])
    
    # Create violin plot
    parts = ax.violinplot(
        plot_data,
        positions=positions,
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Color the violins
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    
    # Add box plot overlay if requested
    if show_box:
        bp = ax.boxplot(
            plot_data,
            positions=positions,
            widths=0.15,
            notch=True,
            patch_artist=False,
            showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5),
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(color='black', linewidth=1),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1)
        )
    
    # Add individual points with jitter
    for i, (data, pos) in enumerate(zip(plot_data, positions)):
        if len(data) > 0 and not all(np.isnan(data)):
            # Add jitter
            jitter = np.random.normal(0, 0.02, len(data))
            x_jittered = np.ones(len(data)) * pos + jitter
            ax.scatter(x_jittered, data, color='black', alpha=0.3, s=20, zorder=5)
    
    # Set x-axis labels
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add mean ± std annotations below x-axis labels
    # Get current y-axis limits to position text properly
    if y_lim:
        ax.set_ylim(y_lim)
        y_min, y_max = y_lim
    else:
        # Calculate y limits based on data
        all_vals = []
        for data in plot_data:
            if len(data) > 0 and not all(np.isnan(data)):
                all_vals.extend(data)
        if all_vals:
            y_min = min(all_vals) - 0.1
            y_max = max(all_vals) + 0.1
            ax.set_ylim(y_min, y_max)
        else:
            y_min, y_max = ax.get_ylim()
    
    # Calculate position for stats text - below x-axis
    y_range = y_max - y_min
    # Position for mean (first line)
    y_text_mean = y_min - y_range * 0.15
    # Position for std (second line)
    y_text_std = y_min - y_range * 0.20
    
    for i, (data, pos, method) in enumerate(zip(plot_data, positions, METHOD_ORDER)):
        if len(data) > 0 and not all(np.isnan(data)):
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            
            # First line: mean value
            ax.text(pos, y_text_mean, f'{mean_val:.3f}', 
                   ha='center', va='top', fontsize=8, fontweight='bold')
            # Second line: ± std
            ax.text(pos, y_text_std, f'±{std_val:.3f}', 
                   ha='center', va='top', fontsize=8)
    
    # Adjust bottom margin to accommodate the stats text
    ax.set_ylim(y_min - y_range * 0.25, y_max)
    
    # Set y-axis label
    ax.set_ylabel(y_label)
    
    # Set title
    ax.set_title(custom_title if custom_title else dataset_name)
    ax.set_xlabel("")
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.2)
    ax.set_axisbelow(True)
    
    # Add subtle background
    ax.set_facecolor('#f8f8f8')
    
    # Draw a subtle line to separate stats from plot
    ax.axhline(y=y_min - y_range * 0.05, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    return df_metric


def main():
    ap = argparse.ArgumentParser(description="Generate enhanced performance violin plots")
    ap.add_argument("--roots", nargs="+", required=True, 
                    help="List of dataset output roots")
    ap.add_argument("--outdir", default="utility_plots/figs",
                    help="Output directory for figures")
    ap.add_argument("--show_box", action="store_true",
                    help="Show box plot overlay on violin plots")
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating enhanced performance violin plots...")
    print(f"Showing distribution across seeds for 7 methods")
    if args.show_box:
        print("Including box plot overlay for quartiles")
    
    fig = plt.figure(figsize=(17, 11))
    
    # Custom GridSpec - increased vertical space for stats text
    gs = gridspec.GridSpec(2, 100, figure=fig,
                          height_ratios=[1, 1],
                          hspace=0.45, wspace=0.15)
    
    # First row layout
    ax_icu = fig.add_subplot(gs[0, 0:37])
    ax_nhanes = fig.add_subplot(gs[0, 43:80])
    ax_legend = fig.add_subplot(gs[0, 82:100])
    
    # Second row layout
    ax_ctg = fig.add_subplot(gs[1, 0:28])
    ax_cleveland = fig.add_subplot(gs[1, 36:64])
    ax_hydraulic = fig.add_subplot(gs[1, 72:100])
    
    datasets_order = [
        ("overall_ICU_composite_risk_score", ax_icu, "ICU"),
        ("overall_NHANES_metabolic_score", ax_nhanes, "NHANES"),
        ("overall_UCI_CTG_NSPbin", ax_ctg, "CTG"),
        ("overall_UCI_Heart_Cleveland_num", ax_cleveland, "Cleveland"),
        ("overall_UCI_HydraulicSys_fault_score", ax_hydraulic, "Hydraulic")
    ]
    
    all_stats = []  # Collect statistics for summary
    
    for root_name, ax, dset_key in datasets_order:
        root = [r for r in args.roots if Path(r).name == root_name]
        if root:
            df = read_all_data(root[0])
            dname = NAME_MAP.get(root_name, root_name)
            # Use custom title from TITLES
            result = violinplot_perf_enhanced(ax, df, dname, 
                                             custom_title=TITLES.get(dset_key, dname),
                                             show_box=args.show_box)
    
    # Create legend
    ax_legend.axis('off')
    
    # Method legend
    legend_patches = []
    for method in METHOD_ORDER:
        display_name = METHOD_DISPLAY[method]
        color = METHOD_COLORS[method]
        patch = mpatches.Patch(color=color, label=display_name, alpha=0.6)
        legend_patches.append(patch)
    
    ax_legend.legend(handles=legend_patches, loc='upper left', 
                    title='Method', frameon=True, fontsize=10, title_fontsize=11,
                    bbox_to_anchor=(0, 1))
    
    # Add explanation text with proper spacing from legend
    explanation_text = (
        'Visualization Elements:\n\n'
        '• Violin: probability density\n'
        '• Width: frequency of values\n'
        '• Black dots: individual seeds\n'
    )
    
    if args.show_box:
        explanation_text += (
            '• Box: quartiles & median\n'
            '• Whiskers: 1.5×IQR\n'
            '• Red dots: outliers\n'
        )
    
    explanation_text += '\n• Numbers: mean (upper line)\n                    ±std (lower line)'
    
    # Position text lower with more spacing from legend (0.35 instead of 0.55)
    ax_legend.text(0.0, 0.35, explanation_text,
                  transform=ax_legend.transAxes,
                  fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.98, fontweight='bold')
    
    # Save figure
    suffix = "_with_box" if args.show_box else ""
    output_path = Path(args.outdir) / f"01_violin_perf{suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()