#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_pareto_front.py - Generate Pareto front scatter plots with seaborn

Generates publication-quality Pareto front scatter plots showing 
complexity vs CV loss trade-offs for different methods and datasets.

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Seaborn-based publication-quality visualization
- Individual subplot export to /plots subdirectory
- Legend at bottom of figure

Usage:
    python 02_pareto_front.py --roots overall_* --outdir figs

# Pareto plots (2x3布局)
python utility_plots/02_pareto_front.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/pareto
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ========================================
# SEABORN PUBLICATION STYLE CONFIGURATION
# ========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
sns.set_palette("colorblind")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ========================================
# CONFIGURATION
# ========================================
FIG_SUPTITLE = "Pareto Fronts: Complexity vs CV Loss"

# Dataset titles for display
TITLES = {
    "ICU": "MIMIC-IV ICU",
    "eICU": "eICU",
    "NHANES": "NHANES",
    "CTG": "UCI CTG",
    "Cleveland": "UCI Cleveland",
    "Hydraulic": "UCI Hydraulic"
}

# Dataset directory name mapping (updated with eICU)
NAME_MAP = { 
    "overall_NHANES_metabolic_score": "NHANES",
    "overall_ICU_composite_risk_score": "ICU",
    "overall_eICU_composite_risk_score": "eICU",
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
    "lgo_base": r"LGO$_{\mathrm{base}}$",
    "lgo_soft": r"LGO$_{\mathrm{soft}}$",
    "lgo_hard": r"LGO$_{\mathrm{hard}}$"
}

METHOD_ORDER = ["pysr", "pstree", "rils_rols", "operon", "lgo_base", "lgo_soft", "lgo_hard"]

# Color palette - professional scientific colors (same as violin plot)
METHOD_COLORS = {
    "pysr": "#4575b4",       # Blue
    "pstree": "#7b3294",     # Purple
    "rils_rols": "#74add1",  # Light blue
    "operon": "#1a9850",     # Green
    "lgo_base": "#fdae61",   # Orange
    "lgo_soft": "#f46d43",   # Red-orange
    "lgo_hard": "#d73027"    # Red
}

# Marker styles for each method
METHOD_MARKERS = {
    "pysr": "o",
    "pstree": "s",
    "rils_rols": "^",
    "operon": "D",
    "lgo_base": "v",
    "lgo_soft": "<",
    "lgo_hard": ">"
}

# Experiment name normalization
EXPERIMENT_MAPPING = {
    'base': 'base',
    'lgo': 'base',
    'soft': 'soft',
    'lgo_soft': 'soft',
    'hard': 'hard',
    'lgo_hard': 'hard',
}


def read_pareto_data(root):
    """Read pareto_front.csv and restructure for plotting."""
    p = Path(root) / "aggregated" / "pareto_front.csv"
    if not p.exists():
        print(f"Warning: {p} does not exist")
        return pd.DataFrame()
    
    df = pd.read_csv(p)
    
    # Handle loss column naming
    if 'loss' in df.columns and 'value' not in df.columns:
        df = df.rename(columns={'loss': 'value'})
    
    def get_method_combined(row):
        method = str(row['method']).lower()
        if method == 'lgo':
            exp = str(row.get('experiment', 'base')).lower()
            exp_normalized = EXPERIMENT_MAPPING.get(exp, exp)
            if exp_normalized == 'base':
                return 'lgo_base'
            elif exp_normalized == 'soft':
                return 'lgo_soft'
            elif exp_normalized == 'hard':
                return 'lgo_hard'
            else:
                return f'lgo_{exp_normalized}'
        else:
            return method
    
    df['method_combined'] = df.apply(get_method_combined, axis=1)
    return df


def create_pareto_scatter(ax, df, dataset_name):
    """Create enhanced Pareto front scatter plot with seaborn styling."""
    if df.empty:
        ax.text(0.5, 0.5, f"{dataset_name}\nNo Data", 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    
    # Filter to methods in our order
    plot_df = df[df['method_combined'].isin(METHOD_ORDER)].copy()
    
    if plot_df.empty:
        ax.text(0.5, 0.5, f"{dataset_name}\nNo Valid Data", 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    
    # Plot each method with its own color and marker
    for method in METHOD_ORDER:
        method_data = plot_df[plot_df['method_combined'] == method]
        if not method_data.empty:
            color = METHOD_COLORS.get(method, 'gray')
            marker = METHOD_MARKERS.get(method, 'o')
            label = METHOD_DISPLAY.get(method, method)
            
            ax.scatter(
                method_data['complexity'], 
                method_data['value'],
                color=color, 
                marker=marker, 
                label=label, 
                s=60, 
                alpha=0.8, 
                edgecolors='white', 
                linewidth=0.8,
                zorder=3
            )
    
    # Axis labels
    ax.set_xlabel("Complexity", fontsize=11, fontweight='bold')
    ax.set_ylabel("CV Loss", fontsize=11, fontweight='bold')
    ax.set_title(TITLES.get(dataset_name, dataset_name), fontsize=12, fontweight='bold', pad=10)
    
    # Set reasonable axis limits
    x_margin = 0.05 * (plot_df['complexity'].max() - plot_df['complexity'].min() + 1)
    y_margin = 0.05 * (plot_df['value'].max() - plot_df['value'].min() + 0.01)
    
    ax.set_xlim(
        max(0, plot_df['complexity'].min() - x_margin),
        plot_df['complexity'].max() + x_margin
    )
    ax.set_ylim(
        max(0, plot_df['value'].min() - y_margin),
        plot_df['value'].max() + y_margin
    )
    
    # Grid styling
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    sns.despine(ax=ax, top=True, right=True)
    
    return plot_df


def save_individual_plot(df, dataset_name, outdir, dpi=300):
    """Save individual subplot as separate file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    create_pareto_scatter(ax, df, dataset_name)
    
    # Add legend to individual plot
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"pareto_{dataset_name.lower().replace(' ', '_')}"
    fig.savefig(plots_dir / f"{filename}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plots_dir / f"{filename}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved individual: {plots_dir / filename}.png/pdf")
    
    plt.close(fig)


def create_legend_handles():
    """Create legend handles with both color and marker."""
    handles = []
    for method in METHOD_ORDER:
        handle = mlines.Line2D(
            [], [],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle='None',
            markersize=8,
            label=METHOD_DISPLAY[method],
            markeredgecolor='white',
            markeredgewidth=0.5
        )
        handles.append(handle)
    return handles


def main():
    ap = argparse.ArgumentParser(description="Generate Pareto front plots")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Dataset directories")
    ap.add_argument("--outdir", default="figs/pareto",
                    help="Output directory")
    ap.add_argument("--no_individual", action="store_true",
                    help="Skip individual plot export")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 60)
    print("Creating Pareto Front Plots (Seaborn Edition)")
    print("=" * 60)
    
    # ========================================
    # 2x3 Layout Figure
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.12)
    
    # Dataset order for 2x3 layout
    # Row 1: ICU (MIMIC-IV), eICU, NHANES
    # Row 2: CTG, Cleveland, Hydraulic
    datasets_layout = [
        [("overall_ICU_composite_risk_score", "ICU"),
         ("overall_eICU_composite_risk_score", "eICU"),
         ("overall_NHANES_metabolic_score", "NHANES")],
        [("overall_UCI_CTG_NSPbin", "CTG"),
         ("overall_UCI_Heart_Cleveland_num", "Cleveland"),
         ("overall_UCI_HydraulicSys_fault_score", "Hydraulic")]
    ]
    
    # Process each dataset
    for row_idx, row_datasets in enumerate(datasets_layout):
        for col_idx, (root_name, dset_key) in enumerate(row_datasets):
            ax = axes[row_idx, col_idx]
            
            # Find matching root directory
            root = None
            for r in args.roots:
                if Path(r).name == root_name:
                    root = r
                    break
            
            if root:
                print(f"\nProcessing: {dset_key} ({root_name})")
                df = read_pareto_data(root)
                
                if not df.empty:
                    create_pareto_scatter(ax, df, dset_key)
                    
                    # Save individual plot
                    if not args.no_individual:
                        save_individual_plot(df, dset_key, args.outdir, dpi=args.dpi)
                else:
                    ax.text(0.5, 0.5, f"{dset_key}\nNo Data", 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, color='gray')
                    ax.set_frame_on(True)
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                print(f"\nSkipping: {dset_key} (directory not found)")
                ax.text(0.5, 0.5, f"{dset_key}\nNot Found", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='gray')
                ax.set_frame_on(True)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # ========================================
    # Legend (at bottom of figure)
    # ========================================
    legend_handles = create_legend_handles()
    
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=7,
        frameon=True,
        fontsize=10,
        title='Method',
        title_fontsize=11
    )
    
    # Main title (at top)
    fig.suptitle(FIG_SUPTITLE, fontsize=14, fontweight='bold', y=0.97)
    
    # ========================================
    # Save combined figure
    # ========================================
    output_path = Path(args.outdir) / "02_pareto_front_2x3.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved combined figure: {output_path}")
    
    # PDF version
    pdf_path = str(output_path).replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF: {pdf_path}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()