#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_gating_usage.py - Visualize gating mechanism usage with seaborn

Creates publication-quality visualizations showing LGO soft and hard gating usage patterns.
Demonstrates the parsimony of hard gating vs soft gating.

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Seaborn-based publication-quality visualization
- Individual subplot export to /plots subdirectory
- Legend at bottom of figure

Usage:
    python 03_gating_usage.py --roots overall_* --outdir figs/gating

# Gating usage plots (2x3布局)
python utility_plots/03_gating_usage.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/gating
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    'axes.labelsize': 14,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
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
FIG_SUPTITLE = "Gating Mechanism Usage Analysis"

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

# LGO experiment display names and colors (consistent with other scripts)
EXPERIMENT_DISPLAY = {
    "lgo_base": r"LGO$_{\mathrm{base}}$",
    "lgo_soft": r"LGO$_{\mathrm{soft}}$",
    "lgo_hard": r"LGO$_{\mathrm{hard}}$"
}

# Experiment name normalization (consistent with other scripts)
EXPERIMENT_MAPPING = {
    'base': 'base',
    'lgo': 'base',
    'soft': 'soft',
    'lgo_soft': 'soft',
    'hard': 'hard',
    'lgo_hard': 'hard',
}

EXPERIMENT_COLORS = {
    "lgo_base": "#fdae61",   # Orange
    "lgo_soft": "#f46d43",   # Red-orange
    "lgo_hard": "#d73027"    # Red
}

METRIC_COLORS = {
    "Gate Usage %": "#4575b4",
    "Median Gates": "#74add1"
}


def read_gating_data(root):
    """Read gating_usage.csv from analysis output."""
    p = Path(root) / "aggregated" / "gating_usage.csv"
    if not p.exists():
        # Try to get from complexity_stats.csv
        p_complex = Path(root) / "aggregated" / "complexity_stats.csv"
        if p_complex.exists():
            df = pd.read_csv(p_complex)
            if 'gates_median' in df.columns:
                # Normalize method names
                df = df.copy()
                df['method_combined'] = df.apply(get_method_combined, axis=1) 
                df = df[df['method_combined'].isin(['lgo_base', 'lgo_soft', 'lgo_hard'])]
                # Add prop_with_gates if not present
                if 'prop_with_gates' not in df.columns:
                    df['prop_with_gates'] = (df['gates_median'] > 0).astype(float)
                return df
        return pd.DataFrame()
    
    df = pd.read_csv(p)
    df['method_combined'] = df.apply(get_method_combined, axis=1)
    return df


def get_method_combined(row):
    """
    Combine method and experiment columns to get standardized method name.
    """
    method = str(row.get('method', '')).lower().strip()
    experiment = str(row.get('experiment', '')).lower().strip()
    
    if method in ['lgo_base', 'lgo_soft', 'lgo_hard']:
        return method
    
    if method.startswith('lgo_lgo_'):
        return method.replace('lgo_lgo_', 'lgo_')
    
    if method == 'lgo':
        exp_normalized = EXPERIMENT_MAPPING.get(experiment, experiment)
        if exp_normalized == 'base' or experiment in ['', 'nan']:
            return 'lgo_base'
        elif exp_normalized == 'soft':
            return 'lgo_soft'
        elif exp_normalized == 'hard':
            return 'lgo_hard'
        else:
            if 'soft' in experiment:
                return 'lgo_soft'
            elif 'hard' in experiment:
                return 'lgo_hard'
            else:
                return 'lgo_base'
    
    if method.startswith('lgo_'):
        return method
    
    return method


def create_gating_bars(ax, df, dataset_name):
    """Create grouped bar plot for gating usage with seaborn styling."""
    if df.empty:
        ax.text(0.5, 0.5, f"{dataset_name}\nNo Data", 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Filter for LGO variants
    df = df[df['method_combined'].isin(['lgo_base', 'lgo_soft', 'lgo_hard'])].copy()
    
    if df.empty:
        ax.text(0.5, 0.5, f"{dataset_name}\nNo LGO Data", 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Prepare data
    experiments = ['lgo_base', 'lgo_soft', 'lgo_hard']
    x = np.arange(len(experiments))
    width = 0.35
    
    gate_usage = []
    median_gates = []
    
    for exp in experiments:
        exp_data = df[df['method_combined'] == exp]
        if not exp_data.empty:
            usage = exp_data['prop_with_gates'].values[0] * 100 if 'prop_with_gates' in exp_data.columns else 0
            gates = exp_data['gates_median'].values[0] if 'gates_median' in exp_data.columns else 0
        else:
            usage = 0
            gates = 0
        gate_usage.append(usage)
        median_gates.append(gates)
    
    # Create bars
    bars1 = ax.bar(x - width/2, gate_usage, width, label='Gate Usage %',
                   color=METRIC_COLORS['Gate Usage %'], edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, median_gates, width, label='Median Gates',
                   color=METRIC_COLORS['Median Gates'], edgecolor='white', linewidth=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_title(TITLES.get(dataset_name, dataset_name), fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Percentage(%)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    #ax.set_xticklabels([EXPERIMENT_DISPLAY[e] for e in experiments], fontsize=9)
    ax.set_xticklabels([EXPERIMENT_DISPLAY[e] for e in experiments], fontsize=11, fontweight='medium')

    ax.tick_params(axis='x', rotation=0)
    
    # Grid styling
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    sns.despine(ax=ax, top=True, right=True)


def save_individual_plot(df, dataset_name, outdir, dpi=300):
    """Save individual subplot as separate file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    create_gating_bars(ax, df, dataset_name)
    
    # Add legend to individual plot
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"gating_{dataset_name.lower().replace(' ', '_')}"
    fig.savefig(plots_dir / f"{filename}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plots_dir / f"{filename}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved individual: {plots_dir / filename}.png/pdf")
    
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize gating usage")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Dataset directories")
    ap.add_argument("--outdir", default="figs/gating",
                    help="Output directory")
    ap.add_argument("--no_individual", action="store_true",
                    help="Skip individual plot export")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 60)
    print("Creating Gating Usage Plots (Seaborn Edition)")
    print("=" * 60)
    
    # ========================================
    # 2x3 Layout Figure
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    #fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.15)
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.90, bottom=0.10)
    
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
                df = read_gating_data(root)
                
                if not df.empty:
                    create_gating_bars(ax, df, dset_key)
                    
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
    legend_patches = [
        mpatches.Patch(color=METRIC_COLORS['Gate Usage %'], label='Gate Usage %'),
        mpatches.Patch(color=METRIC_COLORS['Median Gates'], label='Median Gates')
    ]
    
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fontsize=10,
        title='Metric',
        title_fontsize=11
    )
    
    # Main title (at top)
    fig.suptitle(FIG_SUPTITLE, fontsize=14, fontweight='bold', y=0.97)
    
    # ========================================
    # Save combined figure
    # ========================================
    output_path = Path(args.outdir) / "03_gating_usage_2x3.png"
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