#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_complexity_distribution.py - Visualize model complexity distributions with seaborn

Creates publication-quality bar plots showing complexity distribution across methods.
Demonstrates LGO's ability to find simpler or more complex solutions
depending on gating configuration.

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Seaborn-based publication-quality visualization
- Individual subplot export to /plots subdirectory
- Legend at bottom of figure

Usage:
    python 04_complexity_distribution.py --roots overall_* --outdir figs/complexity

# Complexity plots (2x3布局)
python utility_plots/04_complexity_distribution.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/complexity
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
FIG_SUPTITLE = "Model Complexity Distribution Analysis"

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

# Method display names (consistent with other scripts)
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

# Color palette - professional scientific colors (consistent with other scripts)
METHOD_COLORS = {
    "pysr": "#4575b4",       # Blue
    "pstree": "#7b3294",     # Purple
    "rils_rols": "#74add1",  # Light blue
    "operon": "#1a9850",     # Green
    "lgo_base": "#fdae61",   # Orange
    "lgo_soft": "#f46d43",   # Red-orange
    "lgo_hard": "#d73027"    # Red
}

EXPERIMENT_MAPPING = {
    'base': 'base',
    'lgo': 'base',
    'soft': 'soft',
    'lgo_soft': 'soft',
    'hard': 'hard',
    'lgo_hard': 'hard',
}


def read_complexity_data(root):
    """Read complexity_stats.csv from analysis output."""
    p = Path(root) / "aggregated" / "complexity_stats.csv"
    if not p.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(p)
    
    # Create combined method names
    def get_method_combined(row):
        method = str(row['method']).lower()
        # Handle lgo_base, lgo_lgo_hard etc.
        if method.startswith('lgo_lgo_'):
            return method.replace('lgo_lgo_', 'lgo_')
        elif method.startswith('lgo_') and method != 'lgo_base':
            return method
        elif method == 'lgo':
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
            # Remove _base suffix from other methods
            return method.replace('_base', '')
    
    df['method_combined'] = df.apply(get_method_combined, axis=1)
    return df


def create_complexity_bars(ax, df, dataset_name):
    """Create bar plot for complexity comparison with seaborn styling."""
    if df.empty:
        ax.text(0.5, 0.5, f"{dataset_name}\nNo Data", 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Get unique methods (drop duplicates)
    df_unique = df.drop_duplicates(subset=['method_combined'])
    
    # Prepare data
    methods = []
    medians = []
    q1s = []
    q3s = []
    colors = []
    
    for method in METHOD_ORDER:
        method_data = df_unique[df_unique['method_combined'] == method]
        if not method_data.empty:
            methods.append(METHOD_DISPLAY.get(method, method))
            medians.append(method_data['complexity_median'].values[0])
            q1s.append(method_data['complexity_q1'].values[0] if 'complexity_q1' in method_data.columns else 0)
            q3s.append(method_data['complexity_q3'].values[0] if 'complexity_q3' in method_data.columns else 0)
            colors.append(METHOD_COLORS.get(method, 'gray'))
    
    if not methods:
        ax.text(0.5, 0.5, f"{dataset_name}\nNo Complexity Data", 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    x = np.arange(len(methods))
    
    # Calculate error bars (IQR)
    errors_low = [max(0, m - q) for m, q in zip(medians, q1s)]
    errors_high = [max(0, q - m) for m, q in zip(medians, q3s)]
    
    # Create bar plot with error bars
    bars = ax.bar(x, medians, color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
    ax.errorbar(x, medians, yerr=[errors_low, errors_high], fmt='none', 
               color='black', capsize=3, capthick=1)
    
    # Add value labels
    for bar, median in zip(bars, medians):
        ax.annotate(f'{median:.0f}', xy=(bar.get_x() + bar.get_width()/2, median),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_title(TITLES.get(dataset_name, dataset_name), fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel("Complexity (median)", fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    
    # Grid styling
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    sns.despine(ax=ax, top=True, right=True)


def save_individual_plot(df, dataset_name, outdir, dpi=300):
    """Save individual subplot as separate file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    create_complexity_bars(ax, df, dataset_name)
    
    plt.tight_layout()
    
    # Save
    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"complexity_{dataset_name.lower().replace(' ', '_')}"
    fig.savefig(plots_dir / f"{filename}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plots_dir / f"{filename}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved individual: {plots_dir / filename}.png/pdf")
    
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize model complexity")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Dataset directories")
    ap.add_argument("--outdir", default="figs/complexity",
                    help="Output directory")
    ap.add_argument("--no_individual", action="store_true",
                    help="Skip individual plot export")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 60)
    print("Creating Complexity Distribution Plots (Seaborn Edition)")
    print("=" * 60)
    
    # ========================================
    # 2x3 Layout Figure
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(hspace=0.40, wspace=0.25, top=0.92, bottom=0.15)
    
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
                df = read_complexity_data(root)
                
                if not df.empty:
                    create_complexity_bars(ax, df, dset_key)
                    
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
    legend_patches = []
    for method in METHOD_ORDER:
        patch = mpatches.Patch(
            color=METHOD_COLORS[method], 
            label=METHOD_DISPLAY[method], 
            alpha=0.8
        )
        legend_patches.append(patch)
    
    fig.legend(
        handles=legend_patches,
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
    output_path = Path(args.outdir) / "04_complexity_distribution_2x3.png"
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