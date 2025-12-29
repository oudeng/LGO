#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_median_performance_violin.py - Enhanced violin plots with seaborn

Generates publication-quality violin plots showing performance distribution.
Supports both regression (R²) and classification (AUROC) metrics.

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Seaborn-based publication-quality visualization
- Individual subplot export to /plots subdirectory
- Mean ± std annotations
- Legend at bottom of figure

Usage:
    python 01_median_performance_violin.py --roots overall_* --outdir figs --show_box

# Violin plots (2x3布局)
python utility_plots/01_median_performance_violin.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/performance \
  --show_box 
    
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
FIG_SUPTITLE = "Performance Comparison Across Methods and Datasets"

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

# Color palette - professional scientific colors
METHOD_COLORS = {
    "pysr": "#4575b4",       # Blue
    "pstree": "#7b3294",     # Purple
    "rils_rols": "#74add1",  # Light blue
    "operon": "#1a9850",     # Green
    "lgo_base": "#fdae61",   # Orange
    "lgo_soft": "#f46d43",   # Red-orange
    "lgo_hard": "#d73027"    # Red
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


def read_metrics_data(root):
    """Read overall_metrics.csv and restructure for plotting."""
    p = Path(root) / "aggregated" / "overall_metrics.csv"
    if not p.exists():
        print(f"Warning: {p} does not exist")
        return pd.DataFrame()
    
    df = pd.read_csv(p)
    df["metric"] = df["metric"].str.upper()
    
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


def is_classification(df):
    """Check if the metrics are for classification tasks."""
    metrics = df["metric"].unique()
    return "AUROC" in metrics or "AUPRC" in metrics


def create_violin_plot(ax, df, dataset_name, show_box=True, show_stats=True):
    """Create enhanced seaborn violin plot with optional box overlay."""
    if df.empty:
        ax.set_visible(False)
        return None
    
    # Choose metric based on task type
    if is_classification(df):
        metric = "AUROC" if "AUROC" in df["metric"].values else "AUPRC"
        y_label = metric
        y_lim = (-0.05, 1.1)
    else:
        metric = "R2"
        if metric not in df["metric"].values:
            ax.set_visible(False)
            return None
        y_label = r"$R^2$"
        y_lim = None
    
    df_metric = df[df["metric"] == metric].copy()
    
    # Prepare data for seaborn
    plot_df = df_metric[df_metric['method_combined'].isin(METHOD_ORDER)].copy()
    plot_df['method_combined'] = pd.Categorical(
        plot_df['method_combined'], 
        categories=METHOD_ORDER, 
        ordered=True
    )
    plot_df = plot_df.sort_values('method_combined')
    
    # Create color palette in order
    palette = [METHOD_COLORS.get(m, 'gray') for m in METHOD_ORDER if m in plot_df['method_combined'].values]
    methods_present = [m for m in METHOD_ORDER if m in plot_df['method_combined'].values]
    palette_dict = dict(zip(methods_present, palette))
    
    # Seaborn violin plot
    sns.violinplot(
        data=plot_df,
        x='method_combined',
        y='value',
        hue='method_combined',
        palette=palette_dict,
        inner=None,
        linewidth=1,
        saturation=0.8,
        ax=ax,
        legend=False,
        cut=0
    )
    
    # Add box plot overlay
    if show_box:
        sns.boxplot(
            data=plot_df,
            x='method_combined',
            y='value',
            width=0.15,
            showcaps=True,
            boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.2},
            whiskerprops={'color': 'black', 'linewidth': 1},
            capprops={'color': 'black', 'linewidth': 1},
            medianprops={'color': 'black', 'linewidth': 1.5},
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 
                       'markeredgecolor': 'red', 'markersize': 4, 'alpha': 0.6},
            ax=ax,
            legend=False
        )
    
    # Add strip plot for individual points
    sns.stripplot(
        data=plot_df,
        x='method_combined',
        y='value',
        color='black',
        alpha=0.4,
        size=4,
        jitter=0.05,
        ax=ax,
        legend=False
    )
    
    # Add mean ± std annotations below x-axis
    if show_stats:
        stats_text = []
        for i, method in enumerate(methods_present):
            method_data = plot_df[plot_df['method_combined'] == method]['value']
            if len(method_data) > 0:
                mean_val = method_data.mean()
                std_val = method_data.std()
                stats_text.append(f"{mean_val:.3f}\n±{std_val:.3f}")
            else:
                stats_text.append("")
        
        # Add text annotations below each violin
        for i, txt in enumerate(stats_text):
            ax.annotate(txt, xy=(i, ax.get_ylim()[0]), xytext=(i, -0.18),
                       textcoords=('data', 'axes fraction'),
                       ha='center', va='top', fontsize=7,
                       color='#404040')
    
    # X-axis labels
    xlabels = [METHOD_DISPLAY.get(m, m) for m in methods_present]
    ax.set_xticks(range(len(methods_present)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=9)
    
    # Y limits
    if y_lim:
        ax.set_ylim(y_lim)
    else:
        all_vals = plot_df['value'].dropna()
        if len(all_vals) > 0:
            y_min = all_vals.min() - 0.15 * (all_vals.max() - all_vals.min())
            y_max = all_vals.max() + 0.1 * (all_vals.max() - all_vals.min())
            ax.set_ylim(y_min, y_max)
    
    # Labels and title
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.set_title(TITLES.get(dataset_name, dataset_name), fontsize=12, fontweight='bold', pad=10)
    
    # Grid styling
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    sns.despine(ax=ax, top=True, right=True)
    
    return plot_df


def save_individual_plot(df, dataset_name, outdir, show_box=True, dpi=300):
    """Save individual subplot as separate file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    create_violin_plot(ax, df, dataset_name, show_box=show_box, show_stats=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"violin_{dataset_name.lower().replace(' ', '_')}"
    fig.savefig(plots_dir / f"{filename}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plots_dir / f"{filename}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved individual: {plots_dir / filename}.png/pdf")
    
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate performance violin plots")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Dataset directories")
    ap.add_argument("--outdir", default="figs/performance",
                    help="Output directory")
    ap.add_argument("--show_box", action="store_true", default=True,
                    help="Show box plot overlay")
    ap.add_argument("--no_individual", action="store_true",
                    help="Skip individual plot export")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 60)
    print("Creating Performance Violin Plots (Seaborn Edition)")
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
                df = read_metrics_data(root)
                
                if not df.empty:
                    create_violin_plot(ax, df, dset_key, show_box=args.show_box, show_stats=True)
                    
                    # Save individual plot
                    if not args.no_individual:
                        save_individual_plot(df, dset_key, args.outdir, 
                                           show_box=args.show_box, dpi=args.dpi)
                else:
                    ax.set_visible(False)
                    ax.text(0.5, 0.5, f"{dset_key}\nNo Data", 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, color='gray')
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
    suffix = "_with_box" if args.show_box else ""
    output_path = Path(args.outdir) / f"01_violin_performance_2x3{suffix}.png"
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