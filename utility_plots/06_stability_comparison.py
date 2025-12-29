#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_stability_comparison.py - Visualize model stability across seeds

Creates visualizations showing the stability (IQR variance) of different
methods across multiple random seeds.

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- IQR comparison by method
- Stability heatmap across datasets
- Lower IQR = more stable results
- Publication-quality output (PNG + PDF)

Usage:
    python 06_stability_comparison.py --roots overall_* --outdir figs/stability

python utility_plots/06_stability_comparison.py \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/stability
    
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
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
FIG_SUPTITLE = "Model Stability Analysis (Lower IQR = More Stable)"

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

METHOD_DISPLAY = {
    "pysr": "PySR",
    "pstree": "PSTree", 
    "rils_rols": "RILS-ROLS",
    "operon": "Operon",
    "lgo_base": r"LGO$_\mathrm{base}$",
    "lgo_soft": r"LGO$_\mathrm{soft}$",
    "lgo_hard": r"LGO$_\mathrm{hard}$"
}

METHOD_ORDER = ["pysr", "pstree", "rils_rols", "operon", "lgo_base", "lgo_soft", "lgo_hard"]

METHOD_COLORS = {
    "pysr": "#4575b4",
    "pstree": "#7b3294",
    "rils_rols": "#74add1",
    "operon": "#1a9850",
    "lgo_base": "#fdae61",
    "lgo_soft": "#f46d43",
    "lgo_hard": "#d73027"
}

EXPERIMENT_MAPPING = {
    'base': 'base',
    'lgo': 'base',
    'soft': 'soft',
    'lgo_soft': 'soft',
    'hard': 'hard',
    'lgo_hard': 'hard',
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


def read_stability_data(root):
    """Read stability_summary.csv from analysis output."""
    p = Path(root) / "aggregated" / "stability_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(p)
    
    # Create combined method names
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


def create_stability_bars(ax, df, dataset_name, metric='R2_IQR'):
    """Create bar plot showing IQR (stability) for each method."""
    if df.empty:
        ax.text(0.5, 0.5, f"{TITLES.get(dataset_name, dataset_name)}\nNo Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Auto-detect metric
    available_metrics = [c for c in df.columns if c.endswith('_IQR')]
    if metric not in df.columns and available_metrics:
        metric = available_metrics[0]
    elif metric not in df.columns:
        ax.text(0.5, 0.5, f"{TITLES.get(dataset_name, dataset_name)}\nNo IQR Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Prepare data for plotting
    methods = []
    iqrs = []
    colors = []
    
    for method in METHOD_ORDER:
        method_data = df[df['method_combined'] == method]
        if not method_data.empty and pd.notna(method_data[metric].values[0]):
            methods.append(METHOD_DISPLAY.get(method, method))
            iqrs.append(method_data[metric].values[0])
            colors.append(METHOD_COLORS.get(method, 'gray'))
    
    if not methods:
        ax.text(0.5, 0.5, f"{TITLES.get(dataset_name, dataset_name)}\nNo Stability Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    x = np.arange(len(methods))
    bars = ax.bar(x, iqrs, color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
    
    # Add value labels
    for bar, iqr in zip(bars, iqrs):
        ax.annotate(f'{iqr:.3f}', xy=(bar.get_x() + bar.get_width()/2, iqr),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7)
    
    ax.set_title(TITLES.get(dataset_name, dataset_name), fontsize=11, fontweight='bold')
    ax.set_ylabel(f"IQR ({metric.replace('_IQR', '')})", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    
    # Add reference line at median
    if iqrs:
        ax.axhline(y=np.median(iqrs), color='red', linestyle='--', alpha=0.3, linewidth=1)


def create_stability_heatmap(ax, all_data):
    """Create heatmap showing stability across datasets."""
    if not all_data:
        ax.set_axis_off()
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Find primary metric
    metric_col = None
    for m in ['R2_IQR', 'AUROC_IQR', 'RMSE_IQR', 'MAE_IQR']:
        if m in combined.columns:
            metric_col = m
            break
    
    if not metric_col:
        ax.set_axis_off()
        return
    
    # Create pivot table
    pivot = combined.pivot_table(
        values=metric_col,
        index='Dataset',
        columns='method_combined',
        aggfunc='first'
    )
    
    # Reorder columns
    cols = [c for c in METHOD_ORDER if c in pivot.columns]
    if cols:
        pivot = pivot[cols]
        # Rename columns for display
        pivot.columns = [METHOD_DISPLAY.get(c, c) for c in pivot.columns]
    
    # Reorder rows to match dataset order
    dataset_order = [d[1] for d in DATASETS_ORDER]
    pivot = pivot.reindex([d for d in dataset_order if d in pivot.index])
    
    # Create heatmap (lower is better -> reversed colormap)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
               ax=ax, cbar_kws={'label': f'IQR ({metric_col.replace("_IQR", "")})'})
    
    ax.set_title("Stability Comparison Across Methods", fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")


def main():
    ap = argparse.ArgumentParser(description="Visualize model stability")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Dataset directories")
    ap.add_argument("--outdir", default="figs/stability",
                    help="Output directory")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating stability comparison visualizations...")
    
    # 2x3 layout for individual plots + heatmap
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.30,
                          top=0.92, bottom=0.08)
    
    # Layout: Row 0-1 for 2x3 bar plots, Row 2 for heatmap
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    all_data = []
    
    for (root_name, dset_key), (row, col) in zip(DATASETS_ORDER, positions):
        ax = fig.add_subplot(gs[row, col])
        root = [r for r in args.roots if Path(r).name == root_name]
        if root:
            df = read_stability_data(root[0])
            if not df.empty:
                df['Dataset'] = dset_key
                all_data.append(df)
            create_stability_bars(ax, df, dset_key)
        else:
            # No data for this dataset
            ax.text(0.5, 0.5, f"{TITLES.get(dset_key, dset_key)}\nNo Data",
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, color='gray')
            ax.set_frame_on(True)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Heatmap at bottom
    ax_heatmap = fig.add_subplot(gs[2, :])
    create_stability_heatmap(ax_heatmap, all_data)
    
    plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.97, fontweight='bold')
    
    # Add legend at bottom
    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_DISPLAY[m], alpha=0.8)
               for m in METHOD_ORDER if m in METHOD_COLORS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01),
              ncol=len(handles), fontsize=9, frameon=True)
    
    output_path = Path(args.outdir) / "06_stability_comparison.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = str(output_path).replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    main()