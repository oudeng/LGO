#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_runtime_efficiency.py - Visualize runtime efficiency

Creates visualizations comparing computational efficiency (runtime)
of different methods across datasets.

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Runtime comparison by method
- Cross-dataset summary
- Speedup statistics
- Publication-quality output (PNG + PDF)

Usage:
    python 08_runtime_efficiency.py --roots overall_* --outdir figs/runtime

python utility_plots/08_runtime_efficiency.py \
  --roots overall_* \
  --outdir utility_plots/figs/runtime

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
FIG_SUPTITLE = "Computational Efficiency Comparison"

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
    "lgo": "LGO",
    "lgo_base": r"LGO$_\mathrm{base}$",
    "lgo_soft": r"LGO$_\mathrm{soft}$",
    "lgo_hard": r"LGO$_\mathrm{hard}$"
}

METHOD_ORDER = ["pysr", "pstree", "rils_rols", "operon", "lgo"]

METHOD_COLORS = {
    "pysr": "#4575b4",
    "pstree": "#7b3294",
    "rils_rols": "#74add1",
    "operon": "#1a9850",
    "lgo": "#d73027",
    "lgo_base": "#fdae61",
    "lgo_soft": "#f46d43",
    "lgo_hard": "#d73027"
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


def read_runtime_data(root):
    """Read runtime_profile.csv from analysis output."""
    p = Path(root) / "aggregated" / "runtime_profile.csv"
    if not p.exists():
        return pd.DataFrame()
    
    try:
        # Try standard read first
        df = pd.read_csv(p)
    except pd.errors.ParserError:
        try:
            # Try with error handling - skip bad lines
            df = pd.read_csv(p, on_bad_lines='skip')
            print(f"  [WARN] Skipped malformed lines in {p}")
        except Exception as e:
            print(f"  [ERROR] Could not parse {p}: {e}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  [ERROR] Could not read {p}: {e}")
        return pd.DataFrame()
    
    # Filter for fit phase
    if 'phase' in df.columns:
        df = df[df['phase'] == 'fit']
    
    return df


def create_runtime_bars(ax, df, dataset_name):
    """Create bar plot for runtime comparison."""
    if df.empty:
        ax.text(0.5, 0.5, f"{TITLES.get(dataset_name, dataset_name)}\nNo Runtime Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Determine time column
    time_col = 'duration_s' if 'duration_s' in df.columns else 'runtime_sec'
    if time_col not in df.columns:
        ax.text(0.5, 0.5, f"{TITLES.get(dataset_name, dataset_name)}\nNo Timing Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Calculate median runtime by method
    runtime_by_method = df.groupby('method')[time_col].median().reset_index()
    
    # Prepare data
    methods = []
    runtimes = []
    colors = []
    
    for method in METHOD_ORDER:
        method_data = runtime_by_method[runtime_by_method['method'].str.lower() == method.lower()]
        if not method_data.empty:
            methods.append(METHOD_DISPLAY.get(method, method))
            runtimes.append(method_data[time_col].values[0])
            colors.append(METHOD_COLORS.get(method, 'gray'))
    
    if not methods:
        ax.text(0.5, 0.5, f"{TITLES.get(dataset_name, dataset_name)}\nNo Method Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    x = np.arange(len(methods))
    bars = ax.bar(x, runtimes, color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
    
    # Log scale if needed
    if max(runtimes) > 100:
        ax.set_yscale('log')
        ylabel = 'Runtime (s) [log]'
    else:
        ylabel = 'Runtime (s)'
    
    # Add value labels
    for bar, runtime in zip(bars, runtimes):
        label = f'{runtime:.0f}s' if runtime >= 1 else f'{runtime:.2f}s'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7)
    
    ax.set_title(TITLES.get(dataset_name, dataset_name), fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)


def create_runtime_summary(ax, all_data):
    """Create summary bar plot across datasets."""
    if not all_data:
        ax.set_axis_off()
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    time_col = 'duration_s' if 'duration_s' in combined.columns else 'runtime_sec'
    if time_col not in combined.columns:
        ax.set_axis_off()
        return
    
    # Calculate median runtime by dataset and method
    summary = combined.groupby(['Dataset', 'method'])[time_col].median().reset_index()
    
    # Pivot for plotting
    pivot = summary.pivot(index='Dataset', columns='method', values=time_col)
    
    # Reorder columns
    cols = [c for c in METHOD_ORDER if c in pivot.columns]
    if cols:
        pivot = pivot[cols]
        pivot.columns = [METHOD_DISPLAY.get(c, c) for c in pivot.columns]
    
    # Reorder rows to match dataset order
    dataset_order = [d[1] for d in DATASETS_ORDER]
    pivot = pivot.reindex([d for d in dataset_order if d in pivot.index])
    
    # Plot
    colors_list = [METHOD_COLORS.get(c.lower(), 'gray') for c in cols]
    pivot.plot(kind='bar', ax=ax, width=0.7, logy=True, color=colors_list)
    
    ax.set_title("Runtime Comparison Across Datasets", fontsize=11, fontweight='bold')
    ax.set_ylabel("Runtime (s) [log scale]", fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    sns.despine(ax=ax, top=True, right=True)


def main():
    ap = argparse.ArgumentParser(description="Visualize runtime efficiency")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Dataset directories")
    ap.add_argument("--outdir", default="figs/runtime",
                    help="Output directory")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating runtime efficiency visualizations...")
    
    # 2x3 layout for individual plots + summary
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.30,
                          top=0.92, bottom=0.08)
    
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    all_data = []
    
    for (root_name, dset_key), (row, col) in zip(DATASETS_ORDER, positions):
        ax = fig.add_subplot(gs[row, col])
        root = [r for r in args.roots if Path(r).name == root_name]
        if root:
            df = read_runtime_data(root[0])
            if not df.empty:
                df['Dataset'] = dset_key
                all_data.append(df)
            create_runtime_bars(ax, df, dset_key)
        else:
            # No data for this dataset
            ax.text(0.5, 0.5, f"{TITLES.get(dset_key, dset_key)}\nNo Data",
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, color='gray')
            ax.set_frame_on(True)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Summary plot (bottom left 2 columns)
    ax_summary = fig.add_subplot(gs[2, :2])
    create_runtime_summary(ax_summary, all_data)
    
    # Statistics (bottom right)
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis('off')
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        time_col = 'duration_s' if 'duration_s' in combined.columns else 'runtime_sec'
        
        if time_col in combined.columns:
            stats_text = "Runtime Statistics\n" + "="*25 + "\n\n"
            
            # LGO median
            lgo_df = combined[combined['method'].str.lower() == 'lgo']
            if not lgo_df.empty:
                lgo_median = lgo_df[time_col].median()
                stats_text += f"LGO median: {lgo_median:.1f}s\n\n"
                
                # Comparison with other methods
                stats_text += "Relative to LGO:\n"
                for method in ['pysr', 'operon', 'pstree', 'rils_rols']:
                    method_df = combined[combined['method'].str.lower() == method]
                    if not method_df.empty:
                        method_median = method_df[time_col].median()
                        if lgo_median > 0:
                            ratio = method_median / lgo_median
                            stats_text += f"  {METHOD_DISPLAY.get(method, method)}: {ratio:.1f}x\n"
                
                # Fastest/slowest dataset for LGO
                stats_text += "\nLGO by Dataset:\n"
                lgo_by_ds = lgo_df.groupby('Dataset')[time_col].median().sort_values()
                for ds, t in lgo_by_ds.items():
                    stats_text += f"  {ds}: {t:.1f}s\n"
            
            ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                         fontsize=9, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.97, fontweight='bold')
    
    # Add legend at bottom
    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_DISPLAY[m], alpha=0.8)
               for m in METHOD_ORDER if m in METHOD_COLORS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01),
              ncol=len(handles), fontsize=9, frameon=True)
    
    output_path = Path(args.outdir) / "08_runtime_efficiency.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = str(output_path).replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    main()