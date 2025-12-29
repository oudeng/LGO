#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_04_gating_parsimony.py - Gating usage and parsimony analysis

Analyzes LGO's gate usage patterns across datasets, demonstrating:
- LGO_hard uses fewer gates than LGO_soft (parsimony)
- Gates are used selectively where thresholding is beneficial
- Comparison of complexity vs accuracy trade-offs

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Generates Figure 3 style visualizations
- LaTeX table output for publication
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Configure matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0

# Colors (consistent with other scripts)
COLOR_SOFT = '#f46d43'   # Red-orange for LGO_soft
COLOR_HARD = '#d73027'   # Red for LGO_hard
COLOR_BASE = '#fdae61'   # Orange for LGO_base

# Dataset display names (updated with eICU)
DATASET_DISPLAY = {
    'ICU_composite_risk_score': 'MIMIC-IV ICU',
    'eICU_composite_risk_score': 'eICU',
    'NHANES_metabolic_score': 'NHANES',
    'UCI_Heart_Cleveland_num': 'UCI Cleveland',
    'UCI_HydraulicSys_fault_score': 'UCI Hydraulic',
    'UCI_CTG_NSPbin': 'UCI CTG',
}

# Short names for plots
DATASET_SHORT = {
    'ICU_composite_risk_score': 'ICU',
    'eICU_composite_risk_score': 'eICU',
    'NHANES_metabolic_score': 'NHANES',
    'UCI_Heart_Cleveland_num': 'Cleveland',
    'UCI_HydraulicSys_fault_score': 'Hydraulic',
    'UCI_CTG_NSPbin': 'CTG',
}

DATASET_ORDER = ['ICU', 'eICU', 'NHANES', 'CTG', 'Cleveland', 'Hydraulic']


def load_gating_data(dataset_dirs):
    """Load gating_usage.csv from all dataset directories"""
    all_data = []
    
    for dir_path in dataset_dirs:
        gating_path = Path(dir_path) / 'aggregated' / 'gating_usage.csv'
        if gating_path.exists():
            df = pd.read_csv(gating_path)
            all_data.append(df)
            print(f"  Loaded: {gating_path}")
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def create_gate_usage_comparison(df, output_path, dpi=300):
    """Create bar chart comparing gate usage % across datasets"""
    
    # Prepare data
    plot_data = []
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        experiment = row['experiment']
        
        # Skip base (no gates by design)
        if experiment == 'base':
            continue
        
        # Get short name
        short_name = DATASET_SHORT.get(dataset, dataset.split('_')[0])
        
        plot_data.append({
            'Dataset': short_name,
            'Experiment': experiment,
            'Gate Usage %': row['prop_with_gates'] * 100,
            'Median Gates': row['gates_median'],
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    if plot_df.empty:
        print("[WARN] No gating data to plot")
        return None
    
    # Sort by dataset order
    dataset_order = [d for d in DATASET_ORDER if d in plot_df['Dataset'].values]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Gate Usage %
    x = np.arange(len(dataset_order))
    width = 0.35
    
    soft_usage = []
    hard_usage = []
    for ds in dataset_order:
        ds_data = plot_df[plot_df['Dataset'] == ds]
        soft_val = ds_data[ds_data['Experiment'] == 'lgo_soft']['Gate Usage %'].values
        hard_val = ds_data[ds_data['Experiment'] == 'lgo_hard']['Gate Usage %'].values
        soft_usage.append(soft_val[0] if len(soft_val) > 0 else 0)
        hard_usage.append(hard_val[0] if len(hard_val) > 0 else 0)
    
    bars1 = ax1.bar(x - width/2, soft_usage, width, label=r'LGO$_{\mathrm{soft}}$', 
                   color=COLOR_SOFT, edgecolor='white', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, hard_usage, width, label=r'LGO$_{\mathrm{hard}}$', 
                   color=COLOR_HARD, edgecolor='white', linewidth=0.8)
    
    ax1.set_ylabel('Models with Gates (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Gate Usage Rate', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_order, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    sns.despine(ax=ax1, top=True, right=True)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Median Gates
    soft_gates = []
    hard_gates = []
    for ds in dataset_order:
        ds_data = plot_df[plot_df['Dataset'] == ds]
        soft_val = ds_data[ds_data['Experiment'] == 'lgo_soft']['Median Gates'].values
        hard_val = ds_data[ds_data['Experiment'] == 'lgo_hard']['Median Gates'].values
        soft_gates.append(soft_val[0] if len(soft_val) > 0 else 0)
        hard_gates.append(hard_val[0] if len(hard_val) > 0 else 0)
    
    bars3 = ax2.bar(x - width/2, soft_gates, width, label=r'LGO$_{\mathrm{soft}}$', 
                   color=COLOR_SOFT, edgecolor='white', linewidth=0.8)
    bars4 = ax2.bar(x + width/2, hard_gates, width, label=r'LGO$_{\mathrm{hard}}$', 
                   color=COLOR_HARD, edgecolor='white', linewidth=0.8)
    
    ax2.set_ylabel('Median Gates per Model', fontsize=11, fontweight='bold')
    ax2.set_title('Gate Count (Parsimony)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_order, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2, top=True, right=True)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def create_parsimony_table(df, output_path):
    """Create LaTeX table of gate usage statistics"""
    
    # Prepare data
    rows = []
    
    datasets = df['dataset'].unique()
    for dataset in datasets:
        ds_data = df[df['dataset'] == dataset]
        display_name = DATASET_SHORT.get(dataset, dataset)
        
        row = {'Dataset': display_name}
        
        for exp in ['lgo_soft', 'lgo_hard']:
            exp_data = ds_data[ds_data['experiment'] == exp]
            if not exp_data.empty:
                row[f'{exp}_usage'] = exp_data['prop_with_gates'].values[0] * 100
                row[f'{exp}_median'] = exp_data['gates_median'].values[0]
            else:
                row[f'{exp}_usage'] = np.nan
                row[f'{exp}_median'] = np.nan
        
        rows.append(row)
    
    table_df = pd.DataFrame(rows)
    
    # Sort by dataset order
    table_df['sort_order'] = table_df['Dataset'].map(
        {d: i for i, d in enumerate(DATASET_ORDER)}
    )
    table_df = table_df.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Save as CSV
    csv_path = output_path.replace('.tex', '.csv')
    table_df.to_csv(csv_path, index=False)
    print(f"  [OK] CSV: {csv_path}")
    
    # Generate LaTeX
    latex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Gate usage by dataset: median number of gates among top-100 results.}',
        r'\label{tab:gate_usage}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Dataset & \multicolumn{2}{c}{LGO$_{\text{soft}}$} & \multicolumn{2}{c}{LGO$_{\text{hard}}$} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
        r' & Usage (\%) & Median & Usage (\%) & Median \\',
        r'\midrule',
    ]
    
    for _, row in table_df.iterrows():
        soft_usage = f"{row['lgo_soft_usage']:.0f}" if pd.notna(row['lgo_soft_usage']) else '-'
        soft_median = f"{row['lgo_soft_median']:.1f}" if pd.notna(row['lgo_soft_median']) else '-'
        hard_usage = f"{row['lgo_hard_usage']:.0f}" if pd.notna(row['lgo_hard_usage']) else '-'
        hard_median = f"{row['lgo_hard_median']:.1f}" if pd.notna(row['lgo_hard_median']) else '-'
        
        latex_lines.append(
            f"  {row['Dataset']} & {soft_usage} & {soft_median} & {hard_usage} & {hard_median} \\\\"
        )
    
    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze gating usage and parsimony'
    )
    parser.add_argument('--dataset_dirs', nargs='+', required=True,
                       help='Dataset directories containing aggregated/*.csv')
    parser.add_argument('--outdir', default='figs/gating',
                       help='Output directory')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Loading gating data...")
    df = load_gating_data(args.dataset_dirs)
    
    if df.empty:
        print("[ERROR] No gating data found")
        return 1
    
    print(f"\nLoaded {len(df)} rows")
    
    # Create comparison plot
    print("\nCreating gate usage comparison plot...")
    plot_path = os.path.join(args.outdir, 'gate_usage_comparison.png')
    result = create_gate_usage_comparison(df, plot_path, dpi=args.dpi)
    if result:
        print(f"  [OK] {result}")
    
    # PDF version
    pdf_path = plot_path.replace('.png', '.pdf')
    create_gate_usage_comparison(df, pdf_path, dpi=args.dpi)
    
    # Create LaTeX table
    print("\nCreating parsimony table...")
    table_path = os.path.join(args.outdir, 'gate_usage_table.tex')
    result = create_parsimony_table(df, table_path)
    if result:
        print(f"  [OK] {result}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("PARSIMONY SUMMARY")
    print("="*60)
    
    for dataset in df['dataset'].unique():
        ds_data = df[df['dataset'] == dataset]
        display_name = DATASET_SHORT.get(dataset, dataset)
        
        soft_data = ds_data[ds_data['experiment'] == 'lgo_soft']
        hard_data = ds_data[ds_data['experiment'] == 'lgo_hard']
        
        if not soft_data.empty and not hard_data.empty:
            soft_median = soft_data['gates_median'].values[0]
            hard_median = hard_data['gates_median'].values[0]
            reduction = (soft_median - hard_median) / soft_median * 100 if soft_median > 0 else 0
            
            print(f"\n{display_name}:")
            print(f"  LGO_soft median gates: {soft_median:.1f}")
            print(f"  LGO_hard median gates: {hard_median:.1f}")
            print(f"  Reduction: {reduction:.1f}%")
    
    print("\n[DONE] Gating analysis complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())