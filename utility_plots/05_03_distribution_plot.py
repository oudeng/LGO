#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_03_distribution_plot.py - Generate threshold distribution plots

Creates Figure 2B/D style distribution plots showing:
- Horizontal IQR bars for learned thresholds
- Median markers
- Guideline reference lines

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Clear IQR visualization with median markers
- Guideline anchors shown as vertical lines
- Y-axis aligned with heatmap ordering
- Publication-quality output
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Configure matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0

# Colors
COLOR_IQR = '#87CEEB'      # Light blue for IQR bars
COLOR_MEDIAN = 'black'      # Black for median markers
COLOR_GUIDELINE = '#d62728' # Red for guideline lines

# Dataset display names (updated with eICU)
DATASET_DISPLAY = {
    'ICU_composite_risk_score': 'MIMIC-IV ICU',
    'eICU_composite_risk_score': 'eICU',
    'NHANES_metabolic_score': 'NHANES',
    'UCI_Heart_Cleveland_num': 'UCI Cleveland',
    'UCI_HydraulicSys_fault_score': 'UCI Hydraulic',
    'UCI_CTG_NSPbin': 'UCI CTG',
}

# Dataset order for 2x3 layout
DATASET_ORDER = [
    'ICU_composite_risk_score',
    'eICU_composite_risk_score', 
    'NHANES_metabolic_score',
    'UCI_CTG_NSPbin',
    'UCI_Heart_Cleveland_num',
    'UCI_HydraulicSys_fault_score'
]


def create_distribution_plot(df, dataset_name, output_path, figsize=(8, 6), dpi=300):
    """Create threshold distribution plot for a single dataset"""
    
    # Filter to features with IQR data
    df = df[df['has_iqr'] == True].copy()
    
    # Also require guideline for meaningful comparison
    df = df[df['guideline'].notna()].copy()
    
    if df.empty:
        print(f"  [WARN] No features with IQR data for {dataset_name}")
        return None
    
    # Sort by error category (same order as heatmap)
    df = df.sort_values(['error_cat', 'feature']).reset_index(drop=True)
    
    # Reverse for matplotlib (y-axis goes bottom to top)
    df = df.iloc[::-1].reset_index(drop=True)
    
    n_features = len(df)
    
    # Create figure
    fig_height = max(4, n_features * 0.5 + 1)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    
    y_pos = np.arange(n_features)
    
    # Plot IQR bars and markers
    for i, row in df.iterrows():
        q1 = row['q1']
        q3 = row['q3']
        median = row['median']
        guideline = row['guideline']
        
        # IQR bar
        bar_width = q3 - q1
        ax.barh(i, bar_width, left=q1, height=0.5,
               color=COLOR_IQR, edgecolor='black', linewidth=1, alpha=0.7)
        
        # Median marker
        ax.plot(median, i, 'ko', markersize=7, zorder=5)
        
        # Guideline marker
        if np.isfinite(guideline):
            ax.plot(guideline, i, 'r|', markersize=12, markeredgewidth=2.5, zorder=4)
    
    # Y-axis labels
    ax.set_yticks(y_pos)
    labels = []
    for _, row in df.iterrows():
        label = row.get('display_name', row['feature'])
        unit = row.get('unit', '')
        if unit and pd.notna(unit) and unit != '':
            label = f"{label} ({unit})"
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=10, fontweight='medium')
    
    # X-axis
    ax.set_xlabel('Threshold Value', fontsize=11)
    
    # Title
    display_name = DATASET_DISPLAY.get(dataset_name, dataset_name)
    ax.set_title(f'{display_name} Threshold Distribution', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_IQR, alpha=0.7, label='IQR (Q1-Q3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MEDIAN,
               markersize=7, label='Median', linestyle=''),
        Line2D([0], [0], marker='|', color=COLOR_GUIDELINE,
               markersize=12, markeredgewidth=2.5, label='Guideline', linestyle=''),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
             frameon=True, fancybox=False, framealpha=0.95, edgecolor='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def create_combined_distribution_2x3(df, output_path, dpi=300):
    """Create 2x3 layout distribution plots for 6 datasets"""
    
    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.35, wspace=0.30, top=0.92, bottom=0.10)
    
    # Dataset layout
    datasets_layout = [
        ['ICU_composite_risk_score', 'eICU_composite_risk_score', 'NHANES_metabolic_score'],
        ['UCI_CTG_NSPbin', 'UCI_Heart_Cleveland_num', 'UCI_HydraulicSys_fault_score']
    ]
    
    for row_idx, row_datasets in enumerate(datasets_layout):
        for col_idx, ds_name in enumerate(row_datasets):
            ax = axes[row_idx, col_idx]
            
            ds_data = df[df['dataset'] == ds_name].copy()
            ds_data = ds_data[(ds_data['has_iqr'] == True) & (ds_data['guideline'].notna())]
            
            if ds_data.empty:
                ax.text(0.5, 0.5, f"{DATASET_DISPLAY.get(ds_name, ds_name)}\nNo Data",
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='gray')
                ax.set_frame_on(True)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Sort and reverse
            ds_data = ds_data.sort_values(['error_cat', 'feature']).reset_index(drop=True)
            ds_data = ds_data.iloc[::-1].reset_index(drop=True)
            
            n_features = len(ds_data)
            y_pos = np.arange(n_features)
            
            # Plot bars and markers
            for i, row in ds_data.iterrows():
                q1, q3, median = row['q1'], row['q3'], row['median']
                guideline = row['guideline']
                
                # IQR bar
                ax.barh(i, q3 - q1, left=q1, height=0.5,
                       color=COLOR_IQR, edgecolor='black', linewidth=1, alpha=0.7)
                
                # Median marker
                ax.plot(median, i, 'ko', markersize=6, zorder=5)
                
                # Guideline marker
                if np.isfinite(guideline):
                    ax.plot(guideline, i, 'r|', markersize=10, markeredgewidth=2, zorder=4)
            
            # Y-axis labels
            ax.set_yticks(y_pos)
            labels = []
            for _, row in ds_data.iterrows():
                label = row.get('display_name', row['feature'])
                unit = row.get('unit', '')
                if unit and pd.notna(unit) and unit != '':
                    label = f"{label} ({unit})"
                labels.append(label)
            ax.set_yticklabels(labels, fontsize=9)
            
            # Title and labels
            display_name = DATASET_DISPLAY.get(ds_name, ds_name)
            ax.set_title(f'{display_name}', fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel('Threshold Value', fontsize=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
    
    # Shared legend at bottom
    legend_elements = [
        mpatches.Patch(color=COLOR_IQR, alpha=0.7, label='IQR (Q1-Q3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MEDIAN,
               markersize=6, label='Median', linestyle=''),
        Line2D([0], [0], marker='|', color=COLOR_GUIDELINE,
               markersize=10, markeredgewidth=2, label='Guideline', linestyle=''),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    # Main title
    fig.suptitle('Threshold Distribution Analysis', fontsize=14, fontweight='bold', y=0.97)
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def create_combined_distribution(df, datasets, output_path, dpi=300):
    """Create stacked distribution plots for multiple datasets (legacy)"""
    
    # Filter datasets that have data
    valid_datasets = []
    for ds in datasets:
        ds_data = df[df['dataset'] == ds]
        ds_data = ds_data[(ds_data['has_iqr'] == True) & (ds_data['guideline'].notna())]
        if not ds_data.empty:
            valid_datasets.append((ds, ds_data))
    
    if not valid_datasets:
        print("[WARN] No valid datasets for combined distribution plot")
        return None
    
    n_datasets = len(valid_datasets)
    
    # Create figure with subplots
    fig_height = sum(max(3, len(d[1]) * 0.5) for d in valid_datasets) + n_datasets
    fig, axes = plt.subplots(n_datasets, 1, figsize=(10, fig_height))
    if n_datasets == 1:
        axes = [axes]
    
    for idx, (ds_name, ds_data) in enumerate(valid_datasets):
        ax = axes[idx]
        
        # Sort and reverse
        ds_data = ds_data.sort_values(['error_cat', 'feature']).reset_index(drop=True)
        ds_data = ds_data.iloc[::-1].reset_index(drop=True)
        
        n_features = len(ds_data)
        y_pos = np.arange(n_features)
        
        # Plot bars and markers
        for i, row in ds_data.iterrows():
            q1, q3, median = row['q1'], row['q3'], row['median']
            guideline = row['guideline']
            
            # IQR bar
            ax.barh(i, q3 - q1, left=q1, height=0.5,
                   color=COLOR_IQR, edgecolor='black', linewidth=1, alpha=0.7)
            
            # Median marker
            ax.plot(median, i, 'ko', markersize=6, zorder=5)
            
            # Guideline marker
            if np.isfinite(guideline):
                ax.plot(guideline, i, 'r|', markersize=10, markeredgewidth=2, zorder=4)
        
        # Y-axis labels
        ax.set_yticks(y_pos)
        labels = []
        for _, row in ds_data.iterrows():
            label = row.get('display_name', row['feature'])
            unit = row.get('unit', '')
            if unit and pd.notna(unit) and unit != '':
                label = f"{label} ({unit})"
            labels.append(label)
        ax.set_yticklabels(labels, fontsize=9)
        
        # Title and labels
        display_name = DATASET_DISPLAY.get(ds_name, ds_name)
        ax.set_title(f'{display_name}', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Threshold Value', fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Shared legend at bottom
    legend_elements = [
        mpatches.Patch(color=COLOR_IQR, alpha=0.7, label='IQR (Q1-Q3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MEDIAN,
               markersize=6, label='Median', linestyle=''),
        Line2D([0], [0], marker='|', color=COLOR_GUIDELINE,
               markersize=10, markeredgewidth=2, label='Guideline', linestyle=''),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate threshold distribution plots'
    )
    parser.add_argument('--csv', required=True,
                       help='Path to aggregated threshold summary CSV')
    parser.add_argument('--outdir', default='figs/distributions',
                       help='Output directory')
    parser.add_argument('--datasets', nargs='*',
                       help='Specific datasets to plot (default: all)')
    parser.add_argument('--combined', action='store_true',
                       help='Create combined multi-dataset figure')
    parser.add_argument('--layout_2x3', action='store_true', default=True,
                       help='Use 2x3 layout for combined figure (default)')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    
    # Read data
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Get datasets to process
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = df['dataset'].unique().tolist()
    
    print(f"\nProcessing {len(datasets)} datasets...")
    
    # Create individual plots
    for dataset in datasets:
        ds_data = df[df['dataset'] == dataset].copy()
        if ds_data.empty:
            print(f"  [SKIP] No data for {dataset}")
            continue
        
        output_path = os.path.join(args.outdir, f'distribution_{dataset}.png')
        result = create_distribution_plot(ds_data, dataset, output_path, dpi=args.dpi)
        if result:
            print(f"  [OK] {result}")
        
        # PDF version
        pdf_path = output_path.replace('.png', '.pdf')
        create_distribution_plot(ds_data, dataset, pdf_path, dpi=args.dpi)
    
    # Combined figure
    if args.combined:
        # 2x3 layout version
        if args.layout_2x3:
            combined_2x3_path = os.path.join(args.outdir, 'distribution_combined_2x3.png')
            result = create_combined_distribution_2x3(df, combined_2x3_path, dpi=args.dpi)
            if result:
                print(f"\n[OK] Combined 2x3 distribution: {result}")
            
            # PDF version
            pdf_path = combined_2x3_path.replace('.png', '.pdf')
            create_combined_distribution_2x3(df, pdf_path, dpi=args.dpi)
        
        # Legacy stacked layout
        combined_path = os.path.join(args.outdir, 'distribution_combined.png')
        result = create_combined_distribution(df, datasets, combined_path, dpi=args.dpi)
        if result:
            print(f"[OK] Combined distribution: {result}")
        
        # PDF version
        pdf_path = combined_path.replace('.png', '.pdf')
        create_combined_distribution(df, datasets, pdf_path, dpi=args.dpi)
    
    print("\n[DONE] Distribution plot generation complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())