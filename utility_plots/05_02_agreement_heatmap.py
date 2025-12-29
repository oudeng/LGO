#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_02_agreement_heatmap.py - Generate publication-ready agreement heatmaps

Creates Figure 2A/C style heatmaps showing LGO threshold alignment with guidelines.
Each cell shows: threshold value and relative error (Δ%).

Features:
- 2x3 layout for 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- Traffic light color scheme (green ≤10%, yellow ≤20%, red >20%)
- LaTeX-style delta notation
- Unit-aware annotations
- Publication-quality output (300+ DPI)
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
from matplotlib.colors import ListedColormap, BoundaryNorm

# Configure matplotlib for publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 1.0

# Color scheme
COLOR_GREEN = "#2ca02c"   # ≤10%
COLOR_YELLOW = "#ffbf00"  # ≤20%
COLOR_RED = "#d62728"     # >20%
COLOR_NA = "#d9d9d9"      # N/A

# Dataset display names (updated with eICU)
DATASET_DISPLAY = {
    'ICU_composite_risk_score': 'MIMIC-IV ICU',
    'eICU_composite_risk_score': 'eICU',
    'NHANES_metabolic_score': 'NHANES',
    'UCI_Heart_Cleveland_num': 'UCI Cleveland',
    'UCI_HydraulicSys_fault_score': 'UCI Hydraulic',
    'UCI_CTG_NSPbin': 'UCI CTG',
}

# Dataset order for consistent layout
DATASET_ORDER = [
    'ICU_composite_risk_score',
    'eICU_composite_risk_score', 
    'NHANES_metabolic_score',
    'UCI_CTG_NSPbin',
    'UCI_Heart_Cleveland_num',
    'UCI_HydraulicSys_fault_score'
]


def create_heatmap(df, dataset_name, output_path, annotate=True, figsize=(4, 6), dpi=300):
    """Create agreement heatmap for a single dataset"""
    
    # Filter to features with guidelines
    df = df[df['guideline'].notna()].copy()
    
    if df.empty:
        print(f"  [WARN] No features with guidelines for {dataset_name}")
        return None
    
    # Sort by error category (green first), then by feature name
    df = df.sort_values(['error_cat', 'feature']).reset_index(drop=True)
    
    # Prepare data
    n_features = len(df)
    
    # Create figure
    fig_height = max(3, n_features * 0.6)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    
    # Create color array
    cats = df['error_cat'].values.astype(float).reshape(-1, 1)
    
    # Setup colormap
    cmap = ListedColormap([COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_NA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    # Plot heatmap
    im = ax.imshow(cats, cmap=cmap, norm=norm, aspect='auto')
    
    # Y-axis labels (features with units)
    ax.set_yticks(np.arange(n_features))
    labels = []
    for _, row in df.iterrows():
        label = row.get('display_name', row['feature'])
        unit = row.get('unit', '')
        if unit and pd.notna(unit) and unit != '':
            label = f"{label} ({unit})"
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=10, fontweight='medium')
    
    # X-axis - hide
    ax.set_xticks([])
    ax.set_xlabel('')
    
    # Title
    display_name = DATASET_DISPLAY.get(dataset_name, dataset_name)
    ax.set_title(f'{display_name}', fontsize=12, fontweight='bold', pad=10)
    
    # Annotations
    if annotate:
        for i, row in df.iterrows():
            median = row.get('median', np.nan)
            rel_error = row.get('rel_error', np.nan)
            
            text_lines = []
            if np.isfinite(median):
                # Format based on magnitude
                if abs(median) >= 100:
                    text_lines.append(f"{median:.0f}")
                elif abs(median) >= 10:
                    text_lines.append(f"{median:.1f}")
                else:
                    text_lines.append(f"{median:.2f}")
            
            if np.isfinite(rel_error):
                pct = int(rel_error * 100)
                text_lines.append(f"$\\Delta${pct}%")
            
            if text_lines:
                text = '\n'.join(text_lines)
                # White text on colored background, black on grey
                color = "white" if row['error_cat'] < 3 else "black"
                ax.text(0, i, text, ha='center', va='center',
                       fontsize=9, color=color, fontweight='bold')
    
    # Legend
    handles = [
        mpatches.Patch(facecolor=COLOR_GREEN, edgecolor='none', label='≤10%'),
        mpatches.Patch(facecolor=COLOR_YELLOW, edgecolor='none', label='≤20%'),
        mpatches.Patch(facecolor=COLOR_RED, edgecolor='none', label='>20%'),
    ]
    ax.legend(handles=handles, title='Relative Error',
             loc='center left', bbox_to_anchor=(1.05, 0.5),
             frameon=False, fontsize=9, title_fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def create_combined_heatmap_2x3(df, output_path, annotate=True, dpi=300):
    """Create 2x3 layout heatmaps for 6 datasets"""
    
    # Setup colormap
    cmap = ListedColormap([COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_NA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.35, wspace=0.30, top=0.92, bottom=0.08)
    
    # Dataset layout
    datasets_layout = [
        ['ICU_composite_risk_score', 'eICU_composite_risk_score', 'NHANES_metabolic_score'],
        ['UCI_CTG_NSPbin', 'UCI_Heart_Cleveland_num', 'UCI_HydraulicSys_fault_score']
    ]
    
    for row_idx, row_datasets in enumerate(datasets_layout):
        for col_idx, ds_name in enumerate(row_datasets):
            ax = axes[row_idx, col_idx]
            
            ds_data = df[df['dataset'] == ds_name].copy()
            ds_data = ds_data[ds_data['guideline'].notna()]
            
            if ds_data.empty:
                ax.text(0.5, 0.5, f"{DATASET_DISPLAY.get(ds_name, ds_name)}\nNo Data",
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='gray')
                ax.set_frame_on(True)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Sort by error category, then feature
            ds_data = ds_data.sort_values(['error_cat', 'feature']).reset_index(drop=True)
            n_features = len(ds_data)
            
            # Create color array
            cats = ds_data['error_cat'].values.astype(float).reshape(-1, 1)
            
            # Plot heatmap
            im = ax.imshow(cats, cmap=cmap, norm=norm, aspect='auto')
            
            # Y-axis labels
            ax.set_yticks(np.arange(n_features))
            labels = []
            for _, row in ds_data.iterrows():
                label = row.get('display_name', row['feature'])
                unit = row.get('unit', '')
                if unit and pd.notna(unit) and unit != '':
                    label = f"{label} ({unit})"
                labels.append(label)
            ax.set_yticklabels(labels, fontsize=9, fontweight='medium')
            
            # X-axis - hide
            ax.set_xticks([])
            
            # Title
            display_name = DATASET_DISPLAY.get(ds_name, ds_name)
            ax.set_title(f'{display_name}', fontsize=11, fontweight='bold', pad=8)
            
            # Annotations
            if annotate:
                for i, row in ds_data.iterrows():
                    median = row.get('median', np.nan)
                    rel_error = row.get('rel_error', np.nan)
                    
                    text_lines = []
                    if np.isfinite(median):
                        if abs(median) >= 100:
                            text_lines.append(f"{median:.0f}")
                        elif abs(median) >= 10:
                            text_lines.append(f"{median:.1f}")
                        else:
                            text_lines.append(f"{median:.2f}")
                    
                    if np.isfinite(rel_error):
                        pct = int(rel_error * 100)
                        text_lines.append(f"$\\Delta${pct}%")
                    
                    if text_lines:
                        text = '\n'.join(text_lines)
                        color = "white" if row['error_cat'] < 3 else "black"
                        ax.text(0, i, text, ha='center', va='center',
                               fontsize=8, color=color, fontweight='bold')
    
    # Add shared legend at bottom
    handles = [
        mpatches.Patch(facecolor=COLOR_GREEN, edgecolor='none', label='≤10%'),
        mpatches.Patch(facecolor=COLOR_YELLOW, edgecolor='none', label='≤20%'),
        mpatches.Patch(facecolor=COLOR_RED, edgecolor='none', label='>20%'),
    ]
    fig.legend(handles=handles, title='Relative Error',
              loc='lower center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, frameon=True, fontsize=10, title_fontsize=10)
    
    # Main title
    fig.suptitle('Threshold Alignment with Clinical Guidelines', 
                fontsize=14, fontweight='bold', y=0.97)
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def create_combined_heatmap(df, datasets, output_path, annotate=True, dpi=300):
    """Create side-by-side heatmaps for multiple datasets (legacy function)"""
    
    # Filter datasets that have data
    valid_datasets = []
    for ds in datasets:
        ds_data = df[df['dataset'] == ds]
        ds_data = ds_data[ds_data['guideline'].notna()]
        if not ds_data.empty:
            valid_datasets.append((ds, ds_data))
    
    if not valid_datasets:
        print("[WARN] No valid datasets for combined heatmap")
        return None
    
    n_datasets = len(valid_datasets)
    
    # Calculate figure size based on number of features
    max_features = max(len(d[1]) for d in valid_datasets)
    fig_height = max(4, max_features * 0.5)
    fig_width = 3.5 * n_datasets + 1
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(fig_width, fig_height))
    if n_datasets == 1:
        axes = [axes]
    
    # Setup colormap
    cmap = ListedColormap([COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_NA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    for idx, (ds_name, ds_data) in enumerate(valid_datasets):
        ax = axes[idx]
        
        # Sort by error category, then feature
        ds_data = ds_data.sort_values(['error_cat', 'feature']).reset_index(drop=True)
        n_features = len(ds_data)
        
        # Create color array
        cats = ds_data['error_cat'].values.astype(float).reshape(-1, 1)
        
        # Plot heatmap
        im = ax.imshow(cats, cmap=cmap, norm=norm, aspect='auto')
        
        # Y-axis labels
        ax.set_yticks(np.arange(n_features))
        labels = []
        for _, row in ds_data.iterrows():
            label = row.get('display_name', row['feature'])
            unit = row.get('unit', '')
            if unit and pd.notna(unit) and unit != '':
                label = f"{label} ({unit})"
            labels.append(label)
        ax.set_yticklabels(labels, fontsize=9, fontweight='medium')
        
        # X-axis - hide
        ax.set_xticks([])
        
        # Title
        display_name = DATASET_DISPLAY.get(ds_name, ds_name)
        ax.set_title(f'{display_name}', fontsize=11, fontweight='bold', pad=8)
        
        # Annotations
        if annotate:
            for i, row in ds_data.iterrows():
                median = row.get('median', np.nan)
                rel_error = row.get('rel_error', np.nan)
                
                text_lines = []
                if np.isfinite(median):
                    if abs(median) >= 100:
                        text_lines.append(f"{median:.0f}")
                    elif abs(median) >= 10:
                        text_lines.append(f"{median:.1f}")
                    else:
                        text_lines.append(f"{median:.2f}")
                
                if np.isfinite(rel_error):
                    pct = int(rel_error * 100)
                    text_lines.append(f"$\\Delta${pct}%")
                
                if text_lines:
                    text = '\n'.join(text_lines)
                    color = "white" if row['error_cat'] < 3 else "black"
                    ax.text(0, i, text, ha='center', va='center',
                           fontsize=8, color=color, fontweight='bold')
    
    # Add shared legend to the right
    handles = [
        mpatches.Patch(facecolor=COLOR_GREEN, edgecolor='none', label='≤10%'),
        mpatches.Patch(facecolor=COLOR_YELLOW, edgecolor='none', label='≤20%'),
        mpatches.Patch(facecolor=COLOR_RED, edgecolor='none', label='>20%'),
    ]
    fig.legend(handles=handles, title='Relative Error',
              loc='center right', bbox_to_anchor=(0.98, 0.5),
              frameon=False, fontsize=9, title_fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate agreement heatmaps for threshold analysis'
    )
    parser.add_argument('--csv', required=True,
                       help='Path to aggregated threshold summary CSV')
    parser.add_argument('--outdir', default='figs/heatmaps',
                       help='Output directory')
    parser.add_argument('--datasets', nargs='*',
                       help='Specific datasets to plot (default: all)')
    parser.add_argument('--combined', action='store_true',
                       help='Create combined multi-dataset figure')
    parser.add_argument('--layout_2x3', action='store_true', default=True,
                       help='Use 2x3 layout for combined figure (default)')
    parser.add_argument('--annotate', action='store_true', default=True,
                       help='Annotate cells with values')
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
    
    # Create individual heatmaps
    for dataset in datasets:
        ds_data = df[df['dataset'] == dataset].copy()
        if ds_data.empty:
            print(f"  [SKIP] No data for {dataset}")
            continue
        
        output_path = os.path.join(args.outdir, f'heatmap_{dataset}.png')
        result = create_heatmap(ds_data, dataset, output_path,
                               annotate=args.annotate, dpi=args.dpi)
        if result:
            print(f"  [OK] {result}")
        
        # Also save PDF
        pdf_path = output_path.replace('.png', '.pdf')
        create_heatmap(ds_data, dataset, pdf_path,
                      annotate=args.annotate, dpi=args.dpi)
    
    # Create combined figure if requested
    if args.combined:
        # 2x3 layout version
        if args.layout_2x3:
            combined_2x3_path = os.path.join(args.outdir, 'heatmap_combined_2x3.png')
            result = create_combined_heatmap_2x3(df, combined_2x3_path,
                                                annotate=args.annotate, dpi=args.dpi)
            if result:
                print(f"\n[OK] Combined 2x3 heatmap: {result}")
            
            # PDF version
            pdf_path = combined_2x3_path.replace('.png', '.pdf')
            create_combined_heatmap_2x3(df, pdf_path, annotate=args.annotate, dpi=args.dpi)
        
        # Legacy row layout
        combined_path = os.path.join(args.outdir, 'heatmap_combined.png')
        result = create_combined_heatmap(df, datasets, combined_path,
                                        annotate=args.annotate, dpi=args.dpi)
        if result:
            print(f"[OK] Combined heatmap: {result}")
        
        # PDF version
        pdf_path = combined_path.replace('.png', '.pdf')
        create_combined_heatmap(df, datasets, pdf_path,
                               annotate=args.annotate, dpi=args.dpi)
    
    print("\n[DONE] Heatmap generation complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())