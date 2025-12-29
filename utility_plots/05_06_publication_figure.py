#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_06_publication_figure.py - Generate publication-ready Figure 2

Creates the main threshold analysis figure with multiple layout options:
- 2x2 layout: ICU + NHANES (original)
- 2x3 layout: All 6 datasets (ICU, eICU, NHANES, CTG, Cleveland, Hydraulic)
- 3x2 layout: Clinical focus (ICU, eICU, NHANES in top row)

Features:
- Professional publication quality (600 DPI)
- Panel labels (A, B, C, D, E, F)
- Consistent styling across panels
- Both PNG and PDF output
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

# Configure matplotlib for publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# Color scheme
COLOR_GREEN = "#2ca02c"
COLOR_YELLOW = "#ffbf00"
COLOR_RED = "#d62728"
COLOR_NA = "#d9d9d9"
COLOR_IQR = '#87CEEB'
COLOR_MEDIAN = 'black'
COLOR_GUIDELINE = '#d62728'

# Dataset display names (updated with eICU)
DATASET_DISPLAY = {
    'ICU_composite_risk_score': 'MIMIC-IV ICU',
    'eICU_composite_risk_score': 'eICU',
    'NHANES_metabolic_score': 'NHANES',
    'UCI_Heart_Cleveland_num': 'UCI Cleveland',
    'UCI_HydraulicSys_fault_score': 'UCI Hydraulic',
    'UCI_CTG_NSPbin': 'UCI CTG',
}

# Short names for compact display
DATASET_SHORT = {
    'ICU_composite_risk_score': 'ICU',
    'eICU_composite_risk_score': 'eICU',
    'NHANES_metabolic_score': 'NHANES',
    'UCI_Heart_Cleveland_num': 'Cleveland',
    'UCI_HydraulicSys_fault_score': 'Hydraulic',
    'UCI_CTG_NSPbin': 'CTG',
}

# Dataset order for 2x3 layout
DATASET_ORDER_2x3 = [
    ['ICU_composite_risk_score', 'eICU_composite_risk_score', 'NHANES_metabolic_score'],
    ['UCI_CTG_NSPbin', 'UCI_Heart_Cleveland_num', 'UCI_HydraulicSys_fault_score']
]


def create_heatmap_panel(ax, df, title, annotate=True):
    """Create agreement heatmap panel"""
    
    # Filter and sort
    df = df[df['guideline'].notna()].copy()
    if df.empty:
        ax.text(0.5, 0.5, f"{title}\nNo Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    df = df.sort_values(['error_cat', 'feature']).reset_index(drop=True)
    n_features = len(df)
    
    # Create color array
    cats = df['error_cat'].values.astype(float).reshape(-1, 1)
    
    # Setup colormap
    cmap = ListedColormap([COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_NA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    # Plot heatmap
    im = ax.imshow(cats, cmap=cmap, norm=norm, aspect='auto')
    
    # Y-axis labels
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
    
    # Title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Annotations
    if annotate:
        for i, row in df.iterrows():
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
                       fontsize=9, color=color, fontweight='bold')


def create_distribution_panel(ax, df, title):
    """Create threshold distribution panel"""
    
    # Filter to features with IQR and guideline
    df = df[(df['has_iqr'] == True) & (df['guideline'].notna())].copy()
    
    if df.empty:
        ax.text(0.5, 0.5, f"{title}\nNo Data",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_frame_on(True)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Sort by error category (same as heatmap)
    df = df.sort_values(['error_cat', 'feature']).reset_index(drop=True)
    
    # Reverse for matplotlib y-axis
    df = df.iloc[::-1].reset_index(drop=True)
    
    n_features = len(df)
    y_pos = np.arange(n_features)
    
    # Plot bars and markers
    for i, row in df.iterrows():
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
    for _, row in df.iterrows():
        label = row.get('display_name', row['feature'])
        unit = row.get('unit', '')
        if unit and pd.notna(unit) and unit != '':
            label = f"{label} ({unit})"
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=10, fontweight='medium')
    
    # X-axis and title
    ax.set_xlabel('Threshold Value', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)


def create_publication_figure_2x3(df, output_path, dpi=300):
    """Create 2x3 layout publication figure for all 6 datasets"""
    
    # Setup colormap
    cmap = ListedColormap([COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_NA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    # Create figure with 2x3 layout - increased wspace to avoid label overlap
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.subplots_adjust(hspace=0.35, wspace=0.50, top=0.92, bottom=0.10)
    
    panel_labels = [['A', 'B', 'C'], ['D', 'E', 'F']]
    
    for row_idx, row_datasets in enumerate(DATASET_ORDER_2x3):
        for col_idx, ds_name in enumerate(row_datasets):
            ax = axes[row_idx, col_idx]
            
            ds_data = df[df['dataset'] == ds_name].copy()
            ds_data = ds_data[ds_data['guideline'].notna()]
            
            display_name = DATASET_SHORT.get(ds_name, ds_name)
            
            if ds_data.empty:
                ax.text(0.5, 0.5, f"{display_name}\nNo Data",
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='gray')
                ax.set_frame_on(True)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Sort by error category, then feature
                ds_data = ds_data.sort_values(['error_cat', 'feature']).reset_index(drop=True)
                n_features = len(ds_data)
                
                # Create color array
                cats = ds_data['error_cat'].values.astype(float).reshape(-1, 1)
                
                # Plot heatmap - narrower width (quarter of original)
                # extent: [left, right, bottom, top]
                im = ax.imshow(cats, cmap=cmap, norm=norm, aspect='auto',
                              extent=[-0.125, 0.125, n_features-0.5, -0.5])
                ax.set_xlim(-0.175, 0.175)  # Narrower margins to match bar width
                
                # Y-axis labels
                ax.set_yticks(np.arange(n_features))
                labels = []
                for _, row in ds_data.iterrows():
                    label = row.get('display_name', row['feature'])
                    unit = row.get('unit', '')
                    if unit and pd.notna(unit) and unit != '':
                        label = f"{label} ({unit})"
                    labels.append(label)
                ax.set_yticklabels(labels, fontsize=14, fontweight='medium')
                
                # X-axis - hide
                ax.set_xticks([])
                
                # Title
                ax.set_title(f'{display_name}', fontsize=14, fontweight='bold', pad=8)
                
                # Annotations
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
                               fontsize=13, color=color, fontweight='bold')
            
            # Panel label removed per user request
    
    # Add shared legend at bottom
    handles = [
        mpatches.Patch(facecolor=COLOR_GREEN, edgecolor='none', label='≤10%'),
        mpatches.Patch(facecolor=COLOR_YELLOW, edgecolor='none', label='≤20%'),
        mpatches.Patch(facecolor=COLOR_RED, edgecolor='none', label='>20%'),
    ]
    fig.legend(handles=handles, title='Relative Error',
              loc='lower center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, frameon=True, fontsize=13, title_fontsize=13)
    
    # Main title
    fig.suptitle('LGO Threshold Alignment with Guidelines', 
                fontsize=17, fontweight='bold', y=0.97)
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def create_publication_figure(df, output_path, dpi=300):
    """Create the main publication figure (ICU + NHANES, 2x2 layout)"""
    
    # Filter to ICU and NHANES
    df_icu = df[df['dataset'] == 'ICU_composite_risk_score'].copy()
    df_nhanes = df[df['dataset'] == 'NHANES_metabolic_score'].copy()
    
    # Check data availability
    icu_has_data = len(df_icu[df_icu['guideline'].notna()]) > 0
    nhanes_has_data = len(df_nhanes[df_nhanes['guideline'].notna()]) > 0
    
    if not icu_has_data and not nhanes_has_data:
        print("[ERROR] No data for either ICU or NHANES")
        return None
    
    # Determine layout
    if icu_has_data and nhanes_has_data:
        # 2x2 layout
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig,
                              width_ratios=[0.3, 0.7],
                              height_ratios=[1, 1],
                              wspace=0.35, hspace=0.35)
        
        # Panel A: ICU Heatmap
        ax_a = fig.add_subplot(gs[0, 0])
        create_heatmap_panel(ax_a, df_icu, 'ICU')
        
        # Panel B: ICU Distribution
        ax_b = fig.add_subplot(gs[0, 1])
        create_distribution_panel(ax_b, df_icu, 'ICU Threshold Distribution')
        
        # Panel C: NHANES Heatmap
        ax_c = fig.add_subplot(gs[1, 0])
        create_heatmap_panel(ax_c, df_nhanes, 'NHANES')
        
        # Panel D: NHANES Distribution
        ax_d = fig.add_subplot(gs[1, 1])
        create_distribution_panel(ax_d, df_nhanes, 'NHANES Threshold Distribution')
        
        # Panel labels
        for ax, label in [(ax_a, 'A'), (ax_b, 'B'), (ax_c, 'C'), (ax_d, 'D')]:
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom')
        
        axes_list = [ax_a, ax_b, ax_c, ax_d]
    
    elif icu_has_data:
        # 1x2 layout for ICU only
        fig = plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 2, figure=fig,
                              width_ratios=[0.3, 0.7],
                              wspace=0.35)
        
        ax_a = fig.add_subplot(gs[0, 0])
        create_heatmap_panel(ax_a, df_icu, 'ICU')
        
        ax_b = fig.add_subplot(gs[0, 1])
        create_distribution_panel(ax_b, df_icu, 'ICU Threshold Distribution')
        
        for ax, label in [(ax_a, 'A'), (ax_b, 'B')]:
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom')
        
        axes_list = [ax_a, ax_b]
    
    else:
        # 1x2 layout for NHANES only
        fig = plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 2, figure=fig,
                              width_ratios=[0.3, 0.7],
                              wspace=0.35)
        
        ax_a = fig.add_subplot(gs[0, 0])
        create_heatmap_panel(ax_a, df_nhanes, 'NHANES')
        
        ax_b = fig.add_subplot(gs[0, 1])
        create_distribution_panel(ax_b, df_nhanes, 'NHANES Threshold Distribution')
        
        for ax, label in [(ax_a, 'A'), (ax_b, 'B')]:
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='bottom')
        
        axes_list = [ax_a, ax_b]
    
    # Add legends
    # Heatmap legend (to the right of first heatmap)
    heatmap_handles = [
        mpatches.Patch(facecolor=COLOR_GREEN, edgecolor='none', label='≤10%'),
        mpatches.Patch(facecolor=COLOR_YELLOW, edgecolor='none', label='≤20%'),
        mpatches.Patch(facecolor=COLOR_RED, edgecolor='none', label='>20%'),
    ]
    
    # Distribution legend
    dist_handles = [
        mpatches.Patch(color=COLOR_IQR, alpha=0.7, label='IQR'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MEDIAN,
               markersize=6, label='Median', linestyle=''),
        Line2D([0], [0], marker='|', color=COLOR_GUIDELINE,
               markersize=10, markeredgewidth=2, label='Guideline', linestyle=''),
    ]
    
    # Add legends to distribution panels
    for ax in axes_list[1::2]:  # Every other axis starting from index 1
        ax.legend(handles=dist_handles, loc='upper right', fontsize=9,
                 frameon=True, fancybox=False, framealpha=0.95, edgecolor='gray')
    
    # Add heatmap legend to first heatmap
    axes_list[0].legend(handles=heatmap_handles, title='Relative Error',
                       loc='center left', bbox_to_anchor=(1.05, 0.5),
                       frameon=False, fontsize=9, title_fontsize=9)
    
    # Main title
    fig.suptitle('LGO Threshold Alignment with Clinical Guidelines',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready Figure 2'
    )
    parser.add_argument('--csv', required=True,
                       help='Path to aggregated threshold summary CSV')
    parser.add_argument('--outdir', default='figs/publication',
                       help='Output directory')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--layout_2x3', action='store_true', default=True,
                       help='Also generate 2x3 layout for all 6 datasets')
    args = parser.parse_args()
    
    # Read data
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Create main figure (ICU + NHANES, 2x2)
    print("\nCreating publication figure (ICU + NHANES)...")
    output_path = os.path.join(args.outdir, 'figure2_thresholds.png')
    result = create_publication_figure(df, output_path, dpi=args.dpi)
    if result:
        print(f"  [OK] {result}")
    
    # PDF version
    pdf_path = output_path.replace('.png', '.pdf')
    create_publication_figure(df, pdf_path, dpi=args.dpi)
    print(f"  [OK] {pdf_path}")
    
    # High-res version for print
    highres_path = output_path.replace('.png', '_highres.png')
    create_publication_figure(df, highres_path, dpi=600)
    print(f"  [OK] {highres_path}")
    
    # Create 2x3 layout for all datasets
    if args.layout_2x3:
        print("\nCreating 2x3 layout figure (all 6 datasets)...")
        output_2x3 = os.path.join(args.outdir, 'figure2_thresholds_2x3.png')
        result = create_publication_figure_2x3(df, output_2x3, dpi=args.dpi)
        if result:
            print(f"  [OK] {result}")
        
        # PDF version
        pdf_2x3 = output_2x3.replace('.png', '.pdf')
        create_publication_figure_2x3(df, pdf_2x3, dpi=args.dpi)
        print(f"  [OK] {pdf_2x3}")
        
        # High-res version
        highres_2x3 = output_2x3.replace('.png', '_highres.png')
        create_publication_figure_2x3(df, highres_2x3, dpi=600)
        print(f"  [OK] {highres_2x3}")
    
    print("\n[DONE] Publication figure generation complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())