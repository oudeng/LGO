# -*- coding: utf-8 -*-
"""
04_thresholds_plot.py - Generate publication-ready combined threshold figure

Creates a 2x2 subplot layout combining ICU and NHANES results:
- Row 1: ICU agreement heatmap (left) and error distribution (right)
- Row 2: NHANES agreement heatmap (left) and error distribution (right)

Improvements:
- Command-line arguments for input/output paths
- Much narrower heatmaps (half width) to save space
- LaTeX-style delta notation for percentage differences
- Reduced height for compact layout
- Larger fonts for better readability
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

# Enable LaTeX rendering for delta symbol
plt.rcParams['text.usetex'] = False  # Use matplotlib's internal LaTeX renderer
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for LaTeX-like appearance

# Feature name and unit mapping for standardized display
FEATURE_NAME_MAP = {
    'lactate_mmol_l': ('Lactate', 'mmol/L'),
    'map_mmhg': ('MAP', 'mmHg'),
    'resprate_max': ('Respiratory rate', 'min$^{-1}$'),
    'hdl_cholesterol': ('HDL', 'mg/dL'),
    'systolic_bp': ('SBP', 'mmHg'),
    'waist_circumference': ('Waist circumference', 'cm'),
    'fasting_glucose': ('Fasting glucose', 'mg/dL')
}

# Color scheme
COLOR_OK = "#2ca02c"    # green
COLOR_WARN = "#ffbf00"  # yellow  
COLOR_BAD = "#d62728"   # red
COLOR_NA = "#d9d9d9"    # grey

def get_feature_label(feature_key, unit=None):
    """Get standardized feature label with proper formatting"""
    if feature_key in FEATURE_NAME_MAP:
        name, default_unit = FEATURE_NAME_MAP[feature_key]
        # Use mapped unit or fall back to provided unit or default
        final_unit = default_unit if default_unit else unit
        if final_unit:
            return f"{name} ({final_unit})"
        return name
    else:
        # Fallback to original logic for unmapped features
        label = feature_key.replace('_', ' ')
        if unit and pd.notna(unit):
            label += f" ({unit})"
        return label

def create_agreement_heatmap(ax, df_dataset, dataset_name, annotate=True):
    """Create agreement heatmap for a dataset"""
    # Sort by error category and feature name
    df_dataset = df_dataset.sort_values(['error_cat', 'feature_key']).reset_index(drop=True)
    
    # Create color array
    cats = df_dataset['error_cat'].values.astype(float).reshape(-1, 1)
    
    # Setup colormap
    cmap = ListedColormap([COLOR_OK, COLOR_WARN, COLOR_BAD, COLOR_NA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    # Plot heatmap
    im = ax.imshow(cats, cmap=cmap, norm=norm, aspect='auto')
    
    # Y-axis labels (features) with standardized names
    ax.set_yticks(np.arange(len(df_dataset)))
    labels = []
    for _, row in df_dataset.iterrows():
        label = get_feature_label(row['feature_key'], row.get('unit'))
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=10, fontweight='medium')  # Slightly reduced from 11 for narrower heatmap
    
    # X-axis - hide for cleaner look
    ax.set_xticks([])
    ax.set_xlabel('')
    
    # Title
    title_map = {
        'ICU_composite_risk_score': 'ICU',
        'NHANES_metabolic_score': 'NHANES'
    }
    ax.set_title(f'{title_map.get(dataset_name, dataset_name)}', 
                 fontsize=11, fontweight='bold', pad=6)  # Slightly reduced fontsize and padding
    
    # Annotations with larger font and delta notation
    if annotate:
        for i, row in df_dataset.iterrows():
            med = row.get('median', np.nan)
            err = row.get('rel_error', np.nan)
            text = ""
            if np.isfinite(med):
                text += f"{med:.1f}"  # Reduced precision for space
            if np.isfinite(err):
                # Use delta notation for percentage
                text += f"\n$\\Delta${int(err*100)}%"
            if text:
                color = "white" if row['error_cat'] < 3 else "black"
                ax.text(0, i, text, ha='center', va='center', 
                       fontsize=10, color=color, fontweight='bold')  # Increased from 8 to 10 and made bold
    
    # Create legend - position on the right side of heatmap
    handles = [
        mpatches.Patch(facecolor=COLOR_OK, edgecolor='none', label='≤10%'),
        mpatches.Patch(facecolor=COLOR_WARN, edgecolor='none', label='≤20%'),
        mpatches.Patch(facecolor=COLOR_BAD, edgecolor='none', label='>20%'),
    ]
    
    # Position legend on the right side of heatmap with proper spacing
    ax.legend(handles=handles, title='Relative Error', 
             loc='center left', bbox_to_anchor=(1.02, 0.5),  # Right side with small offset
             frameon=False, fontsize=10, title_fontsize=10, ncol=1)  # Vertical layout
    
def create_error_distribution(ax, df_dataset, dataset_name):
    """Create error distribution plot for a dataset"""
    # Sort by same order as heatmap (error_cat, then feature_key) - MUST match exactly
    df_dataset_sorted = df_dataset.sort_values(['error_cat', 'feature_key']).reset_index(drop=True)
    
    # Only show features with IQR, but preserve their order from the full sorted list
    df_valid = df_dataset_sorted[df_dataset_sorted['has_iqr']].copy().reset_index(drop=True)
    
    if df_valid.empty:
        ax.text(0.5, 0.5, 'No data with IQR', ha='center', va='center')
        return
    
    # IMPORTANT: Reverse the order for display (matplotlib y-axis goes bottom to top)
    # So we need to reverse to match the top-to-bottom order of the heatmap
    df_valid_reversed = df_valid.iloc[::-1].reset_index(drop=True)
    
    y_pos = np.arange(len(df_valid_reversed))
    
    # Plot error bars (Q1-Q3)
    for i, row in df_valid_reversed.iterrows():
        # IQR bar
        ax.barh(i, row['q3'] - row['q1'], left=row['q1'], height=0.5,  # Reduced height
                color='lightblue', edgecolor='black', linewidth=1, alpha=0.7)
        # Median marker
        ax.plot(row['median'], i, 'ko', markersize=6)
        # Guideline marker
        if np.isfinite(row['guideline']):
            ax.plot(row['guideline'], i, 'r|', markersize=10, markeredgewidth=2)
    
    # Y-axis labels with standardized names - use reversed order
    ax.set_yticks(y_pos)
    labels = []
    for _, row in df_valid_reversed.iterrows():
        label = get_feature_label(row['feature_key'], row.get('unit'))
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=11, fontweight='medium')  # Increased from 9
    
    # X-axis
    ax.set_xlabel('Threshold Value', fontsize=10)
    ax.set_ylabel('')
    
    # Title
    title_map = {
        'ICU_composite_risk_score': 'ICU Threshold Distribution',
        'NHANES_metabolic_score': 'NHANES Threshold Distribution'
    }
    ax.set_title(f'{title_map.get(dataset_name, dataset_name)}',
                 fontsize=12, fontweight='bold', pad=8)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend - positioned at top right to avoid overlapping with data
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(color='lightblue', alpha=0.7, label='IQR'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
               markersize=6, label='Median', linestyle=''),
        Line2D([0], [0], marker='|', color='r', markerfacecolor='r', markeredgecolor='r',
               markersize=10, markeredgewidth=2, label='Guideline', linestyle=''),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
             frameon=True, fancybox=False, framealpha=0.95, edgecolor='gray')

def main():
    """Generate combined publication figure with command-line arguments"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate threshold comparison plots')
    parser.add_argument('--csv', required=True, help='Path to v3_thresholds_summary.csv')
    parser.add_argument('--outdir', required=True, help='Output directory for figures')
    args = parser.parse_args()
    
    # Read the summary CSV
    df = pd.read_csv(args.csv)
    
    # Split by dataset
    df_icu = df[df['dataset'] == 'ICU_composite_risk_score'].copy()
    df_nhanes = df[df['dataset'] == 'NHANES_metabolic_score'].copy()
    
    # Create figure with 2x2 subplots - adjusted for proper spacing
    fig = plt.figure(figsize=(15, 7))  # Optimal width for legend and spacing
    
    # Create a gridspec for better control
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig, 
                          width_ratios=[0.125, 0.50],  # Left heatmap halved to 12.5%, right plot stays at 50%
                          height_ratios=[1, 1],
                          wspace=0.80,  # Increased spacing to use the freed space from narrower heatmap
                          hspace=0.35)  # Vertical spacing between rows
    
    # ICU plots (top row)
    ax1 = fig.add_subplot(gs[0, 0])  # ICU agreement heatmap
    ax2 = fig.add_subplot(gs[0, 1])  # ICU error distribution
    
    # NHANES plots (bottom row)
    ax3 = fig.add_subplot(gs[1, 0])  # NHANES agreement heatmap
    ax4 = fig.add_subplot(gs[1, 1])  # NHANES error distribution
    
    # Create plots
    create_agreement_heatmap(ax1, df_icu, 'ICU_composite_risk_score', annotate=True)
    create_error_distribution(ax2, df_icu, 'ICU_composite_risk_score')
    create_agreement_heatmap(ax3, df_nhanes, 'NHANES_metabolic_score', annotate=True)
    create_error_distribution(ax4, df_nhanes, 'NHANES_metabolic_score')
    
    # Add panel labels with larger font
    for ax, label in zip([ax1, ax2, ax3, ax4], ['A', 'B', 'C', 'D']):
        ax.text(-0.2, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom')
    
    # Overall title
    fig.suptitle('LGO Threshold Alignment with Clinical Guidelines', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)
    
    # Save figure
    output_path = os.path.join(args.outdir, 'v3_thresholds_summary.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved combined figure to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Also saved as PDF: {pdf_path}")
    
    # Save high-res version for print
    highres_path = output_path.replace('.png', '_highres.png')
    fig.savefig(highres_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Also saved high-res version: {highres_path}")
    
    plt.close()

if __name__ == "__main__":
    main()