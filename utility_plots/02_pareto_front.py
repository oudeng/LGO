#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_pareto_front.py - Generate Pareto front scatter plots

This script generates scatter plots showing the Pareto fronts 
(complexity vs loss/CV loss) for different methods and datasets.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CUSTOMIZABLE TITLES - MODIFY HERE
# ========================================
FIG_SUPTITLE = "Pareto Fronts"  # Main figure title

TITLES = {
    "ICU": "ICU",
    "NHANES": "NHANES",
    "CTG": "CTG",
    "Cleveland": "Cleveland",
    "Hydraulic": "Hydraulic"
}

# Dataset name mapping
NAME_MAP = { 
    "overall_NHANES_metabolic_score": "NHANES",
    "overall_ICU_composite_risk_score": "ICU",
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
    "lgo_base": r"LGO$_\mathrm{base}$",
    "lgo_soft": r"LGO$_\mathrm{soft}$",
    "lgo_hard": r"LGO$_\mathrm{hard}$"
}

# Method order for consistent display
METHOD_ORDER = ["pysr", "pstree", "rils_rols", "operon", "lgo_base", "lgo_soft", "lgo_hard"]

# Color palette - using specified colors for better distinction
METHOD_COLORS = {
    "pysr": "#2166ac",      # blue
    "pstree": "#762a83",    # purple
    "rils_rols": "#92c5de", # light blue
    "operon": "#5aae61",    # green
    "lgo_base": "#f4a582",  # light orange
    "lgo_soft": "#d6604d",  # red-orange
    "lgo_hard": "#b2182b"   # dark red
}

# Mapping for various possible experiment names to our standard names
EXPERIMENT_MAPPING = {
    'base': 'base',
    'lgo': 'base',
    'soft': 'soft',
    'lgo_soft': 'soft',
    'hard': 'hard',
    'lgo_hard': 'hard',
    'lgosoft': 'soft',
    'lgohard': 'hard',
}


def read_pareto_data(csv_path):
    """
    Read pareto front data from CSV file.
    Handles both new format (with 'loss' column) and old format (with 'value' column).
    Also handles experiment column for LGO variants.
    """
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check for required columns and rename if necessary
        if 'loss' in df.columns and 'value' not in df.columns:
            # New format: rename 'loss' to 'value' for compatibility
            df = df.rename(columns={'loss': 'value'})
        
        # Check if we have the minimum required columns
        if not {'value', 'complexity', 'method'}.issubset(df.columns):
            print(f"Warning: Missing required columns in {csv_path}")
            print(f"  Available columns: {df.columns.tolist()}")
            return None
        
        # Handle experiment column to distinguish LGO variants
        if 'experiment' in df.columns:
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # For LGO method, combine with experiment to create variants
            lgo_mask = df['method'] == 'lgo'
            if lgo_mask.any():
                # Map experiment names to standard names
                df.loc[lgo_mask, 'experiment'] = df.loc[lgo_mask, 'experiment'].map(
                    lambda x: EXPERIMENT_MAPPING.get(str(x).lower().strip(), x)
                )
                
                # Create combined method names for LGO
                df.loc[lgo_mask, 'method'] = df.loc[lgo_mask].apply(
                    lambda row: f"lgo_{row['experiment']}", axis=1
                )
        
        # If there's a dataset column, filter for the specific dataset
        if 'dataset' in df.columns:
            unique_datasets = df['dataset'].unique()
            if len(unique_datasets) > 0:
                # Use the first dataset or filter based on file path
                dataset_name = csv_path.parent.parent.name
                if dataset_name in NAME_MAP:
                    target_dataset = NAME_MAP[dataset_name]
                    # Try to match dataset name
                    matching = df[df['dataset'].str.contains(dataset_name, case=False, na=False)]
                    if not matching.empty:
                        df = matching
        
        return df
        
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def pareto_scatter(ax, root, dataset_name, custom_title=None):
    """Create Pareto front scatter plot for a dataset with proper method colors."""
    
    # Try multiple possible locations for pareto_front.csv
    possible_paths = [
        Path(root) / "aggregated" / "pareto_front.csv",
        Path(root) / "pareto_front.csv",
        Path("/mnt/user-data/uploads") / "pareto_front.csv"  # Fallback to uploaded file
    ]
    
    df = None
    for p in possible_paths:
        if p.exists():
            df = read_pareto_data(p)
            if df is not None:
                break
    
    if df is None:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: pareto_front.csv missing or incomplete")
        return None
    
    try:
        # Filter to only include methods we want to display
        valid_methods = df['method'].isin(METHOD_ORDER)
        df_filtered = df[valid_methods].copy()
        
        if df_filtered.empty:
            # If no exact matches, try to show what we have
            df_filtered = df.copy()
        
        # Create scatter plot with custom colors
        for method in METHOD_ORDER:
            method_data = df_filtered[df_filtered['method'] == method]
            if not method_data.empty:
                color = METHOD_COLORS.get(method, 'gray')
                label = METHOD_DISPLAY.get(method, method)
                ax.scatter(method_data['complexity'], method_data['value'], 
                          color=color, label=label, s=40, alpha=0.8)
        
        ax.set_xlabel("Complexity")
        ax.set_ylabel("CV Loss")
        ax.set_title(custom_title if custom_title else dataset_name)
        
        # Set reasonable axis limits if needed
        if df_filtered['complexity'].max() > 100:
            ax.set_xlim(0, df_filtered['complexity'].max() * 1.1)
        
        # Remove legend from individual subplot (will be shown in separate legend area)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        return df_filtered
        
    except Exception as e:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: Error creating plot")
        print(f"Error in pareto_scatter for {dataset_name}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Generate Pareto front scatter plots")
    ap.add_argument("--roots", nargs="+", 
                    help="List of dataset output roots (optional)")
    ap.add_argument("--csv", 
                    help="Direct path to pareto_front.csv file (optional)")
    ap.add_argument("--outdir", default="utility_plots/figs",
                    help="Output directory for figures")
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating Pareto front plots...")
    
    # If a single CSV file is provided, create a simple plot
    if args.csv:
        df = read_pareto_data(Path(args.csv))
        if df is not None:
            # Create a simple figure with all datasets if present
            if 'dataset' in df.columns:
                datasets = df['dataset'].unique()
                n_datasets = len(datasets)
                
                if n_datasets > 1:
                    # Multiple datasets in one CSV
                    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5))
                    if n_datasets == 1:
                        axes = [axes]
                    
                    for ax, dataset in zip(axes, datasets):
                        df_subset = df[df['dataset'] == dataset]
                        
                        # Plot with custom colors
                        for method in METHOD_ORDER:
                            method_data = df_subset[df_subset['method'] == method]
                            if not method_data.empty:
                                color = METHOD_COLORS.get(method, 'gray')
                                label = METHOD_DISPLAY.get(method, method)
                                ax.scatter(method_data['complexity'], method_data['value'], 
                                          color=color, label=label, s=40, alpha=0.8)
                        
                        ax.set_xlabel("Complexity")
                        ax.set_ylabel("CV Loss")
                        ax.set_title(dataset.split('_')[-1] if '_' in dataset else dataset)
                        ax.legend(fontsize=8)
                else:
                    # Single dataset
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Plot with custom colors
                    for method in METHOD_ORDER:
                        method_data = df[df['method'] == method]
                        if not method_data.empty:
                            color = METHOD_COLORS.get(method, 'gray')
                            label = METHOD_DISPLAY.get(method, method)
                            ax.scatter(method_data['complexity'], method_data['value'], 
                                      color=color, label=label, s=40, alpha=0.8)
                    
                    ax.set_xlabel("Complexity")
                    ax.set_ylabel("CV Loss")
                    ax.set_title("Pareto Front")
                    ax.legend()
            else:
                # No dataset column, plot all data
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot with custom colors
                for method in METHOD_ORDER:
                    method_data = df[df['method'] == method]
                    if not method_data.empty:
                        color = METHOD_COLORS.get(method, 'gray')
                        label = METHOD_DISPLAY.get(method, method)
                        ax.scatter(method_data['complexity'], method_data['value'], 
                                  color=color, label=label, s=40, alpha=0.8)
                
                ax.set_xlabel("Complexity")
                ax.set_ylabel("CV Loss")
                ax.set_title("Pareto Front")
                ax.legend()
            
            plt.tight_layout()
            output_path = Path(args.outdir) / "pareto.png"
            fig.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()
            return
    
    # Original multi-dataset layout (when --roots is provided)
    if args.roots:
        fig = plt.figure(figsize=(15, 8))
        
        # Use 100 columns grid for precise control
        gs = gridspec.GridSpec(2, 100, figure=fig,
                              height_ratios=[1, 1],
                              hspace=0.3, wspace=0.1)
        
        # First row layout
        ax_icu = fig.add_subplot(gs[0, 0:39])
        ax_nhanes = fig.add_subplot(gs[0, 45:84])
        ax_legend = fig.add_subplot(gs[0, 85:100])
        
        # Second row layout
        ax_ctg = fig.add_subplot(gs[1, 0:29])
        ax_cleveland = fig.add_subplot(gs[1, 34:63])
        ax_hydraulic = fig.add_subplot(gs[1, 70:99])
        
        datasets_order = [
            ("overall_ICU_composite_risk_score", ax_icu, "ICU"),
            ("overall_NHANES_metabolic_score", ax_nhanes, "NHANES"),
            ("overall_UCI_CTG_NSPbin", ax_ctg, "CTG"),
            ("overall_UCI_Heart_Cleveland_num", ax_cleveland, "Cleveland"),
            ("overall_UCI_HydraulicSys_fault_score", ax_hydraulic, "Hydraulic")
        ]
        
        methods_found = set()
        for root_name, ax, dset_key in datasets_order:
            root = [r for r in args.roots if Path(r).name == root_name]
            if root:
                dname = NAME_MAP.get(root_name, root_name)
                # Use custom title from TITLES
                df = pareto_scatter(ax, root[0], dname, custom_title=TITLES.get(dset_key, dname))
                if df is not None and "method" in df.columns:
                    methods_found.update(df["method"].unique())
        
        # Create shared legend with all methods in proper order
        ax_legend.axis('off')
        legend_patches = []
        for method in METHOD_ORDER:
            # Only include methods that were found in the data or always show all
            if method in methods_found or True:  # Show all methods regardless
                display_name = METHOD_DISPLAY[method]
                color = METHOD_COLORS[method]
                patch = mpatches.Patch(color=color, label=display_name, alpha=0.8)
                legend_patches.append(patch)
        
        ax_legend.legend(handles=legend_patches, loc='upper left', 
                        title='Method', frameon=True, fontsize=11, title_fontsize=12)
        
        plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.98)
        
        output_path = Path(args.outdir) / "02_pareto.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    # If neither --roots nor --csv is provided, try to use the uploaded file
    if not args.roots and not args.csv:
        uploaded_csv = Path("/mnt/user-data/uploads/pareto_front.csv")
        if uploaded_csv.exists():
            print("Using uploaded pareto_front.csv file...")
            args.csv = str(uploaded_csv)
            main()  # Recursive call with csv argument set
        else:
            print("Error: Please provide either --roots or --csv argument")
            print("Usage:")
            print("  python 02_pareto_front.py --csv path/to/pareto_front.csv")
            print("  python 02_pareto_front.py --roots path/to/dataset1 path/to/dataset2 ...")


if __name__ == "__main__":
    main()