#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_threshold_analysis.py - Visualize learned thresholds vs medical guidelines

This script creates visualizations showing how LGU learns interpretable thresholds
that align with medical guidelines, demonstrating interpretability advantage.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CUSTOMIZABLE TITLES
# ========================================
FIG_SUPTITLE = "Learned Thresholds vs Medical Guidelines"

TITLES = {
    "ICU": "ICU Risk",
    "NHANES": "NHANES Metabolic", 
    "CTG": "CTG Fetal",
    "Cleveland": "Cleveland Heart",
    "Hydraulic": "Hydraulic Fault"
}

NAME_MAP = { 
    "overall_NHANES_metabolic_score": "NHANES",
    "overall_ICU_composite_risk_score": "ICU",
    "overall_UCI_CTG_NSPbin": "CTG",
    "overall_UCI_Heart_Cleveland_num": "Cleveland",
    "overall_UCI_HydraulicSys_fault_score": "Hydraulic"
}

# Key features to highlight per dataset
KEY_FEATURES = {
    "ICU": ["lactate", "creatinine", "map_mmhg", "spo2_min", "gcs"],
    "NHANES": ["glucose", "triglycerides", "hdl_cholesterol", "bmi", "systolic_bp"],
    "CTG": ["LB", "UB", "ASTV", "AC", "DS"],
    "Cleveland": ["chol", "thalach", "oldpeak", "age", "trestbps"],
    "Hydraulic": ["PS1_mean", "FS1_std", "VS1_mean", "TS1_std", "CE_mean"]
}


def read_threshold_data(root):
    """Read threshold analysis results."""
    p = Path(root) / "aggregated" / "thresholds_units.csv"
    if not p.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(p)
    
    # Also try to read audit data if available
    p_audit = Path(root) / "aggregated" / "threshold_audit.csv"
    if p_audit.exists():
        df_audit = pd.read_csv(p_audit)
        # Merge if possible
        merge_cols = ['feature']
        if all(c in df.columns for c in merge_cols) and all(c in df_audit.columns for c in merge_cols):
            df = pd.merge(df, df_audit[['feature', 'guideline', 'hit_10pct', 'hit_20pct']], 
                         on='feature', how='left').drop_duplicates()
    
    return df


def read_guidelines(guideline_path):
    """Read medical guidelines from YAML file."""
    if not guideline_path or not Path(guideline_path).exists():
        return {}
    
    with open(guideline_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Flatten guidelines
    guidelines = {}
    if 'global' in data:
        guidelines.update(data['global'])
    if 'datasets' in data:
        for dataset, values in data['datasets'].items():
            if isinstance(values, dict):
                guidelines.update(values)
    
    return guidelines


def plot_threshold_comparison(ax, df, dataset_name, guidelines, key_features=None):
    """Create threshold comparison plot for a dataset."""
    if df.empty:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No data")
        return
    
    # Filter for key features if specified
    if key_features:
        df = df[df['feature'].isin(key_features)]
    
    if df.empty:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No key features")
        return
    
    # Focus on LGU experiments
    if 'experiment' in df.columns:
        df = df[df['experiment'].isin(['lgu_soft', 'lgu_hard'])]
    
    # Prepare data for plotting
    plot_data = []
    features = df['feature'].unique()[:5]  # Limit to top 5 features
    
    for feat in features:
        feat_df = df[df['feature'] == feat]
        
        # Add guideline value if available
        guideline_val = guidelines.get(feat, np.nan)
        if pd.notna(guideline_val):
            plot_data.append({
                'Feature': feat[:10],  # Truncate long names
                'Source': 'Guideline',
                'Value': guideline_val
            })
        
        # Add learned thresholds
        for _, row in feat_df.iterrows():
            if pd.notna(row.get('b_raw_median', row.get('b_z_median'))):
                plot_data.append({
                    'Feature': feat[:10],
                    'Source': row.get('experiment', 'LGU'),
                    'Value': row.get('b_raw_median', row.get('b_z_median'))
                })
    
    if not plot_data:
        ax.set_axis_off()
        ax.set_title(f"{dataset_name}: No threshold data")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped bar plot
    sns.barplot(data=plot_df, x='Feature', y='Value', hue='Source', ax=ax)
    
    ax.set_title(TITLES.get(dataset_name, dataset_name))
    ax.set_xlabel("")
    ax.set_ylabel("Threshold Value")
    ax.tick_params(axis='x', rotation=45)
    
    # Adjust legend
    if ax.get_legend():
        ax.legend(loc='upper right', fontsize=8)


def plot_threshold_accuracy(ax, all_data):
    """Plot accuracy of threshold learning across datasets."""
    if not all_data:
        ax.set_axis_off()
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Calculate hit rates for guidelines
    if 'hit_20pct' in combined.columns:
        summary = combined.groupby(['Dataset', 'experiment'])['hit_20pct'].mean().reset_index()
        summary['hit_20pct'] *= 100  # Convert to percentage
        
        summary_pivot = summary.pivot(index='Dataset', columns='experiment', values='hit_20pct')
        
        summary_pivot.plot(kind='bar', ax=ax, width=0.7)
        ax.set_title("Threshold Alignment with Guidelines (±20%)")
        ax.set_ylabel("Alignment Rate (%)")
        ax.set_xlabel("")
        ax.legend(title="Experiment")
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', fontsize=8)


def main():
    ap = argparse.ArgumentParser(description="Visualize learned thresholds analysis")
    ap.add_argument("--roots", nargs="+", required=True, 
                    help="List of dataset output roots")
    ap.add_argument("--guidelines", help="Path to guidelines.yaml file")
    ap.add_argument("--outdir", default="utility_plots/figs",
                    help="Output directory for figures")
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("Creating threshold analysis visualizations...")
    
    # Load guidelines
    guidelines = read_guidelines(args.guidelines) if args.guidelines else {}
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Dataset order
    datasets_order = [
        ("overall_ICU_composite_risk_score", "ICU"),
        ("overall_NHANES_metabolic_score", "NHANES"),
        ("overall_UCI_CTG_NSPbin", "CTG"),
        ("overall_UCI_Heart_Cleveland_num", "Cleveland"),
        ("overall_UCI_HydraulicSys_fault_score", "Hydraulic")
    ]
    
    # Individual dataset plots
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    all_data = []
    
    for (root_name, dset_key), (row, col) in zip(datasets_order, positions):
        ax = fig.add_subplot(gs[row, col])
        
        root = [r for r in args.roots if Path(r).name == root_name]
        if root:
            df = read_threshold_data(root[0])
            if not df.empty:
                df['Dataset'] = dset_key
                all_data.append(df)
            plot_threshold_comparison(ax, df, dset_key, guidelines, 
                                    KEY_FEATURES.get(dset_key))
    
    # Summary plot
    ax_summary = fig.add_subplot(gs[2, :])
    plot_threshold_accuracy(ax_summary, all_data)
    
    plt.suptitle(FIG_SUPTITLE, fontsize=14, y=0.98)
    
    output_path = Path(args.outdir) / "05_threshold_analysis.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()