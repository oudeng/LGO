#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_05_cross_dataset_table.py - Generate cross-dataset summary tables

Creates publication-ready tables summarizing LGO threshold alignment:
- Table 4 style: Feature-level threshold comparison with guidelines
- Summary statistics: Hit rates at 10%/20% bands
- LaTeX formatted output for paper

Features:
- Supports 6 datasets: ICU, eICU, NHANES, CTG, Cleveland, Hydraulic
- Automatic feature selection (anchored features only)
- Relative error calculation and categorization
- Both CSV and LaTeX output formats
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Dataset display names (updated with eICU)
DATASET_DISPLAY = {
    'ICU_composite_risk_score': 'MIMIC-IV ICU',
    'eICU_composite_risk_score': 'eICU',
    'NHANES_metabolic_score': 'NHANES',
    'UCI_Heart_Cleveland_num': 'UCI Cleveland',
    'UCI_HydraulicSys_fault_score': 'UCI Hydraulic',
    'UCI_CTG_NSPbin': 'UCI CTG',
}

# Short names for tables
DATASET_SHORT = {
    'ICU_composite_risk_score': 'ICU',
    'eICU_composite_risk_score': 'eICU',
    'NHANES_metabolic_score': 'NHANES',
    'UCI_Heart_Cleveland_num': 'Cleveland',
    'UCI_HydraulicSys_fault_score': 'Hydraulic',
    'UCI_CTG_NSPbin': 'CTG',
}

# Dataset order for consistent presentation
DATASET_ORDER = ['ICU', 'eICU', 'NHANES', 'CTG', 'Cleveland', 'Hydraulic']


def generate_threshold_table(df, output_path):
    """Generate Table 4 style threshold comparison table"""
    
    # Filter to features with both guideline and threshold
    df = df[(df['guideline'].notna()) & (df['median'].notna())].copy()
    
    if df.empty:
        print("[WARN] No features with both guideline and threshold")
        return None
    
    # Map dataset names to short names
    df['dataset_short'] = df['dataset'].map(DATASET_SHORT)
    
    # Sort by dataset order, then error category
    df['sort_order'] = df['dataset_short'].map({d: i for i, d in enumerate(DATASET_ORDER)})
    df = df.sort_values(['sort_order', 'error_cat', 'feature']).reset_index(drop=True)
    
    # Create table rows
    rows = []
    for _, row in df.iterrows():
        dataset = row['dataset_short']
        feature = row.get('display_name', row['feature'])
        unit = row.get('unit', '')
        
        median = row['median']
        q1 = row.get('q1', np.nan)
        q3 = row.get('q3', np.nan)
        guideline = row['guideline']
        rel_error = row.get('rel_error', np.nan)
        
        # Format median with IQR
        if pd.notna(q1) and pd.notna(q3) and q1 != q3:
            if abs(median) >= 100:
                median_str = f"{median:.0f} [{q1:.0f}, {q3:.0f}]"
            elif abs(median) >= 10:
                median_str = f"{median:.1f} [{q1:.1f}, {q3:.1f}]"
            else:
                median_str = f"{median:.2f} [{q1:.2f}, {q3:.2f}]"
        else:
            if abs(median) >= 100:
                median_str = f"{median:.0f}"
            elif abs(median) >= 10:
                median_str = f"{median:.1f}"
            else:
                median_str = f"{median:.2f}"
        
        # Format guideline
        if abs(guideline) >= 100:
            guideline_str = f"{guideline:.0f}"
        elif abs(guideline) >= 10:
            guideline_str = f"{guideline:.1f}"
        else:
            guideline_str = f"{guideline:.2f}"
        
        # Format relative error
        if pd.notna(rel_error):
            error_str = f"{rel_error*100:.1f}%"
        else:
            error_str = "-"
        
        rows.append({
            'Dataset': dataset,
            'Feature': feature,
            'Unit': unit if pd.notna(unit) else '',
            'Median [Q1, Q3]': median_str,
            'Anchor': guideline_str,
            'Rel. Err.': error_str,
        })
    
    table_df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = output_path.replace('.tex', '.csv')
    table_df.to_csv(csv_path, index=False)
    print(f"  [OK] CSV: {csv_path}")
    
    # Generate LaTeX
    latex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{LGO-discovered thresholds (median [Q1, Q3]) versus domain anchors. Values are in natural units.}',
        r'\label{tab:thresholds}',
        r'\begin{tabular}{llllll}',
        r'\toprule',
        r'Dataset & Feature & Unit & Median [Q1, Q3] & Anchor & Rel. Err. \\',
        r'\midrule',
    ]
    
    current_dataset = None
    for _, row in table_df.iterrows():
        # Add dataset separator
        if row['Dataset'] != current_dataset:
            if current_dataset is not None:
                latex_lines.append(r'\midrule')
            current_dataset = row['Dataset']
        
        # Escape special characters
        feature = row['Feature'].replace('_', r'\_').replace('%', r'\%')
        unit = row['Unit'].replace('%', r'\%')
        
        latex_lines.append(
            f"  {row['Dataset']} & {feature} & {unit} & {row['Median [Q1, Q3]']} & {row['Anchor']} & {row['Rel. Err.']} \\\\"
        )
    
    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    return output_path


def generate_summary_table(df, output_path):
    """Generate summary statistics table"""
    
    rows = []
    
    # Map and sort datasets
    df['dataset_short'] = df['dataset'].map(DATASET_SHORT)
    df['sort_order'] = df['dataset_short'].map({d: i for i, d in enumerate(DATASET_ORDER)})
    
    for dataset in df.sort_values('sort_order')['dataset'].unique():
        ds_data = df[df['dataset'] == dataset]
        display_name = DATASET_SHORT.get(dataset, dataset)
        
        # Total features
        n_total = len(ds_data)
        
        # Features with guidelines
        anchored = ds_data[ds_data['guideline'].notna()]
        n_anchored = len(anchored)
        
        if n_anchored == 0:
            continue
        
        # Agreement statistics
        n_green = (anchored['error_cat'] == 0).sum()
        n_yellow = (anchored['error_cat'] == 1).sum()
        n_red = (anchored['error_cat'] == 2).sum()
        
        hit10_rate = n_green / n_anchored * 100
        hit20_rate = (n_green + n_yellow) / n_anchored * 100
        
        # Median relative error
        median_error = anchored['rel_error'].median() * 100 if anchored['rel_error'].notna().any() else np.nan
        
        rows.append({
            'Dataset': display_name,
            'N Features': n_anchored,
            'Hit@10%': f"{hit10_rate:.0f}%",
            'Hit@20%': f"{hit20_rate:.0f}%",
            'Median Err.': f"{median_error:.1f}%" if pd.notna(median_error) else '-',
            'Green': n_green,
            'Yellow': n_yellow,
            'Red': n_red,
        })
    
    # Add overall row
    total_anchored = df[df['guideline'].notna()]
    if len(total_anchored) > 0:
        n_green = (total_anchored['error_cat'] == 0).sum()
        n_yellow = (total_anchored['error_cat'] == 1).sum()
        n_red = (total_anchored['error_cat'] == 2).sum()
        n_total = len(total_anchored)
        
        rows.append({
            'Dataset': 'Overall',
            'N Features': n_total,
            'Hit@10%': f"{n_green/n_total*100:.0f}%",
            'Hit@20%': f"{(n_green+n_yellow)/n_total*100:.0f}%",
            'Median Err.': f"{total_anchored['rel_error'].median()*100:.1f}%" if total_anchored['rel_error'].notna().any() else '-',
            'Green': n_green,
            'Yellow': n_yellow,
            'Red': n_red,
        })
    
    table_df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = output_path.replace('.tex', '.csv')
    table_df.to_csv(csv_path, index=False)
    print(f"  [OK] CSV: {csv_path}")
    
    # Generate LaTeX
    latex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Threshold alignment summary across datasets.}',
        r'\label{tab:threshold_summary}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Dataset & N & Hit@10\% & Hit@20\% & Median Err. & Green & Yellow \\',
        r'\midrule',
    ]
    
    for i, row in table_df.iterrows():
        if row['Dataset'] == 'Overall':
            latex_lines.append(r'\midrule')
        latex_lines.append(
            f"  {row['Dataset']} & {row['N Features']} & {row['Hit@10%']} & {row['Hit@20%']} & {row['Median Err.']} & {row['Green']} & {row['Yellow']} \\\\"
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
        description='Generate cross-dataset summary tables'
    )
    parser.add_argument('--csv', required=True,
                       help='Path to aggregated threshold summary CSV')
    parser.add_argument('--outdir', default='figs/tables',
                       help='Output directory')
    args = parser.parse_args()
    
    # Read data
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Generate threshold comparison table
    print("\nGenerating threshold comparison table...")
    table_path = os.path.join(args.outdir, 'threshold_table.tex')
    result = generate_threshold_table(df, table_path)
    if result:
        print(f"  [OK] {result}")
    
    # Generate summary table
    print("\nGenerating summary table...")
    summary_path = os.path.join(args.outdir, 'summary_table.tex')
    result = generate_summary_table(df, summary_path)
    if result:
        print(f"  [OK] {result}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("THRESHOLD ALIGNMENT SUMMARY")
    print("="*60)
    
    total_anchored = df[df['guideline'].notna()]
    if len(total_anchored) > 0:
        n_green = (total_anchored['error_cat'] == 0).sum()
        n_yellow = (total_anchored['error_cat'] == 1).sum()
        n_red = (total_anchored['error_cat'] == 2).sum()
        n_total = len(total_anchored)
        
        print(f"\nTotal assessed features: {n_total}")
        print(f"  Green (≤10%): {n_green} ({n_green/n_total*100:.1f}%)")
        print(f"  Yellow (≤20%): {n_yellow} ({n_yellow/n_total*100:.1f}%)")
        print(f"  Red (>20%): {n_red} ({n_red/n_total*100:.1f}%)")
        print(f"\nOverall success (≤20%): {(n_green+n_yellow)/n_total*100:.1f}%")
    
    print("\n[DONE] Table generation complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())