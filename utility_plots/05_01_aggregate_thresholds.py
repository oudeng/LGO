#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_01_aggregate_thresholds.py - Aggregate threshold data from all datasets

This script collects threshold data from multiple experiment directories and
creates a unified summary for downstream analysis and visualization.

Features:
- Collects thresholds_units.csv from all dataset directories
- Merges with guidelines from config/ directories
- Supports 6 datasets: ICU, eICU, NHANES, CTG, Cleveland, Hydraulic
- Filters operator pseudo-features
- Computes relative errors and agreement categories
- Outputs a clean summary CSV for publication figures
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Operator tokens to filter out
OPERATOR_TOKENS = {
    'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'pow', 'exp', 'sin', 'cos',
    'id', 'idf', 'zero', 'one', 'unknown', 'gate_expr', 'lgo_thre', 'lgu_thre',
    'lgo_and2', 'lgo_or2', 'lgo_and3', 'lgoor2', 'lgoand2', 'lgoand3'
}

# Feature name normalization and display mapping
FEATURE_DISPLAY = {
    # ICU / MIMIC-IV features
    'lactate_mmol_l': ('Lactate', 'mmol/L'),
    'map_mmhg': ('MAP', 'mmHg'),
    'resprate_max': ('Respiratory rate', 'min⁻¹'),
    'spo2_min': ('SpO2', '%'),
    'gcs': ('GCS', 'points'),
    'creatinine_mg_dl': ('Creatinine', 'mg/dL'),
    'hemoglobin_min': ('Hemoglobin', 'g/dL'),
    'urine_output_min': ('Urine output', 'mL/kg/hr'),
    'sbp_min': ('SBP', 'mmHg'),
    'vasopressor_dose': ('Vasopressor', 'dose'),
    'vasopressor_use_std': ('Vasopressor use', 'std'),
    'charlson_index': ('Charlson Index', 'points'),
    
    # eICU features (similar to ICU but may have different naming)
    'lactate': ('Lactate', 'mmol/L'),
    'map': ('MAP', 'mmHg'),
    'resprate': ('Respiratory rate', 'min⁻¹'),
    'spo2': ('SpO2', '%'),
    'creatinine': ('Creatinine', 'mg/dL'),
    'hemoglobin': ('Hemoglobin', 'g/dL'),
    'urine_output': ('Urine output', 'mL/kg/hr'),
    'sbp': ('SBP', 'mmHg'),
    'heartrate': ('Heart rate', 'bpm'),
    'temperature': ('Temperature', '°C'),
    'fio2': ('FiO2', '%'),
    'platelet': ('Platelet', '×10³/μL'),
    'wbc': ('WBC', '×10³/μL'),
    'bilirubin': ('Bilirubin', 'mg/dL'),
    'bun': ('BUN', 'mg/dL'),
    'sodium': ('Sodium', 'mEq/L'),
    'potassium': ('Potassium', 'mEq/L'),
    'glucose': ('Glucose', 'mg/dL'),
    
    # NHANES features
    'hdl_cholesterol': ('HDL', 'mg/dL'),
    'systolic_bp': ('SBP', 'mmHg'),
    'diastolic_bp': ('DBP', 'mmHg'),
    'waist_circumference': ('Waist circumference', 'cm'),
    'fasting_glucose': ('Fasting glucose', 'mg/dL'),
    'triglycerides': ('Triglycerides', 'mg/dL'),
    'bmi': ('BMI', 'kg/m²'),
    'age': ('Age', 'years'),
    'hba1c': ('HbA1c', '%'),
    
    # Cleveland features
    'chol': ('Cholesterol', 'mg/dL'),
    'thalach': ('Max heart rate', 'bpm'),
    'oldpeak': ('ST depression', 'mm'),
    'trestbps': ('Resting BP', 'mmHg'),
    'ca': ('Major vessels', 'count'),
    'thal': ('Thalassemia', 'code'),
}

# Alias mapping for guideline lookup
ALIAS_MAP = {
    'lactate_mmol_l': ['lactate', 'lactatemmoll'],
    'creatinine_mg_dl': ['creatinine', 'creatininemgdl'],
    'map_mmhg': ['map', 'mapmmhg', 'mean_arterial_pressure'],
    'spo2_min': ['spo2', 'oxygen_saturation'],
    'resprate_max': ['respiratory_rate', 'resprate'],
    'sbp_min': ['systolic_bp', 'sbp'],
    'hemoglobin_min': ['hemoglobin'],
    'urine_output_min': ['urine_output'],
    'hdl_cholesterol': ['hdl', 'hdlcholesterol'],
    'systolic_bp': ['sbp', 'bp_systolic'],
    'diastolic_bp': ['dbp', 'bp_diastolic'],
    'waist_circumference': ['waist', 'waistcircumference'],
    'fasting_glucose': ['glucose', 'fastingglucose'],
    # eICU aliases
    'lactate': ['lactate_mmol_l'],
    'map': ['map_mmhg'],
    'spo2': ['spo2_min'],
    'resprate': ['resprate_max'],
    'sbp': ['sbp_min'],
    'creatinine': ['creatinine_mg_dl'],
    'hemoglobin': ['hemoglobin_min'],
    'urine_output': ['urine_output_min'],
}


def load_guidelines(config_path):
    """Load guidelines from YAML file"""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    guidelines = {}
    
    # Global guidelines
    if 'global' in data and isinstance(data['global'], dict):
        for k, v in data['global'].items():
            try:
                guidelines[k.lower()] = float(v)
            except (ValueError, TypeError):
                pass
    
    # Dataset-specific guidelines (flatten)
    if 'datasets' in data:
        for ds_name, ds_data in data['datasets'].items():
            if isinstance(ds_data, dict):
                # Handle nested structure (from_ground_truth, additional)
                for section in ['from_ground_truth', 'additional']:
                    if section in ds_data and isinstance(ds_data[section], dict):
                        for k, v in ds_data[section].items():
                            try:
                                guidelines[k.lower()] = float(v)
                            except (ValueError, TypeError):
                                pass
                # Handle flat structure
                for k, v in ds_data.items():
                    if k not in ['from_ground_truth', 'additional']:
                        try:
                            guidelines[k.lower()] = float(v)
                        except (ValueError, TypeError):
                            pass
    
    return guidelines


def load_ground_truth(config_path):
    """Load ground truth from JSON file"""
    import json
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    guidelines = {}
    for k, v in data.items():
        if isinstance(v, dict):
            # Handle structured format with 'value' key
            if 'value' in v:
                try:
                    guidelines[k.lower()] = float(v['value'])
                except (ValueError, TypeError):
                    pass
        else:
            try:
                guidelines[k.lower()] = float(v)
            except (ValueError, TypeError):
                pass
    
    return guidelines


def lookup_guideline(feature, guidelines):
    """Enhanced guideline lookup with alias support"""
    feature_lower = feature.lower().strip()
    
    # Direct match
    if feature_lower in guidelines:
        return guidelines[feature_lower]
    
    # Check aliases
    for base_name, aliases in ALIAS_MAP.items():
        if feature_lower == base_name.lower() or feature_lower in [a.lower() for a in aliases]:
            # Try base name
            if base_name.lower() in guidelines:
                return guidelines[base_name.lower()]
            # Try aliases
            for alias in aliases:
                if alias.lower() in guidelines:
                    return guidelines[alias.lower()]
    
    # Try without underscores
    feature_no_underscore = feature_lower.replace('_', '')
    for k, v in guidelines.items():
        if k.replace('_', '') == feature_no_underscore:
            return v
    
    return np.nan


def is_operator_token(feature):
    """Check if feature is an operator pseudo-feature"""
    feature_lower = str(feature).lower().strip()
    return feature_lower in OPERATOR_TOKENS


def get_display_info(feature):
    """Get display name and unit for a feature"""
    feature_lower = str(feature).lower().strip()
    if feature_lower in FEATURE_DISPLAY:
        return FEATURE_DISPLAY[feature_lower]
    # Clean up feature name
    display_name = feature.replace('_', ' ').title()
    return (display_name, '')


def process_dataset(dataset_dir, dataset_name=None):
    """Process a single dataset directory"""
    dataset_dir = Path(dataset_dir)
    
    if dataset_name is None:
        dataset_name = dataset_dir.name.replace('overall_', '')
    
    # Read thresholds_units.csv
    thresholds_path = dataset_dir / 'aggregated' / 'thresholds_units.csv'
    if not thresholds_path.exists():
        print(f"[WARN] No thresholds_units.csv in {dataset_dir}")
        return pd.DataFrame()
    
    df = pd.read_csv(thresholds_path)
    
    # Load guidelines from multiple sources
    guidelines = {}
    
    # Try YAML guidelines
    guidelines_yaml = dataset_dir / 'config' / 'guidelines.yaml'
    if guidelines_yaml.exists():
        guidelines.update(load_guidelines(guidelines_yaml))
        print(f"  Loaded {len(guidelines)} guidelines from guidelines.yaml")
    
    # Try JSON ground truth
    ground_truth_json = dataset_dir / 'config' / 'ground_truth.json'
    if ground_truth_json.exists():
        gt = load_ground_truth(ground_truth_json)
        guidelines.update(gt)
        print(f"  Loaded {len(gt)} anchors from ground_truth.json")
    
    # Filter to lgo_hard experiment
    if 'experiment' in df.columns:
        df = df[df['experiment'].isin(['lgo_hard', 'LGOhard', 'hard'])].copy()
    
    # Filter operator tokens
    if 'feature' in df.columns:
        df = df[~df['feature'].apply(is_operator_token)].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Process each row
    results = []
    for _, row in df.iterrows():
        feature = row.get('feature', '')
        
        # Get median threshold (natural units)
        median = row.get('b_raw_median', np.nan)
        q1 = row.get('b_raw_q1', np.nan)
        q3 = row.get('b_raw_q3', np.nan)
        
        # Skip if no valid threshold
        if pd.isna(median):
            continue
        
        # Lookup guideline
        guideline = lookup_guideline(feature, guidelines)
        
        # Compute relative error
        if pd.notna(guideline) and guideline != 0:
            rel_error = abs(median - guideline) / abs(guideline)
        else:
            rel_error = np.nan
        
        # Determine error category
        if pd.isna(rel_error):
            error_cat = 3  # N/A
        elif rel_error <= 0.10:
            error_cat = 0  # Green
        elif rel_error <= 0.20:
            error_cat = 1  # Yellow
        else:
            error_cat = 2  # Red
        
        # Get display info
        display_name, default_unit = get_display_info(feature)
        unit = row.get('unit', default_unit) or default_unit
        
        results.append({
            'dataset': dataset_name,
            'feature': feature,
            'feature_key': feature.lower(),
            'display_name': display_name,
            'unit': unit,
            'median': median,
            'q1': q1,
            'q3': q3,
            'guideline': guideline,
            'rel_error': rel_error,
            'error_cat': error_cat,
            'has_iqr': pd.notna(q1) and pd.notna(q3) and q1 != q3,
            'count': row.get('count', 0),
            'mu': row.get('mu', np.nan),
            'sigma': row.get('sigma', np.nan),
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate threshold data from all datasets'
    )
    parser.add_argument('--dataset_dirs', nargs='+', required=True,
                       help='Dataset directories (e.g., overall_ICU_*, overall_eICU_*)')
    parser.add_argument('--outdir', default='aggregated_analysis',
                       help='Output directory')
    parser.add_argument('--only_anchored', action='store_true',
                       help='Only include features with guidelines')
    parser.add_argument('--min_count', type=int, default=1,
                       help='Minimum count for inclusion')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("="*60)
    print("Aggregating threshold data from all datasets")
    print("="*60)
    
    all_results = []
    
    for dataset_dir in args.dataset_dirs:
        if not os.path.isdir(dataset_dir):
            print(f"[WARN] Not a directory: {dataset_dir}")
            continue
        
        print(f"\nProcessing: {dataset_dir}")
        df = process_dataset(dataset_dir)
        
        if not df.empty:
            print(f"  Found {len(df)} valid features")
            all_results.append(df)
    
    if not all_results:
        print("\n[ERROR] No data found")
        return 1
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    
    # Apply filters
    if args.only_anchored:
        combined = combined[combined['guideline'].notna()]
        print(f"\nFiltered to {len(combined)} features with guidelines")
    
    if args.min_count > 1:
        combined = combined[combined['count'] >= args.min_count]
        print(f"Filtered to {len(combined)} features with count >= {args.min_count}")
    
    # Save combined results
    output_path = os.path.join(args.outdir, 'all_thresholds_summary.csv')
    combined.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for dataset in combined['dataset'].unique():
        ds_data = combined[combined['dataset'] == dataset]
        n_total = len(ds_data)
        n_anchored = ds_data['guideline'].notna().sum()
        
        if n_anchored > 0:
            anchored = ds_data[ds_data['guideline'].notna()]
            n_green = (anchored['error_cat'] == 0).sum()
            n_yellow = (anchored['error_cat'] == 1).sum()
            n_red = (anchored['error_cat'] == 2).sum()
            
            print(f"\n{dataset}:")
            print(f"  Features: {n_total} total, {n_anchored} with guidelines")
            print(f"  Agreement: {n_green} green (≤10%), {n_yellow} yellow (≤20%), {n_red} red (>20%)")
            
            success_rate = (n_green + n_yellow) / n_anchored * 100
            print(f"  Success rate (≤20%): {success_rate:.1f}%")
    
    # Overall statistics
    total_anchored = combined['guideline'].notna().sum()
    if total_anchored > 0:
        anchored = combined[combined['guideline'].notna()]
        n_green = (anchored['error_cat'] == 0).sum()
        n_yellow = (anchored['error_cat'] == 1).sum()
        n_red = (anchored['error_cat'] == 2).sum()
        
        print(f"\n{'='*60}")
        print("OVERALL")
        print(f"{'='*60}")
        print(f"Total features with guidelines: {total_anchored}")
        print(f"Green (≤10%): {n_green} ({n_green/total_anchored*100:.1f}%)")
        print(f"Yellow (≤20%): {n_yellow} ({n_yellow/total_anchored*100:.1f}%)")
        print(f"Red (>20%): {n_red} ({n_red/total_anchored*100:.1f}%)")
        print(f"Overall success (≤20%): {(n_green+n_yellow)/total_anchored*100:.1f}%")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())