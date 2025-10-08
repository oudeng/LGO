#!/usr/bin/env python
"""
Diagnostic script to check threshold data availability and structure
"""
import os
import sys
import pandas as pd
import numpy as np

def check_csv(path, name):
    """Check a CSV file and report its structure"""
    if not os.path.exists(path):
        print(f"  ❌ {name}: Not found at {path}")
        return None
    
    try:
        df = pd.read_csv(path)
        print(f"  ✓ {name}: {len(df)} rows, {len(df.columns)} columns")
        print(f"    Columns: {', '.join(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"    ... and {len(df.columns)-10} more columns")
        
        # Check for key columns
        key_cols = ['dataset', 'method', 'experiment', 'feature', 'feature_norm', 
                   'b_raw', 'b_raw_median', 'threshold', 'anchor', 'rel_err', 
                   'seed', 'model_id', 'gate_type']
        found_keys = [c for c in key_cols if c in df.columns]
        if found_keys:
            print(f"    Key columns found: {', '.join(found_keys)}")
        
        # Check unique values in important columns
        for col in ['dataset', 'method', 'experiment', 'gate_type']:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10:
                    print(f"    {col} values: {', '.join(map(str, unique_vals))}")
                else:
                    print(f"    {col}: {len(unique_vals)} unique values")
        
        # Check for anchored features
        if 'anchor' in df.columns:
            n_anchored = df['anchor'].notna().sum()
            print(f"    Anchored features: {n_anchored}/{len(df)} rows")
        
        return df
    except Exception as e:
        print(f"  ❌ {name}: Error reading - {e}")
        return None

def check_directory(dir_path, dataset_name):
    """Check a dataset directory for expected files"""
    print(f"\n📁 Checking {dataset_name} ({dir_path})")
    
    if not os.path.exists(dir_path):
        print(f"  ❌ Directory not found")
        return
    
    # Check for aggregated subdirectory
    agg_dir = os.path.join(dir_path, "aggregated")
    if os.path.exists(agg_dir):
        print(f"  ✓ Found aggregated/ subdirectory")
        
        # Check for expected CSV files
        files_to_check = [
            ("thresholds_units.csv", os.path.join(agg_dir, "thresholds_units.csv")),
            ("threshold_audit.csv", os.path.join(agg_dir, "threshold_audit.csv")),
            ("threshold_audit_summary.csv", os.path.join(agg_dir, "threshold_audit_summary.csv"))
        ]
        
        dfs = {}
        for name, path in files_to_check:
            df = check_csv(path, name)
            if df is not None:
                dfs[name] = df
        
        # Check data compatibility
        if dfs:
            print(f"\n  📊 Data Analysis for {dataset_name}:")
            
            # Check if method='lgu' and experiment='lgu_hard' exist
            for name, df in dfs.items():
                if 'method' in df.columns and 'experiment' in df.columns:
                    lgu_hard = df[(df['method']=='lgu') & (df['experiment']=='lgu_hard')]
                    if len(lgu_hard) > 0:
                        print(f"    ✓ {name}: Found {len(lgu_hard)} rows with method='lgu' and experiment='lgu_hard'")
                    else:
                        print(f"    ⚠️  {name}: No rows with method='lgu' and experiment='lgu_hard'")
                        # Show what methods/experiments are available
                        if 'method' in df.columns:
                            methods = df['method'].dropna().unique()
                            print(f"       Available methods: {', '.join(map(str, methods[:5]))}")
                        if 'experiment' in df.columns:
                            experiments = df['experiment'].dropna().unique()
                            print(f"       Available experiments: {', '.join(map(str, experiments[:5]))}")
            
            # Check coverage calculation feasibility
            if 'threshold_audit.csv' in dfs:
                audit = dfs['threshold_audit.csv']
                if 'seed' in audit.columns:
                    n_seeds = len(audit['seed'].dropna().unique())
                    print(f"    Seeds in audit: {n_seeds}")
                if 'model_id' in audit.columns:
                    n_models = len(audit['model_id'].dropna().unique())
                    print(f"    Models in audit: {n_models}")
    else:
        print(f"  ⚠️  No aggregated/ subdirectory found")
        # List what's in the directory
        contents = os.listdir(dir_path)[:10]
        print(f"  Directory contains: {', '.join(contents)}")
        if len(os.listdir(dir_path)) > 10:
            print(f"  ... and {len(os.listdir(dir_path))-10} more items")

def check_config(config_dir):
    """Check configuration directory"""
    print(f"\n⚙️  Checking config directory: {config_dir}")
    
    if not os.path.exists(config_dir):
        print(f"  ❌ Directory not found")
        return
    
    guidelines_path = os.path.join(config_dir, "guidelines.yaml")
    if os.path.exists(guidelines_path):
        print(f"  ✓ Found guidelines.yaml")
        try:
            import yaml
            with open(guidelines_path, 'r') as f:
                data = yaml.safe_load(f)
            if 'global' in data:
                print(f"    Global anchors: {len(data.get('global', {}))} features")
            if 'datasets' in data:
                datasets = data.get('datasets', {})
                for ds_name, ds_data in datasets.items():
                    if isinstance(ds_data, dict) and ds_data:
                        print(f"    Dataset '{ds_name}': {len(ds_data)} anchors")
        except Exception as e:
            print(f"  ⚠️  Could not parse guidelines.yaml: {e}")
    else:
        print(f"  ❌ guidelines.yaml not found")

def main():
    print("="*60)
    print("THRESHOLD DATA DIAGNOSTIC REPORT")
    print("="*60)
    
    # Dataset directories to check
    dataset_dirs = [
        ("overall_ICU_composite_risk_score", "ICU_composite_risk_score"),
        ("overall_NHANES_metabolic_score", "NHANES_metabolic_score"),
        ("overall_UCI_HydraulicSys_fault_score", "UCI_HydraulicSys_fault_score"),
        ("overall_UCI_Heart_Cleveland_num", "UCI_Heart_Cleveland_num"),
        ("overall_UCI_CTG_NSPbin", "UCI_CTG_NSPbin")
    ]
    
    # Check each dataset directory
    for dir_name, dataset_name in dataset_dirs:
        check_directory(dir_name, dataset_name)
    
    # Check config directory
    check_config("config")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("""
1. If 'No rows with method='lgu' and experiment='lgu_hard'':
   - Check what methods/experiments are available in your data
   - Adjust --method and --experiment parameters accordingly
   
2. If 'No aggregated/ subdirectory':
   - Ensure data is in <dataset_dir>/aggregated/*.csv format
   - Or use --thresholds_units_csv and --threshold_audit_csv to specify paths
   
3. If 'Anchored features: 0':
   - Remove --only_anchored flag, or
   - Ensure guidelines.yaml contains anchors for your features
   
4. If coverage is low:
   - Reduce --min_coverage threshold (currently 0.30)
   - Check that enough seeds/models have threshold data
    """)

if __name__ == "__main__":
    main()