#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_convert_ground_truth_to_guidelines.py 
Convert ground_truth.json to guidelines.yaml format with comprehensive medical thresholds
Supports incremental updates for multiple datasets

Features:
- Converts ground_truth.json to standardized guidelines.yaml format
- Selects conservative thresholds when multiple values exist
- Adds comprehensive medical guidelines based on dataset type
- Creates ground_truth_snapshot.json for reproducibility and audit trail
"""
import argparse
import json
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

def load_ground_truth(path):
    """Load ground_truth.json file"""
    path = Path(path)
    if not path.exists():
        print(f"[WARN] ground_truth.json not found at: {path}")
        return None
    
    with open(path, 'r') as f:
        return json.load(f)

def save_ground_truth_snapshot(source_path, output_dir, verbose=False):
    """
    Save a snapshot of the source ground_truth.json to the output config directory.
    This ensures reproducibility by preserving the exact source data used for conversion.
    
    Args:
        source_path: Path to the original ground_truth.json
        output_dir: Directory where guidelines.yaml is saved (snapshot goes here too)
        verbose: Whether to print detailed info
    
    Returns:
        Path to the snapshot file, or None if failed
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    
    if not source_path.exists():
        print(f"[WARN] Cannot create snapshot: source file not found at {source_path}")
        return None
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create snapshot filename
    snapshot_path = output_dir / "ground_truth_snapshot.json"
    
    # Load source data and add metadata
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # Create snapshot with metadata wrapper
    snapshot_data = {
        "_metadata": {
            "description": "Snapshot of ground_truth.json used for threshold audit",
            "source_file": str(source_path.resolve()),
            "snapshot_created": datetime.now().isoformat(),
            "note": "This file preserves the exact ground_truth used for guidelines generation"
        },
        "ground_truth": source_data
    }
    
    # Write snapshot with pretty formatting
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"[INFO] Snapshot metadata:")
        print(f"  Source: {source_path.resolve()}")
        print(f"  Created: {snapshot_data['_metadata']['snapshot_created']}")
    
    return snapshot_path

def load_existing_guidelines(path):
    """Load existing guidelines.yaml if it exists with data validation"""
    path = Path(path)
    if not path.exists():
        return {"global": {}, "datasets": {}}
    
    try:
        with open(path, 'r') as f:
            content = f.read()
            data = yaml.safe_load(content)
            
            if data is None:
                data = {"global": {}, "datasets": {}}
            
            if "datasets" not in data:
                data["datasets"] = {}
            
            if "global" not in data:
                data["global"] = {}
            
            # Validate and fix dataset entries
            datasets_to_fix = []
            for dataset_name, dataset_value in data.get("datasets", {}).items():
                # Check if dataset value is None or not a dict
                if dataset_value is None or not isinstance(dataset_value, dict):
                    print(f"[WARN] Dataset '{dataset_name}' has invalid format, will be fixed")
                    datasets_to_fix.append(dataset_name)
                # Check if it has the expected structure
                elif not isinstance(dataset_value.get("from_ground_truth"), (dict, type(None))) or \
                     not isinstance(dataset_value.get("additional"), (dict, type(None))):
                    # Old format - convert to new format
                    print(f"[INFO] Converting dataset '{dataset_name}' from old to new format")
                    datasets_to_fix.append(dataset_name)
            
            # Fix problematic datasets
            for dataset_name in datasets_to_fix:
                old_value = data["datasets"].get(dataset_name)
                if isinstance(old_value, dict) and "from_ground_truth" not in old_value:
                    # Old format - all values were mixed together
                    data["datasets"][dataset_name] = {
                        "from_ground_truth": old_value,
                        "additional": {}
                    }
                else:
                    # Invalid format - reset
                    data["datasets"][dataset_name] = {
                        "from_ground_truth": {},
                        "additional": {}
                    }
            
            return data
    except Exception as e:
        print(f"[WARN] Could not parse existing guidelines.yaml: {e}")
        print("[INFO] Starting with empty guidelines")
        return {"global": {}, "datasets": {}}

def convert_ground_truth_values(ground_truth):
    """
    Convert ground_truth.json format to guidelines values
    Selects conservative thresholds when multiple values exist
    """
    guidelines = {}
    
    for feature, data in ground_truth.items():
        values = data.get("values", [])
        if not values:
            continue
        
        # Determine which threshold to use
        if "_min" in feature:
            # For minimum values, use higher threshold (more conservative)
            guideline_value = max(values)
        elif "_max" in feature:
            # For maximum values, use lower threshold (more conservative)
            guideline_value = min(values)
        else:
            # For general features, use the first/lower threshold
            guideline_value = min(values)
        
        guidelines[feature] = float(guideline_value)
    
    return guidelines

def get_additional_guidelines(dataset_name, existing_features):
    """
    Add comprehensive medical/technical guidelines not in ground_truth
    Based on dataset type, clinical standards, and medical literature
    """
    additional = {}
    dataset_upper = dataset_name.upper()
    
    # ICU/MIMIC-IV Critical Care Thresholds
    icu_thresholds = {
        # Vital signs and hemodynamics
        "map_mmhg": 65.0,              # Mean arterial pressure (shock threshold)
        "sbp_min": 90.0,               # Systolic BP (hypotension)
        "dbp_min": 60.0,               # Diastolic BP
        "hr_max": 100.0,               # Tachycardia threshold
        "resprate_max": 20.0,          # Tachypnea threshold
        "spo2_min": 92.0,              # Hypoxemia threshold
        
        # Laboratory values
        "lactate_mmol_l": 2.0,         # Lactate elevation (sepsis/shock)
        "creatinine_mg_dl": 1.2,       # Acute kidney injury stage 1
        "hemoglobin_min": 7.0,         # Transfusion threshold (restrictive)
        "sodium_min": 135.0,           # Hyponatremia
        "sodium_max": 145.0,           # Hypernatremia
        
        # Neurological
        "gcs": 8.0,                    # Severe impairment (intubation threshold)
        
        # Urine output
        "urine_output_min": 0.5,       # Oliguria (ml/kg/hr)
        
        # Medications
        "vasopressor_dose": 0.1,       # Any vasopressor use
        
        # Scores and indices
        "charlson_index": 3.0,         # Moderate comorbidity burden
        
        # Blood gas parameters
        "ph_min": 7.35,                # Acidosis
        "ph_max": 7.45,                # Alkalosis
        "pco2_min": 35.0,              # Hypocapnia
        "pco2_max": 45.0,              # Hypercapnia
        "po2_min": 60.0,               # Hypoxemia (mmHg)
        "hco3_min": 22.0,              # Metabolic acidosis
        "hco3_max": 28.0,              # Metabolic alkalosis
        
        # Additional lab values
        "wbc_min": 4.0,                # Leukopenia
        "wbc_max": 11.0,               # Leukocytosis
        "platelet_min": 150.0,         # Thrombocytopenia
        "inr_max": 1.5,                # Coagulopathy
        "bilirubin_max": 1.2,          # Hyperbilirubinemia
        "albumin_min": 3.5,            # Hypoalbuminemia
        "glucose_min": 70.0,           # Hypoglycemia
        "glucose_max": 180.0,          # Hyperglycemia (ICU target)
        
        # Temperature
        "temperature_min": 36.0,       # Hypothermia
        "temperature_max": 38.3,       # Fever
        
        # Ventilation parameters
        "peep_min": 5.0,               # Minimum PEEP
        "fio2_max": 0.6,               # High oxygen requirement
        "tidal_volume_max": 8.0,       # Lung protective (ml/kg IBW)
    }
    
    # NHANES Metabolic Syndrome Criteria (ATP III/WHO)
    nhanes_thresholds = {
        # Metabolic syndrome components
        "waist_circumference": 102.0,  # Men (cm); Women: 88
        "systolic_bp": 130.0,          # Pre-hypertension/Stage 1
        "diastolic_bp": 85.0,          # Metabolic syndrome threshold
        "triglycerides": 150.0,        # Elevated (mg/dL)
        "hdl_cholesterol": 40.0,       # Low HDL (men); Women: 50
        "fasting_glucose": 100.0,      # Impaired fasting glucose
        
        # Additional metabolic markers
        "bmi": 25.0,                   # Overweight threshold
        "hba1c": 5.7,                  # Prediabetes (%)
        "total_cholesterol": 200.0,    # Borderline high
        "ldl_cholesterol": 100.0,      # Optimal/near optimal
        
        # Blood pressure stages
        "bp_systolic_stage1": 130.0,   # Stage 1 hypertension
        "bp_systolic_stage2": 140.0,   # Stage 2 hypertension
        "bp_diastolic_stage1": 80.0,   # Stage 1 hypertension
        "bp_diastolic_stage2": 90.0,   # Stage 2 hypertension
        
        # Diabetes thresholds
        "glucose_diabetes": 126.0,     # Diabetes diagnosis
        "hba1c_diabetes": 6.5,         # Diabetes diagnosis
        
        # Lipid targets
        "non_hdl_cholesterol": 130.0,  # Target for high risk
        "apolipoprotein_b": 90.0,      # High risk target
        
        # Kidney function
        "egfr_min": 60.0,              # CKD stage 3
        "uacr_max": 30.0,              # Microalbuminuria
        
        # Liver function
        "alt_max": 40.0,               # Upper normal limit
        "ast_max": 40.0,               # Upper normal limit
    }
    
    # CTG (Cardiotocography) fetal monitoring thresholds
    ctg_thresholds = {
        # Fetal heart rate
        "fhr_baseline_min": 110.0,     # Bradycardia threshold
        "fhr_baseline_max": 160.0,     # Tachycardia threshold
        "fhr_variability_min": 5.0,    # Minimal variability
        "fhr_variability_max": 25.0,   # Marked variability
        
        # Decelerations and accelerations
        "accelerations_min": 2.0,      # Reactive pattern
        "decelerations_max": 0.0,      # Late/variable decels
        
        # Uterine contractions
        "contractions_max": 5.0,       # Per 10 minutes
    }
    
    # Cleveland Heart Disease thresholds
    cleveland_thresholds = {
        # Risk factors
        "age": 45.0,                   # Increased risk threshold
        "trestbps": 140.0,             # Resting BP (hypertension)
        "chol": 200.0,                 # Total cholesterol
        "thalach": 150.0,              # Max heart rate achieved
        "oldpeak": 2.0,                # ST depression
        
        # Binary/categorical (for reference)
        "fbs": 120.0,                  # Fasting blood sugar threshold
        "ca": 1.0,                     # Number of major vessels
    }
    
    # Hydraulic system thresholds (engineering)
    hydraulic_thresholds = {
        # Pressure sensors
        "ps1": 100.0,                  # Pressure sensor 1
        "ps2": 100.0,                  # Pressure sensor 2
        "ps3": 100.0,                  # Pressure sensor 3
        "ps4": 100.0,                  # Pressure sensor 4
        "ps5": 100.0,                  # Pressure sensor 5
        "ps6": 100.0,                  # Pressure sensor 6
        
        # Flow and efficiency
        "eps1": 2000.0,                # Motor power
        "fs1": 10.0,                   # Volume flow
        "fs2": 10.0,                   # Volume flow
        
        # Temperature
        "ts1": 40.0,                   # Oil temperature
        "ts2": 40.0,                   # Oil temperature
        "ts3": 40.0,                   # Oil temperature
        "ts4": 40.0,                   # Oil temperature
        
        # Vibration
        "vs1": 0.5,                    # Vibration sensor
        
        # Efficiency
        "se": 90.0,                    # Efficiency factor
        "ce": 90.0,                    # Cooling efficiency
        "cp": 50.0,                    # Cooling power
    }
    
    # Select defaults based on dataset type
    if "ICU" in dataset_upper or "MIMIC" in dataset_upper:
        defaults = icu_thresholds
    elif "NHANES" in dataset_upper or "METABOLIC" in dataset_upper:
        defaults = nhanes_thresholds
    elif "CTG" in dataset_upper or "CARDIOTOCOGRAPHY" in dataset_upper:
        defaults = ctg_thresholds
    elif "CLEVELAND" in dataset_upper or "HEART" in dataset_upper:
        defaults = cleveland_thresholds
    elif "HYDRAULIC" in dataset_upper:
        defaults = hydraulic_thresholds
    else:
        # Generic defaults
        defaults = {
            "age": 50.0,
            "bmi": 25.0,
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "heart_rate": 70.0,
            "glucose": 100.0,
            "cholesterol": 200.0,
        }
    
    # Add only features not already present
    for key, value in defaults.items():
        if key not in existing_features:
            additional[key] = value
    
    return additional

def write_guidelines_yaml(guidelines_data, output_path):
    """
    Write guidelines.yaml with proper formatting and comments
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build YAML content with comments
    lines = []
    lines.append("# Medical Guidelines Reference Values")
    lines.append("# Auto-generated and maintained by 08_convert_ground_truth_to_guidelines.py")
    lines.append("# Enhanced with comprehensive medical thresholds from clinical guidelines")
    lines.append("")
    
    # Global section
    lines.append("global:")
    if guidelines_data.get("global"):
        for key, value in sorted(guidelines_data["global"].items()):
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  # Add global guidelines here")
    lines.append("")
    
    # Datasets section
    lines.append("datasets:")
    for dataset_name, dataset_values in guidelines_data.get("datasets", {}).items():
        lines.append(f"  {dataset_name}:")
        
        # Handle None or invalid dataset_values
        if dataset_values is None:
            print(f"[WARN] Dataset '{dataset_name}' has None value, skipping")
            lines.append("    # No guidelines defined")
            lines.append("")
            continue
        
        if not isinstance(dataset_values, dict):
            print(f"[WARN] Dataset '{dataset_name}' has invalid type: {type(dataset_values)}, skipping")
            lines.append("    # Invalid data format")
            lines.append("")
            continue
        
        # Separate ground_truth values from additional ones
        from_ground_truth = dataset_values.get("from_ground_truth", {})
        additional = dataset_values.get("additional", {})
        
        # Ensure they are dicts
        if not isinstance(from_ground_truth, dict):
            from_ground_truth = {}
        if not isinstance(additional, dict):
            additional = {}
        
        # Write ground_truth values first
        if from_ground_truth:
            lines.append("    # From ground_truth.json")
            for key, value in sorted(from_ground_truth.items()):
                lines.append(f"    {key}: {value}  # from ground_truth.json")
        
        # Write additional values
        if additional:
            if from_ground_truth:
                lines.append("    # Additional medical guidelines")
            for key, value in sorted(additional.items()):
                lines.append(f"    {key}: {value}  # additional medical guidelines")
        
        if not from_ground_truth and not additional:
            lines.append("    # No guidelines defined")
        
        lines.append("")  # Blank line between datasets
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

def main():
    parser = argparse.ArgumentParser(
        description="Convert ground_truth.json to guidelines.yaml with comprehensive medical thresholds"
    )
    parser.add_argument(
        "--ground_truth", 
        required=True, 
        help="Path to ground_truth.json file"
    )
    parser.add_argument(
        "--dataset", 
        required=True,
        help="Dataset name (e.g., ICU_composite_risk_score, NHANES_metabolic_score, CTG_nsp, UCI_CTG_NSPbin)"
    )
    parser.add_argument(
        "--output", 
        default="config/guidelines.yaml",
        help="Output guidelines.yaml path (default: config/guidelines.yaml)"
    )
    parser.add_argument(
        "--no_additional",
        action="store_true",
        help="Do not add additional medical guidelines"
    )
    parser.add_argument(
        "--no_snapshot",
        action="store_true",
        help="Do not create ground_truth_snapshot.json"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about added thresholds"
    )
    args = parser.parse_args()
    
    # Load ground_truth
    ground_truth = load_ground_truth(args.ground_truth)
    if ground_truth is None:
        print(f"[ERROR] Cannot proceed without ground_truth.json")
        print(f"Please check the file exists at: {args.ground_truth}")
        return 1
    
    # Convert ground_truth values
    gt_guidelines = convert_ground_truth_values(ground_truth)
    print(f"[INFO] Extracted {len(gt_guidelines)} guidelines from ground_truth.json:")
    if args.verbose:
        for k, v in sorted(gt_guidelines.items()):
            print(f"  {k}: {v}")
    
    # Load existing guidelines
    existing = load_existing_guidelines(args.output)
    
    # Prepare dataset entry
    dataset_entry = {
        "from_ground_truth": gt_guidelines,
        "additional": {}
    }
    
    # Add additional guidelines if not disabled
    if not args.no_additional:
        additional = get_additional_guidelines(args.dataset, gt_guidelines.keys())
        if additional:
            dataset_entry["additional"] = additional
            print(f"[INFO] Added {len(additional)} additional medical guidelines")
            if args.verbose:
                print("[INFO] Additional guidelines added:")
                for k, v in sorted(additional.items()):
                    print(f"  {k}: {v}")
    
    # Update or add dataset entry
    if args.dataset in existing.get("datasets", {}):
        print(f"[INFO] Updating existing entry for dataset: {args.dataset}")
    else:
        print(f"[INFO] Adding new entry for dataset: {args.dataset}")
    
    existing["datasets"][args.dataset] = dataset_entry
    
    # Set comprehensive global defaults if empty
    if not existing.get("global"):
        existing["global"] = {
            # Vital signs
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "heart_rate": 70.0,
            "respiratory_rate": 16.0,
            "temperature": 37.0,
            "spo2": 95.0,
            
            # Basic labs
            "glucose": 100.0,
            "hemoglobin": 12.0,
            "wbc": 7.5,
            "platelet": 250.0,
            "creatinine": 1.0,
            "bun": 15.0,
            
            # Electrolytes
            "sodium": 140.0,
            "potassium": 4.0,
            "chloride": 100.0,
            "bicarbonate": 24.0,
            
            # Lipids
            "cholesterol": 200.0,
            "triglycerides": 150.0,
            "hdl": 40.0,
            "ldl": 100.0,
            
            # Other
            "lactate": 2.0,
            "age": 50.0,
            "bmi": 25.0,
        }
    
    # Write updated guidelines
    write_guidelines_yaml(existing, args.output)
    print(f"[OK] Written guidelines to: {args.output}")
    
    # Create ground_truth snapshot for reproducibility
    if not args.no_snapshot:
        output_dir = Path(args.output).parent
        snapshot_path = save_ground_truth_snapshot(
            args.ground_truth, 
            output_dir, 
            verbose=args.verbose
        )
        if snapshot_path:
            print(f"[OK] Created snapshot: {snapshot_path}")
    
    # Summary
    total_datasets = len(existing.get("datasets", {}))
    total_global = len(existing.get("global", {}))
    print(f"[INFO] Total datasets in guidelines: {total_datasets}")
    print(f"[INFO] Total global guidelines: {total_global}")
    
    # Dataset-specific summary
    if args.dataset in existing["datasets"]:
        ds = existing["datasets"][args.dataset]
        total_gt = len(ds.get("from_ground_truth", {}))
        total_add = len(ds.get("additional", {}))
        print(f"[INFO] Dataset '{args.dataset}': {total_gt} from ground_truth, {total_add} additional")
    
    return 0

if __name__ == "__main__":
    exit(main())