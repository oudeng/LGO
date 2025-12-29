#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opt_gen_calibration.py
Calibration analysis for binary classification predictions
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

def find_prediction_files(pred_dir, method=None, experiment=None):
    """Find prediction files matching criteria"""
    pred_dir = Path(pred_dir)
    pattern = "test_predictions_*.csv"
    
    files = []
    for f in pred_dir.glob(pattern):
        # Parse filename: test_predictions_{method}_{experiment}_seed{N}.csv
        parts = f.stem.replace("test_predictions_", "").split("_")
        if len(parts) >= 3 and parts[-1].startswith("seed"):
            file_method = "_".join(parts[:-2])
            file_experiment = parts[-2]
            seed = parts[-1].replace("seed", "")
            
            # Filter by method/experiment if specified
            if method and file_method != method:
                continue
            if experiment and file_experiment != experiment:
                continue
                
            files.append({
                "path": f,
                "method": file_method,
                "experiment": file_experiment,
                "seed": seed
            })
    
    return files

def reliability_bins(y_true, y_prob, n_bins=10):
    """Calculate reliability diagram bin statistics"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    results = []
    for i in range(n_bins):
        mask = (bin_indices == i)
        
        if mask.sum() > 0:
            bin_prob = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            bin_count = mask.sum()
            
            results.append({
                'bin': i,
                'bin_lower': bin_edges[i],
                'bin_upper': bin_edges[i+1],
                'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                'predicted_prob': bin_prob,
                'actual_prob': bin_acc,
                'count': bin_count,
                'gap': abs(bin_prob - bin_acc)
            })
        else:
            results.append({
                'bin': i,
                'bin_lower': bin_edges[i],
                'bin_upper': bin_edges[i+1],
                'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                'predicted_prob': np.nan,
                'actual_prob': np.nan,
                'count': 0,
                'gap': np.nan
            })
    
    return pd.DataFrame(results)

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    ece = 0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        bin_size = mask.sum()
        
        if bin_size > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
    
    return ece

def apply_platt_calibration(y_true, y_score_raw, cv_folds=3):
    """Apply Platt scaling calibration"""
    print("[INFO] Applying Platt calibration...")
    
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    scores_2d = y_score_raw.reshape(-1, 1)
    
    if cv_folds > 1 and len(y_true) > cv_folds * 10:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        calibrated_probs = np.zeros_like(y_score_raw)
        
        for train_idx, val_idx in skf.split(scores_2d, y_true):
            fold_calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
            fold_calibrator.fit(scores_2d[train_idx], y_true[train_idx])
            calibrated_probs[val_idx] = fold_calibrator.predict_proba(scores_2d[val_idx])[:, 1]
        
        calibrator.fit(scores_2d, y_true)
    else:
        calibrator.fit(scores_2d, y_true)
        calibrated_probs = calibrator.predict_proba(scores_2d)[:, 1]
    
    return calibrated_probs, calibrator

def apply_isotonic_calibration(y_true, y_prob, cv_folds=3):
    """Apply isotonic regression calibration"""
    print("[INFO] Applying Isotonic calibration...")
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    
    if cv_folds > 1 and len(y_true) > cv_folds * 10:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        calibrated_probs = np.zeros_like(y_prob)
        
        for train_idx, val_idx in skf.split(y_prob, y_true):
            fold_calibrator = IsotonicRegression(out_of_bounds='clip')
            fold_calibrator.fit(y_prob[train_idx], y_true[train_idx])
            calibrated_probs[val_idx] = fold_calibrator.transform(y_prob[val_idx])
        
        calibrator.fit(y_prob, y_true)
    else:
        calibrator.fit(y_prob, y_true)
        calibrated_probs = calibrator.transform(y_prob)
    
    calibrated_probs = np.clip(calibrated_probs, 0, 1)
    return calibrated_probs, calibrator

def main():
    p = argparse.ArgumentParser(
        description="Calibration analysis for binary classification"
    )
    p.add_argument("--dataset_dir", required=True, help="Dataset directory")
    p.add_argument("--dataset", default="CUSTOM", help="Dataset name")
    p.add_argument("--method", default=None, help="Method name filter")
    p.add_argument("--experiment", default=None, help="Experiment name filter")
    p.add_argument("--calibrator", default="none", 
                   choices=['none', 'platt', 'isotonic', 'both'],
                   help="Calibration method")
    p.add_argument("--cv_folds", type=int, default=3,
                   help="Cross-validation folds for calibration")
    p.add_argument("--n_bins", type=int, default=10,
                   help="Number of bins for calibration analysis")
    args = p.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    pred_dir = dataset_dir / "predictions"
    ag = dataset_dir / "aggregated"
    ag.mkdir(parents=True, exist_ok=True)
    
    if not pred_dir.exists():
        print(f"[ERROR] Predictions directory not found: {pred_dir}")
        print("       Run 7j with --save_predictions flag first")
        return 1
    
    # Find prediction files
    pred_files = find_prediction_files(pred_dir, args.method, args.experiment)
    if not pred_files:
        print(f"[WARN] No prediction files found matching criteria")
        print(f"       Method: {args.method}, Experiment: {args.experiment}")
        return 1
    
    print(f"[INFO] Found {len(pred_files)} prediction files")
    
    all_bins = []
    all_ece = []
    
    for file_info in pred_files:
        print(f"\n[INFO] Processing: {file_info['path'].name}")
        df = pd.read_csv(file_info['path'])
        
        # Check for required columns
        if 'y_true' not in df.columns:
            print(f"[WARN] Missing y_true column, skipping")
            continue
            
        # Check if this is a classification task
        if 'y_prob' not in df.columns:
            print(f"[WARN] Not a classification task (no y_prob), skipping")
            continue
        
        y_true = df['y_true'].values
        y_prob = df['y_prob'].values
        
        # Get raw scores if available
        y_score_raw = None
        if 'y_score_raw' in df.columns:
            y_score_raw = df['y_score_raw'].values
        elif args.calibrator == 'platt':
            # Try to reconstruct from probabilities
            eps = 1e-7
            y_prob_clipped = np.clip(y_prob, eps, 1-eps)
            y_score_raw = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Calculate original calibration
        bins_orig = reliability_bins(y_true, y_prob, args.n_bins)
        ece_orig = calculate_ece(y_true, y_prob, args.n_bins)
        
        bins_orig['dataset'] = args.dataset
        bins_orig['method'] = file_info['method']
        bins_orig['experiment'] = file_info['experiment']
        bins_orig['seed'] = file_info['seed']
        bins_orig['calibrator'] = 'none'
        
        all_bins.append(bins_orig)
        all_ece.append({
            'dataset': args.dataset,
            'method': file_info['method'],
            'experiment': file_info['experiment'],
            'seed': file_info['seed'],
            'calibrator': 'none',
            'ECE': ece_orig,
            'n_samples': len(y_true)
        })
        
        print(f"       Original ECE: {ece_orig:.4f}")
        
        # Apply calibration if requested
        if args.calibrator in ['platt', 'both'] and y_score_raw is not None:
            y_prob_platt, _ = apply_platt_calibration(y_true, y_score_raw, args.cv_folds)
            bins_platt = reliability_bins(y_true, y_prob_platt, args.n_bins)
            ece_platt = calculate_ece(y_true, y_prob_platt, args.n_bins)
            
            bins_platt['dataset'] = args.dataset
            bins_platt['method'] = file_info['method']
            bins_platt['experiment'] = file_info['experiment']
            bins_platt['seed'] = file_info['seed']
            bins_platt['calibrator'] = f'platt_cv{args.cv_folds}'
            
            all_bins.append(bins_platt)
            all_ece.append({
                'dataset': args.dataset,
                'method': file_info['method'],
                'experiment': file_info['experiment'],
                'seed': file_info['seed'],
                'calibrator': f'platt_cv{args.cv_folds}',
                'ECE': ece_platt,
                'n_samples': len(y_true)
            })
            
            print(f"       Platt ECE: {ece_platt:.4f} (improvement: {(ece_orig-ece_platt)/ece_orig*100:.1f}%)")
        
        if args.calibrator in ['isotonic', 'both']:
            y_prob_iso, _ = apply_isotonic_calibration(y_true, y_prob, args.cv_folds)
            bins_iso = reliability_bins(y_true, y_prob_iso, args.n_bins)
            ece_iso = calculate_ece(y_true, y_prob_iso, args.n_bins)
            
            bins_iso['dataset'] = args.dataset
            bins_iso['method'] = file_info['method']
            bins_iso['experiment'] = file_info['experiment']
            bins_iso['seed'] = file_info['seed']
            bins_iso['calibrator'] = f'isotonic_cv{args.cv_folds}'
            
            all_bins.append(bins_iso)
            all_ece.append({
                'dataset': args.dataset,
                'method': file_info['method'],
                'experiment': file_info['experiment'],
                'seed': file_info['seed'],
                'calibrator': f'isotonic_cv{args.cv_folds}',
                'ECE': ece_iso,
                'n_samples': len(y_true)
            })
            
            print(f"       Isotonic ECE: {ece_iso:.4f} (improvement: {(ece_orig-ece_iso)/ece_orig*100:.1f}%)")
    
    # Save results
    if all_bins:
        bins_df = pd.concat(all_bins, ignore_index=True)
        bins_path = ag / "calibration_bins.csv"
        
        # Append or create
        if bins_path.exists():
            existing = pd.read_csv(bins_path)
            # Remove duplicates based on key columns
            key_cols = ['dataset', 'method', 'experiment', 'seed', 'calibrator', 'bin']
            combined = pd.concat([existing, bins_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep='last')
            combined.to_csv(bins_path, index=False)
            print(f"\n[OK] Updated calibration_bins.csv")
        else:
            bins_df.to_csv(bins_path, index=False)
            print(f"\n[OK] Created calibration_bins.csv")
    
    if all_ece:
        ece_df = pd.DataFrame(all_ece)
        ece_path = ag / "calibration_ece.csv"
        
        # Append or create
        if ece_path.exists():
            existing = pd.read_csv(ece_path)
            # Remove duplicates
            key_cols = ['dataset', 'method', 'experiment', 'seed', 'calibrator']
            combined = pd.concat([existing, ece_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep='last')
            combined.to_csv(ece_path, index=False)
            print(f"[OK] Updated calibration_ece.csv")
        else:
            ece_df.to_csv(ece_path, index=False)
            print(f"[OK] Created calibration_ece.csv")
        
        # Summary statistics
        print(f"\n=== Calibration Summary ===")
        summary = ece_df.groupby(['method', 'experiment', 'calibrator'])['ECE'].agg(['mean', 'std'])
        print(summary)
    
    return 0

if __name__ == "__main__":
    exit(main())