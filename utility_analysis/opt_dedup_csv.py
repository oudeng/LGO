#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opt_dedup_csv.py
Remove duplicate rows from CSV files based on key columns
Useful for cleaning aggregated results after multiple runs
"""
import argparse
import pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser(
        description="Remove duplicate rows from CSV files based on key columns"
    )
    p.add_argument("--csv", required=True, help="Input CSV file path")
    p.add_argument("--keys", required=True, 
                   help="Comma-separated column names to identify duplicates")
    p.add_argument("--out", default="", 
                   help="Output file path (default: overwrite input)")
    p.add_argument("--keep", default="last", choices=["first", "last"],
                   help="Which duplicate to keep (default: last)")
    p.add_argument("--backup", action="store_true",
                   help="Create backup of original file")
    args = p.parse_args()
    
    # Parse key columns
    keys = [k.strip() for k in args.keys.split(",")]
    
    # Read CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {args.csv}")
    
    df = pd.read_csv(csv_path)
    before = len(df)
    
    # Check if all key columns exist
    missing_keys = [k for k in keys if k not in df.columns]
    if missing_keys:
        raise ValueError(f"Key columns not found in CSV: {missing_keys}")
    
    # Create backup if requested
    if args.backup:
        backup_path = csv_path.with_suffix(".csv.bak")
        df.to_csv(backup_path, index=False)
        print(f"[INFO] Backup saved to: {backup_path}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=keys, keep=args.keep)
    after = len(df)
    
    # Save output
    out_path = Path(args.out) if args.out else csv_path
    df.to_csv(out_path, index=False)
    
    # Report results
    removed = before - after
    if removed > 0:
        print(f"[OK] {args.csv} -> {out_path}")
        print(f"     Removed {removed} duplicate rows using keys={keys} (keep={args.keep})")
        print(f"     Rows: {before} -> {after}")
    else:
        print(f"[OK] No duplicates found in {args.csv}")
        
    # Show sample of removed rows if any
    if removed > 0 and removed <= 10:
        print(f"[INFO] Duplicate key combinations removed:")
        # Find what was removed
        df_original = pd.read_csv(csv_path)
        df_kept_keys = df[keys].drop_duplicates()
        df_all_keys = df_original[keys]
        df_removed = df_all_keys[~df_all_keys.apply(tuple,1).isin(df_kept_keys.apply(tuple,1))]
        for _, row in df_removed.drop_duplicates().head(5).iterrows():
            print(f"       {dict(row)}")

if __name__ == "__main__":
    main()