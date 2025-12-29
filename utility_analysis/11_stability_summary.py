#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_stability_summary.py
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


NAME = "11_stability_summary"

REG_METRICS = ["R2", "RMSE", "MAE"]
CLF_METRICS = ["AUROC", "AUPRC", "Brier", "BRIER"]


def _iqr(x: pd.Series) -> float:
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    agg = ddir / "aggregated"; agg.mkdir(parents=True, exist_ok=True)
    p = agg / "overall_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"[{NAME}] 未找到 {p}")
    df = pd.read_csv(p)

    dataset_name = args.dataset or ddir.name.replace("overall_", "")
    df["dataset"] = dataset_name
    if "experiment" not in df.columns:
        df["experiment"] = "unknown"

    # Check if data is in long format (has 'metric' and 'value' columns)
    if "metric" in df.columns and "value" in df.columns:
        print(f"[{NAME}] Data is in long format, pivoting to wide format...")
        
        # Get unique metrics
        available_metrics = df["metric"].unique()
        print(f"[{NAME}] Available metrics: {available_metrics}")
        
        # Pivot to wide format
        grp_cols = [c for c in ["dataset", "method", "experiment", "seed", "split"] if c in df.columns]
        if len(grp_cols) > 0:
            df_wide = df.pivot_table(
                index=grp_cols,
                columns="metric",
                values="value",
                aggfunc="first"  # Use first if there are duplicates
            ).reset_index()
        else:
            # Fallback if no proper grouping columns
            df_wide = df.pivot_table(
                index=df.index,
                columns="metric",
                values="value"
            ).reset_index()
            # Carry over other columns
            for col in ["dataset", "method", "experiment"]:
                if col in df.columns:
                    df_wide[col] = df[col]
        
        df = df_wide
        
        # Now check which metrics are available
        cols = [c for c in df.columns if c in (REG_METRICS + CLF_METRICS)]
    else:
        # Data is already in wide format
        cols = [c for c in df.columns if c in (REG_METRICS + CLF_METRICS)]
        
    if not cols:
        # Try case-insensitive match
        all_metrics = REG_METRICS + CLF_METRICS
        cols = []
        for col in df.columns:
            for metric in all_metrics:
                if col.upper() == metric.upper():
                    cols.append(col)
                    break
    
    if not cols:
        print(f"[{NAME}] Available columns: {df.columns.tolist()}")
        raise ValueError(f"[{NAME}] 在 overall_metrics.csv 中未发现已知指标列：{REG_METRICS + CLF_METRICS}")

    print(f"[{NAME}] Found metric columns: {cols}")

    grp_cols = [c for c in ["dataset", "method", "experiment"] if c in df.columns]
    outs = []
    
    if len(grp_cols) > 0:
        for keys, sub in df.groupby(grp_cols):
            row = {k: v for k, v in zip(grp_cols, keys)}
            for m in cols:
                vals = pd.to_numeric(sub[m], errors="coerce").dropna()
                row[f"{m}_median"] = float(vals.median()) if len(vals) else np.nan
                row[f"{m}_IQR"]    = float(_iqr(vals))     if len(vals) else np.nan
                row[f"{m}_mean"]   = float(vals.mean())    if len(vals) else np.nan
                row[f"{m}_std"]    = float(vals.std())     if len(vals) else np.nan
                row[f"{m}_n"]      = int(len(vals))
            outs.append(row)
    else:
        # No grouping columns, create single summary
        row = {"dataset": dataset_name}
        for m in cols:
            vals = pd.to_numeric(df[m], errors="coerce").dropna()
            row[f"{m}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{m}_IQR"]    = float(_iqr(vals))     if len(vals) else np.nan
            row[f"{m}_mean"]   = float(vals.mean())    if len(vals) else np.nan
            row[f"{m}_std"]    = float(vals.std())     if len(vals) else np.nan
            row[f"{m}_n"]      = int(len(vals))
        outs.append(row)

    out = pd.DataFrame(outs)
    out.to_csv(agg / "stability_summary.csv", index=False)
    print(f"[{NAME}] 写出: {agg/'stability_summary.csv'} with {len(outs)} rows")


if __name__ == "__main__":
    main()