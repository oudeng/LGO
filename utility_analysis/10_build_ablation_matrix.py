#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_build_ablation_matrix.py (v2)
Updated for new experiment naming:
- base → base (unchanged)
- lgo → lgo_soft
- lgo_thre → lgo_hard
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


NAME = "10_build_ablation_matrix"


def _pick_primary_metric(df: pd.DataFrame) -> str:
    """Fixed to look at actual metric column values, not just column names"""
    # Get actual metric columns from the data
    if "metric" in df.columns:
        # Data is in long format with metric column
        available_metrics = df["metric"].unique()
        # Priority order
        for preferred in ["AUROC", "R2", "RMSE", "MAE", "Brier"]:
            if preferred in available_metrics:
                return preferred
        # Return first available metric
        if len(available_metrics) > 0:
            return available_metrics[0]
    
    # Data might be in wide format
    cols = [c.lower() for c in df.columns]
    if "auroc" in cols:
        return "AUROC"
    if "r2" in cols:
        return "R2"
    if "rmse" in cols:
        return "RMSE"
    if "mae" in cols:
        return "MAE"
    if "brier" in cols:
        return "Brier"
    
    # Default fallback
    return "RMSE"


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
        raise FileNotFoundError(f"[{NAME}] 未找到 {p}（请先汇总整体指标）")
    df = pd.read_csv(p)

    # 标准化列名大小写
    rename = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename)

    if "experiment" not in df.columns:
        df["experiment"] = "unknown"
    dataset_name = args.dataset or ddir.name.replace("overall_", "")
    df["dataset"] = dataset_name

    # Get primary metric
    primary = _pick_primary_metric(df)
    print(f"[{NAME}] Selected primary metric: {primary}")

    # Data is in long format with metric column
    if "metric" in df.columns:
        # Filter to only the primary metric
        df_primary = df[df["metric"] == primary].copy()
        
        # Now we need the actual value column
        value_col = "value" if "value" in df_primary.columns else None
        if value_col is None:
            # Try to find value column
            for col in df_primary.columns:
                if col not in ["dataset", "task", "method", "experiment", "seed", "split", "metric"]:
                    value_col = col
                    break
        
        if value_col is None:
            raise ValueError(f"[{NAME}] Cannot find value column in metrics data")
    else:
        # Wide format - metric is a column
        df_primary = df.copy()
        value_col = primary

    # Filter to only LGO method
    dfl = df_primary[df_primary["method"].astype(str).str.lower().eq("lgo")].copy()
    
    # Normalize experiment names - Updated for v2 naming
    def _exp_norm(s: str) -> str:
        s = str(s).strip().lower()
        # Map old names to new names
        if s in ["lgo_thre", "lgo-thre", "hard", "thre"]:
            return "lgo_hard"
        if s in ["lgo_hard", "lgo-hard"]:  # Already new name
            return "lgo_hard"
        if s in ["lgo", "soft"]:
            return "lgo_soft"
        if s in ["lgo_soft", "lgo-soft"]:  # Already new name
            return "lgo_soft"
        if s in ["base", "none"]:
            return "base"
        return s or "unknown"

    dfl["experiment_norm"] = dfl["experiment"].map(_exp_norm)

    # Calculate statistics for each experiment - using new names internally
    rows = []
    for exp_internal, exp_display in [("base", "base"), ("lgo_soft", "lgo_soft"), ("lgo_hard", "lgo_hard")]:
        sub = dfl[dfl["experiment_norm"] == exp_internal]
        if sub.empty or value_col not in sub.columns:
            rows.append((exp_display, np.nan, np.nan, 0))
        else:
            vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
            med = float(vals.median()) if len(vals) else np.nan
            iqr = _iqr(vals) if len(vals) else np.nan
            n = int(len(vals))
            rows.append((exp_display, med, iqr, n))
            print(f"[{NAME}] {exp_display}: n={n}, median={med:.4f}, IQR={iqr:.4f}" if n > 0 else f"[{NAME}] {exp_display}: n=0")

    # Create output table - Updated column names to match new experiment names
    res = pd.DataFrame({
        "dataset": [dataset_name],
        "primary_metric": [primary],
        "base_median": [rows[0][1]],
        "lgo_soft_median": [rows[1][1]],
        "lgo_hard_median": [rows[2][1]],  # Changed from lgo_thre_median
        "base_IQR": [rows[0][2]],
        "lgo_soft_IQR": [rows[1][2]],
        "lgo_hard_IQR": [rows[2][2]],     # Changed from lgo_thre_IQR
        "n_base": [rows[0][3]],
        "n_soft": [rows[1][3]],
        "n_hard": [rows[2][3]],            # Changed from n_thre
    })
    
    res.to_csv(agg / "ablation_table.csv", index=False)
    print(f"[{NAME}] write out: {agg/'ablation_table.csv'} （primary={primary}）")


if __name__ == "__main__":
    main()