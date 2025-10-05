#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_gen_gating_usage.py
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

NAME = "06_gen_gating_usage"

def _load_csv_if_exists(p: Path):
    return pd.read_csv(p) if p.exists() else None


def _infer_score(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}
    for key in ["cv_loss", "val_loss", "loss", "rmse", "mae", "mse", "brier", "brier_score", "brier score"]:
        if key in cols:
            return df[cols[key]].astype(float)
    # 需要把"越大越好"的指标转为 score
    for key in ["r2", "auroc", "auprc"]:
        if key in cols:
            return 1.0 - df[cols[key]].astype(float)
    return pd.Series(np.nan, index=df.index)


def _ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["num_gates", "complexity"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["experiment", "method", "dataset"]:
        if col not in df.columns:
            df[col] = "unknown"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Dataset directory (e.g., overall_ICU_composite_risk_score)")
    ap.add_argument("--dataset", default=None, help="Dataset name for output (ignored for paths)")
    ap.add_argument("--topk", type=int, default=100, help="Top-k expressions (default k=100)")
    ap.add_argument("--method", default=None)
    ap.add_argument("--experiments", default=None, help="Selected experiment(s)")
    args = ap.parse_args()

    # CRITICAL: Use dataset_dir for ALL paths, NOT dataset
    ddir = Path(args.dataset_dir)  # This should be overall_ICU_composite_risk_score
    print(f"[{NAME}] Using dataset directory: {ddir}")
    
    if not ddir.exists():
        raise FileNotFoundError(f"[{NAME}] Dataset directory does not exist: {ddir}")
    
    agg = ddir / "aggregated"
    agg.mkdir(parents=True, exist_ok=True)
    
    # Look for files in the CORRECT location
    print(f"[{NAME}] Looking for files in: {agg}")
    
    # Try to load expressions and complexity files
    dfe = _load_csv_if_exists(agg / "expressions.csv")
    dfc = _load_csv_if_exists(agg / "complexity_by_model.csv")
    
    # Try alternative paths
    if dfe is None:
        alt_expr_paths = [
            ddir / "expressions" / "topk_expressions.csv",
            agg / "candidates.csv",
        ]
        for alt in alt_expr_paths:
            if alt.exists():
                print(f"[{NAME}] Loading expressions from: {alt}")
                dfe = pd.read_csv(alt)
                break
    
    if dfc is None:
        # Try complexity_stats as fallback (though it won't have per-model data)
        alt = agg / "complexity_stats.csv"
        if alt.exists():
            print(f"[{NAME}] Warning: Using complexity_stats.csv as fallback")
            dfc = pd.read_csv(alt)

    if dfe is None and dfc is None:
        # Show what files DO exist to help debug
        if agg.exists():
            existing = list(agg.glob("*.csv"))
            if existing:
                print(f"[{NAME}] Files found in {agg}:")
                for f in existing[:10]:
                    print(f"  - {f.name}")
            else:
                print(f"[{NAME}] No CSV files found in {agg}")
        raise FileNotFoundError(f"[{NAME}] 需要 expressions.csv 或 complexity_by_model.csv in {agg}")

    # Use whatever we have
    if dfe is None:
        print(f"[{NAME}] No expressions file, using complexity file only")
        dfe = dfc.copy()
    elif dfc is None:
        print(f"[{NAME}] No complexity file, using expressions file only")
        dfc = dfe.copy()
    else:
        print(f"[{NAME}] Found both files, merging...")

    # 合并以获得门控/复杂度 + 性能
    if not dfe.equals(dfc):
        on = [c for c in ["expr_id", "seed", "method", "experiment"] if c in dfe.columns and c in dfc.columns]
        if on:
            df = pd.merge(dfe, dfc, on=on, how="left", suffixes=("", "_c"))
        else:
            dfe = dfe.reset_index().rename(columns={"index": "_row"})
            dfc = dfc.reset_index().rename(columns={"index": "_row"})
            df = pd.merge(dfe, dfc, on="_row", how="left", suffixes=("", "_c"))
    else:
        df = dfe.copy()

    print(f"[{NAME}] Working with {len(df)} rows")

    df = _ensure_core_columns(df)
    df["score"] = _infer_score(df)

    if args.method and args.method.lower() != "all":
        df = df[df["method"].astype(str) == args.method]
        print(f"[{NAME}] Filtered to method '{args.method}': {len(df)} rows")
    
    if args.experiments:
        keep = set([x.strip() for x in args.experiments.split(",") if x.strip()])
        df = df[df["experiment"].astype(str).isin(keep)]
        print(f"[{NAME}] Filtered to experiments {keep}: {len(df)} rows")

    grp_cols = [c for c in ["dataset", "method", "experiment"] if c in df.columns]
    outs = []
    
    for keys, sub in df.groupby(grp_cols) if grp_cols else [((), df)]:
        sub = sub.dropna(subset=["num_gates", "complexity", "score"])
        if len(sub) == 0:
            continue
        sub = sub.sort_values("score", ascending=True).head(args.topk)
        
        if grp_cols:
            record = {k: v for k, v in zip(grp_cols, keys)}
        else:
            record = {}
        
        record.update(dict(
            n_models=int(len(sub)),
            n_with_gates=int((sub["num_gates"] > 0).sum()),
            prop_with_gates=float((sub["num_gates"] > 0).mean()),
            gates_median=float(sub["num_gates"].median()),
            gates_q1=float(np.nanpercentile(sub["num_gates"], 25)),
            gates_q3=float(np.nanpercentile(sub["num_gates"], 75)),
            complexity_median=float(sub["complexity"].median()),
            complexity_q1=float(np.nanpercentile(sub["complexity"], 25)),
            complexity_q3=float(np.nanpercentile(sub["complexity"], 75)),
            score_median=float(sub["score"].median())
        ))
        outs.append(record)

    out = pd.DataFrame(outs) if outs else pd.DataFrame()
    
    # Save to the correct location using dataset_dir
    output_path = agg / "gating_usage.csv"
    out.to_csv(output_path, index=False)
    print(f"[{NAME}] 写出: {output_path} with {len(out)} groups")


if __name__ == "__main__":
    main()