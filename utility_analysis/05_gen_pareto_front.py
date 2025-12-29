#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_gen_pareto_front.py
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

NAME = "05_gen_pareto_front"


def _load_csv_robust(p: Path, description: str = "file"):
    """
    Robustly load CSV file, handling malformed lines by skipping them.
    """
    if not p.exists():
        return None
    
    try:
        # First try standard read
        return pd.read_csv(p)
    except pd.errors.ParserError as e:
        print(f"[{NAME}] Warning: Parser error in {description}, attempting recovery...")
        print(f"[{NAME}]   Error: {str(e)[:80]}...")
        
        try:
            # Try with on_bad_lines='warn' (pandas >= 1.3)
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                df = pd.read_csv(p, on_bad_lines='warn')
            print(f"[{NAME}]   ✓ Recovered {len(df)} rows (bad lines skipped)")
            return df
        except TypeError:
            # Fallback for older pandas versions (< 1.3)
            try:
                df = pd.read_csv(p, error_bad_lines=False, warn_bad_lines=True)
                print(f"[{NAME}]   ✓ Recovered {len(df)} rows (legacy mode)")
                return df
            except Exception as e2:
                print(f"[{NAME}]   ✗ Recovery failed: {e2}")
                return None
    except Exception as e:
        print(f"[{NAME}] Error loading {description}: {e}")
        return None


def _infer_loss(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}
    for key in ["cv_loss", "val_loss", "loss"]:
        if key in cols:
            return pd.to_numeric(df[cols[key]], errors='coerce')
    for key in ["rmse", "mae", "mse", "brier", "brier_score", "brier_score_loss", "brier score", "brier_score_mean", "brier_score__mean__"]:
        if key in cols:
            return pd.to_numeric(df[cols[key]], errors='coerce')
    # 需要把"越大越好"的指标转为 loss
    if "r2" in cols:
        return 1.0 - pd.to_numeric(df[cols["r2"]], errors='coerce')
    if "auroc" in cols:
        return 1.0 - pd.to_numeric(df[cols["auroc"]], errors='coerce')
    # 兜底：若没有可用列，以 NaN
    return pd.Series(np.nan, index=df.index)


def _merge_metrics(df_complex: pd.DataFrame, df_expr: pd.DataFrame) -> pd.DataFrame:
    # 使用共有键合并；优先 expr_id + seed + method + experiment
    on = [c for c in ["expr_id", "seed", "method", "experiment"] if c in df_complex.columns and c in df_expr.columns]
    if not on:
        # 退化：仅按行号拼（不推荐），保证列齐全
        df_expr = df_expr.reset_index().rename(columns={"index": "_row"})
        df_complex = df_complex.reset_index().rename(columns={"index": "_row"})
        dfm = pd.merge(df_complex, df_expr, on="_row", how="left", suffixes=("", "_expr"))
    else:
        dfm = pd.merge(df_complex, df_expr, on=on, how="left", suffixes=("", "_expr"))
    # 推断 loss
    dfm["loss"] = _infer_loss(dfm)
    return dfm


def _pareto_front(group: pd.DataFrame) -> pd.DataFrame:
    g = group.dropna(subset=["complexity", "loss"]).sort_values(["complexity", "loss"], ascending=[True, True])
    best = []
    best_loss = np.inf
    for _, row in g.iterrows():
        if row["loss"] < best_loss - 1e-12:
            best.append(row)
            best_loss = row["loss"]
    if not best:
        return g.head(0)
    return pd.DataFrame(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Dataset directory (e.g., overall_ICU_composite_risk_score)")
    ap.add_argument("--dataset", default=None, help="Dataset name for output (ignored for paths)")
    ap.add_argument("--method", default=None)
    ap.add_argument("--experiments", default=None, help="逗号分隔，仅保留这些 experiment")
    args = ap.parse_args()

    # CRITICAL: Use dataset_dir for ALL paths, NOT dataset
    ddir = Path(args.dataset_dir)  # This should be overall_ICU_composite_risk_score
    print(f"[{NAME}] Using dataset directory: {ddir}")
    
    if not ddir.exists():
        raise FileNotFoundError(f"[{NAME}] Dataset directory does not exist: {ddir}")
    
    agg = ddir / "aggregated"
    agg.mkdir(parents=True, exist_ok=True)
    
    # Build correct path using dataset_dir
    p_complex = agg / "complexity_by_model.csv"
    print(f"[{NAME}] Looking for complexity file at: {p_complex}")
    
    if not p_complex.exists():
        # Show what files DO exist to help debug
        if agg.exists():
            existing = list(agg.glob("*.csv"))
            if existing:
                print(f"[{NAME}] Files found in {agg}:")
                for f in existing[:5]:
                    print(f"  - {f.name}")
        raise FileNotFoundError(f"[{NAME}] 需要 {p_complex}（请先运行 04 脚本）")
    
    dfc = _load_csv_robust(p_complex, "complexity_by_model.csv")
    if dfc is None:
        raise RuntimeError(f"[{NAME}] Failed to load complexity file")
    print(f"[{NAME}] Loaded complexity_by_model.csv with {len(dfc)} rows")

    # Try to find expressions file (with robust loading)
    p_expr = agg / "expressions.csv"
    dfe = _load_csv_robust(p_expr, "expressions.csv")
    
    if dfe is None:
        # Try alternative paths
        alt_paths = [
            ddir / "expressions" / "topk_expressions.csv",
            agg / "candidates.csv",
        ]
        for alt in alt_paths:
            if alt.exists():
                print(f"[{NAME}] Using expressions from: {alt}")
                dfe = _load_csv_robust(alt, str(alt.name))
                if dfe is not None:
                    break
    
    if dfe is None:
        # 退化路径：用 dfc 自身尝试推断 loss
        print(f"[{NAME}] No separate expressions file found, using complexity file")
        dfe = dfc.copy()

    # Check if expressions.csv already has both complexity and loss
    dfe_has_loss = any(c.lower() in ['cv_loss', 'loss', 'val_loss'] for c in dfe.columns)
    dfe_has_complexity = 'complexity' in dfe.columns
    
    if dfe_has_loss and dfe_has_complexity:
        print(f"[{NAME}] expressions.csv has both complexity and loss - using directly")
        dfm = dfe.copy()
        dfm["loss"] = _infer_loss(dfm)
    else:
        # Need to merge with complexity file
        if args.method and args.method.lower() != "all":
            if "method" in dfc.columns:
                dfc = dfc[dfc["method"].astype(str) == args.method]
            if "method" in dfe.columns:
                dfe = dfe[dfe["method"].astype(str) == args.method]
        
        if args.experiments:
            keep = set([x.strip() for x in args.experiments.split(",") if x.strip()])
            if "experiment" in dfc.columns:
                dfc = dfc[dfc["experiment"].astype(str).isin(keep)]
            if "experiment" in dfe.columns:
                dfe = dfe[dfe["experiment"].astype(str).isin(keep)]

        dfm = _merge_metrics(dfc, dfe)
    
    # Apply filters after merge if we used expressions directly
    if dfe_has_loss and dfe_has_complexity:
        if args.method and args.method.lower() != "all":
            if "method" in dfm.columns:
                dfm = dfm[dfm["method"].astype(str) == args.method]
        
        if args.experiments:
            keep = set([x.strip() for x in args.experiments.split(",") if x.strip()])
            if "experiment" in dfm.columns:
                dfm = dfm[dfm["experiment"].astype(str).isin(keep)]
    print(f"[{NAME}] Merged data has {len(dfm)} rows")

    grp_cols = [c for c in ["dataset", "method", "experiment"] if c in dfm.columns]
    outs = []
    for keys, sub in dfm.groupby(grp_cols) if grp_cols else [((), dfm)]:
        pf = _pareto_front(sub)
        outs.append(pf)
    
    out = pd.concat(outs, ignore_index=True) if outs else dfm.head(0)

    keep = [c for c in ["dataset", "method", "experiment", "seed", "expr_id", "complexity", "loss", "expression", "expr"] if c in out.columns]
    out = out[keep].copy()
    
    # Save to the correct location using dataset_dir
    output_path = agg / "pareto_front.csv"
    out.to_csv(output_path, index=False)
    print(f"[{NAME}] 写出: {output_path} with {len(out)} points")


if __name__ == "__main__":
    main()