#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_gen_complexity_and_stability.py (FIXED)
Fixed issue: KeyError 'depth' - now handles missing columns gracefully
"""
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np

GATE_FUNCS = [
    r"lgo_thre",
    r"lgo_and\d+",
    r"lgo_or\d+",
    r"lgo_pair",
    r"gate_expr",
    r"\blgo\(",    # Soft gate (excludes nacked lgo of _thre/_and/_or/_pair). 
]

NAME = "04_gen_complexity_and_stability"

def _find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def _find_expressions_csv(dataset_dir: Path) -> Path:
    cand = [
        dataset_dir / "aggregated" / "expressions.csv",
        dataset_dir / "aggregated" / "candidates.csv",
        dataset_dir / "aggregated" / "expressions_topk.csv",
        dataset_dir / "expressions" / "topk_expressions.csv", 
        dataset_dir / "candidates" / "expressions.csv",
        dataset_dir / "candidates.csv",
    ]
    p = _find_first_existing(cand)
    if p is None:
        raise FileNotFoundError(
            f"[{NAME}] 未找到 expressions/candidates 文件，已尝试：\n" +
            "\n".join(str(x) for x in cand)
        )
    return p


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # 常见别名归一
    if "expr" in df.columns and "expression" not in df.columns:
        df = df.rename(columns={"expr": "expression"})
    if "exp" in df.columns and "experiment" not in df.columns:
        df = df.rename(columns={"exp": "experiment"})
    # 标准列补齐
    for col, default in [("method", "unknown"),
                         ("experiment", "unknown"),
                         ("seed", -1),
                         ("expr_id", None)]:
        if col not in df.columns:
            df[col] = default
    # expr_id 自动生成
    if df["expr_id"].isna().any() or df["expr_id"].iloc[0] is None:
        df["expr_id"] = np.arange(len(df))
    return df


def _paren_depth(s: str) -> int:
    d, md = 0, 0
    for ch in s:
        if ch == "(":
            d += 1
            md = max(md, d)
        elif ch == ")":
            d -= 1
    return max(md, 0)


_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _count_tokens(s: str) -> int:
    # 节点近似计数：函数名/变量名/常数均计为1
    if not isinstance(s, str) or not s:
        return 0
    return len(_TOKEN_RE.findall(s))


def _count_gates(s: str) -> dict:
    if not isinstance(s, str) or not s:
        return {"num_gates": 0, "num_AND": 0, "num_OR": 0, "num_soft": 0, "num_pair": 0, "num_gate_expr": 0}
    s0 = s
    num_thre = len(re.findall(r"\blgo_thre\s*\(", s0))
    num_and  = len(re.findall(r"\blgo_and\d*\s*\(", s0))
    num_or   = len(re.findall(r"\blgo_or\d*\s*\(", s0))
    num_pair = len(re.findall(r"\blgo_pair\s*\(", s0))
    num_gate_expr = len(re.findall(r"\bgate_expr\s*\(", s0))
    # 软门 lgo(   ——排除已经被上面匹配的复合门
    num_lgo_soft = 0
    for m in re.finditer(r"\blgo\s*\(", s0):
        start = m.start()
        # 检查是否为 lgo_thre/lgo_and/lgo_or/lgo_pair 的子串，若是则跳过
        if re.match(r"\blgo_(thre|and\d*|or\d*|pair)\s*\(", s0[start:]):
            continue
        num_lgo_soft += 1
    num_gates = num_thre + num_and + num_or + num_pair + num_gate_expr + num_lgo_soft
    return {
        "num_gates": num_gates,
        "num_AND": num_and,
        "num_OR": num_or,
        "num_soft": num_lgo_soft,
        "num_pair": num_pair,
        "num_gate_expr": num_gate_expr,
    }


def _compute_row_metrics(expr: str) -> dict:
    depth = _paren_depth(expr or "")
    tokens = _count_tokens(expr or "")
    gates = _count_gates(expr or "")
    out = {
        "complexity": int(tokens),
        "depth": int(depth),
        "expr_length": int(len(expr or "")),
    }
    out.update(gates)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="overall_xxx dataset_dir")
    ap.add_argument("--dataset", default=None, help="datset name")
    ap.add_argument("--method", default=None, help="for seleced method (optional)")
    ap.add_argument("--experiments", default=None, help="for selected experiment (optional)")
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    out_dir = ddir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple paths for expressions
    expr_paths = [
        ddir / "expressions" / "topk_expressions.csv",
        ddir / "aggregated" / "expressions.csv",
        ddir / "aggregated" / "candidates.csv",
    ]
    
    p_expr = None
    for p in expr_paths:
        if p.exists():
            p_expr = p
            print(f"[{NAME}] Using expressions file: {p}")
            break
    
    if p_expr is None:
        p_expr = _find_expressions_csv(ddir)
    
    df = pd.read_csv(p_expr)
    df = _normalize_cols(df)

    if args.method and args.method.lower() != "all":
        df = df[df["method"].astype(str) == args.method]
    if args.experiments:
        keep = set([x.strip() for x in args.experiments.split(",") if x.strip()])
        if "experiment" in df.columns:
            df = df[df["experiment"].astype(str).isin(keep)]

    dataset_name = args.dataset or ddir.name.replace("overall_", "")
    df["dataset"] = dataset_name

    # Compute metrics for each expression
    print(f"[{NAME}] Computing metrics for {len(df)} expressions...")
    
    # Get expression column name
    expr_col = "expression" if "expression" in df.columns else "expr" if "expr" in df.columns else None
    if expr_col is None:
        raise ValueError(f"[{NAME}] No expression column found in {p_expr}")
    
    # Compute metrics
    metrics_list = []
    for _, row in df.iterrows():
        expr = str(row[expr_col]) if pd.notna(row[expr_col]) else ""
        metrics = _compute_row_metrics(expr)
        metrics_list.append(metrics)
    
    # Create metrics dataframe
    mdf = pd.DataFrame(metrics_list)
    
    # Add computed metrics to original dataframe - FIXED: only add columns that exist in mdf
    need_cols = ["complexity", "depth", "expr_length", "num_gates", "num_AND", "num_OR", "num_soft", "num_pair", "num_gate_expr"]
    for c in need_cols:
        if c in mdf.columns:
            df[c] = mdf[c].values
        else:
            df[c] = 0  # Default value if column doesn't exist

    # dWrite detailed info by models (写逐模型明细)
    keep_cols = ["dataset", "method", "experiment", "seed", "expr_id", expr_col,
                 "complexity", "depth", "expr_length", "num_gates", "num_AND", "num_OR",
                 "num_soft", "num_pair", "num_gate_expr"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_by_model = df[keep_cols].copy()
    
    # Rename expression column to standard name
    if expr_col != "expression":
        df_by_model = df_by_model.rename(columns={expr_col: "expression"})
    
    df_by_model.to_csv(out_dir / "complexity_by_model.csv", index=False)

    # Generate distribution stat info (生成分布统计)
    grp_cols = ["dataset", "method", "experiment"]
    grp_cols = [c for c in grp_cols if c in df_by_model.columns]
    
    if len(grp_cols) > 0:
        agg = df_by_model.groupby(grp_cols).agg(
            n_models=("expr_id", "count"),
            complexity_median=("complexity", "median"),
            complexity_q1=("complexity", lambda s: float(np.nanpercentile(s, 25))),
            complexity_q3=("complexity", lambda s: float(np.nanpercentile(s, 75))),
            complexity_mean=("complexity", "mean"),
            complexity_std=("complexity", "std"),
            gates_median=("num_gates", "median"),
            gates_q1=("num_gates", lambda s: float(np.nanpercentile(s, 25))),
            gates_q3=("num_gates", lambda s: float(np.nanpercentile(s, 75))),
            depth_median=("depth", "median"),
        ).reset_index()
    else:
        # If no grouping columns, create single row summary
        agg = pd.DataFrame([{
            "dataset": dataset_name,
            "n_models": len(df_by_model),
            "complexity_median": df_by_model["complexity"].median(),
            "complexity_q1": float(np.nanpercentile(df_by_model["complexity"], 25)),
            "complexity_q3": float(np.nanpercentile(df_by_model["complexity"], 75)),
            "complexity_mean": df_by_model["complexity"].mean(),
            "complexity_std": df_by_model["complexity"].std(),
            "gates_median": df_by_model["num_gates"].median(),
            "gates_q1": float(np.nanpercentile(df_by_model["num_gates"], 25)),
            "gates_q3": float(np.nanpercentile(df_by_model["num_gates"], 75)),
            "depth_median": df_by_model["depth"].median(),
        }])

    agg.to_csv(out_dir / "complexity_stats.csv", index=False)
    print(f"[{NAME}] 写出: {out_dir/'complexity_by_model.csv'}; {out_dir/'complexity_stats.csv'}")


if __name__ == "__main__":
    main()