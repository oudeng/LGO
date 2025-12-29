#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_threshold_audit.py
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import re


NAME = "08_threshold_audit"


def _load_guidelines(gpath: Path) -> dict:
    if not gpath.exists():
        raise FileNotFoundError(f"[{NAME}] Can NOT find the guildeline file: {gpath}")
    with open(gpath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    global_map = { (k or "").strip(): float(v) for k, v in (data.get("global") or {}).items() }
    ds_map = {}
    for ds, mp in (data.get("datasets") or {}).items():
        ds_map[str(ds).strip()] = { (k or "").strip(): float(v) for k, v in (mp or {}).items() }
    return {"global": global_map, "datasets": ds_map}


def _norm_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\s\-\_]+", "", s)
    return s


def _alias_candidates(s: str):
    # 常见别名映射（必要时补充）
    aliases = {
        "sbp": "systolicbp",
        "systolicbp": "systolicbp",
        "sbpmin": "sbpmin",
        "spo2min": "spo2min",
        "hdlcholesterol": "hdlcholesterol",
        "triglycerides": "triglycerides",
        "bmi": "bmi",
        "fastingglucose": "fastingglucose",
        "lactate": "lactate",
        "thalach": "thalach",
        "chol": "chol",
        "waistcircumference": "waistcircumference",
    }
    k = _norm_key(s)
    if k in aliases:
        return aliases[k]
    return k


def _lookup_guideline(guides: dict, dataset: str, feature: str) -> float:
    gds = guides.get("datasets", {}).get(dataset, {})
    k = _alias_candidates(feature)
    # dataset-specific
    for key in [feature, k]:
        if key in gds:
            return float(gds[key])
    # global fallback
    gglobal = guides.get("global", {})
    for key in [feature, k]:
        if key in gglobal:
            return float(gglobal[key])
    return np.nan


def _first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--guidelines", required=True, help="YAML guildeline file path")
    ap.add_argument("--experiments", default=None, help="Selected experiment (optional)")
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    agg = ddir / "aggregated"; agg.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset or ddir.name.replace("overall_", "")

    p = agg / "thresholds_units.csv"
    if not p.exists():
        raise FileNotFoundError(f"[{NAME}] can NOT find {p} (Please run threshold aggregation script at first)")

    # NEW: thresholds_units.csv 可能存在但为空，pandas 读会报 EmptyDataError
    if p.stat().st_size == 0:
        print(f"[{NAME}] WARNING: {p} exists but is empty, nothing to audit.")
        # 输出空文件占位，避免下游再报错
        (agg / "threshold_audit.csv").write_text("", encoding="utf-8")
        (agg / "threshold_audit_summary.csv").write_text("", encoding="utf-8")
        return

    df = pd.read_csv(p)

    if args.experiments and "experiment" in df.columns:
        keep = set([x.strip() for x in args.experiments.split(",") if x.strip()])
        df = df[df["experiment"].astype(str).isin(keep)]

    # FIXED: Added more variations (兼容列名)
    feat_col = _first_existing(df, ["feature", "feat", "var", "variable"])
    unit_col = _first_existing(df, ["unit", "units"])
    # Handle both naming conventions
    med_col  = _first_existing(df, ["b_raw_median", "braw_median", "braw median", "braw_med", "median_braw", "braw"])
    q1_col   = _first_existing(df, ["b_raw_q1", "braw_q1", "q1", "braw q1"])
    q3_col   = _first_existing(df, ["b_raw_q3", "braw_q3", "q3", "braw q3"])

    if med_col is None:
        print(f"[{NAME}] Available columns: {df.columns.tolist()}")
        raise ValueError(f"[{NAME}] thresholds_units.csv 缺少必要列（feature 与 b_raw_median/braw_median）")
    
    if feat_col is None:
        raise ValueError(f"[{NAME}] thresholds_units.csv 缺少feature列")

    df["dataset"] = dataset_name
    df["feature_norm"] = df[feat_col].astype(str).map(_alias_candidates)
    df["median"] = pd.to_numeric(df[med_col], errors="coerce")
    if q1_col: df["q1"] = pd.to_numeric(df[q1_col], errors="coerce")
    if q3_col: df["q3"] = pd.to_numeric(df[q3_col], errors="coerce")
    if unit_col: df["unit"] = df[unit_col].astype(str)
    else: df["unit"] = ""

    guides = _load_guidelines(Path(args.guidelines))

    glist = []
    for _, r in df.iterrows():
        g = _lookup_guideline(guides, dataset_name, r["feature_norm"])
        glist.append(g)
    df["guideline"] = glist

    # Relative error and hit rate (相对误差与命中)
    eps = 1e-12
    df["rel_error"] = (df["median"] - df["guideline"]).abs() / (df["guideline"].abs() + eps)
    df["hit_10pct"] = (df["rel_error"] <= 0.10).astype(int)
    df["hit_20pct"] = (df["rel_error"] <= 0.20).astype(int)

    # Count each feature grequency (每个 feature 出现次数)
    df["N"] = 1
    grp_cols = [c for c in ["dataset", "experiment", "feature_norm"] if c in df.columns]
    if len(grp_cols) > 0:
        df["N"] = df.groupby(grp_cols)["N"].transform("count")

    keep = [c for c in ["dataset", "experiment", feat_col, "feature_norm", "unit",
                        "median", "q1", "q3", "guideline", "rel_error", "hit_10pct", "hit_20pct", "N"] if c in df.columns]
    out = df[keep].copy()
    out.to_csv(agg / "threshold_audit.csv", index=False)

    # Aggregation (汇总)
    grp_cols2 = [c for c in ["dataset", "experiment"] if c in out.columns]
    if len(grp_cols2) > 0:
        summ = out.groupby(grp_cols2).agg(
            n_features=("feature_norm", "nunique"),
            hit10_rate=("hit_10pct", "mean"),
            hit20_rate=("hit_20pct", "mean"),
            median_rel_error=("rel_error", "median"),
        ).reset_index()
    else:
        # Single row summary if no grouping columns
        summ = pd.DataFrame([{
            "dataset": dataset_name,
            "n_features": out["feature_norm"].nunique(),
            "hit10_rate": out["hit_10pct"].mean(),
            "hit20_rate": out["hit_20pct"].mean(),
            "median_rel_error": out["rel_error"].median(),
        }])
    
    summ.to_csv(agg / "threshold_audit_summary.csv", index=False)

    print(f"[{NAME}] 写出: {agg/'threshold_audit.csv'}; {agg/'threshold_audit_summary.csv'}")


if __name__ == "__main__":
    main()