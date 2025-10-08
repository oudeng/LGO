#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_thresholds.py – Figure 4 generator (v3.7-adapted)
Heatmap of discovered LGU thresholds (median in natural units) vs. domain anchors.

Enhanced with seaborn for better visualization.

Color coding (Figure 4 spec):
  green  : |median - anchor|/|anchor| ≤ green_band  (default 0.10 → 10%)
  yellow : ≤ yellow_band                            (default 0.20 → 20%)
  red    : > yellow_band
  grey   : no anchor available or invalid median for that (dataset, feature) cell
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Optional dependency (PyYAML) for guidelines.yaml
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# -------------------------
# Color map (fixed palette)
# -------------------------
COLOR_OK   = "#2ca02c"  # green
COLOR_WARN = "#ffbf00"  # yellow/orange
COLOR_BAD  = "#d62728"  # red
COLOR_NA   = "#d9d9d9"  # grey

# category: → color
CATEGORY_TO_COLOR = {
    0: COLOR_OK,   # ≤ green_band
    1: COLOR_WARN, # ≤ yellow_band
    2: COLOR_BAD,  # > yellow_band
    3: COLOR_NA,   # no anchor or invalid median
}

# Default dataset names (align with v3.7)
DEFAULT_DATASETS = [
    "ICU_composite_risk_score",
    "NHANES_metabolic_score",
    "UCI_Heart_Cleveland_num",
    "UCI_HydraulicSys_fault_score",
    "UCI_CTG_NSPbin",
]

# Preferred gate types when a feature has multiple threshold rows
DEFAULT_GATE_PREFERENCE = ["lgu_thre", "gate_expr", "lgu_and2", "lgu_or2", "lgu_and3"]

# -------------------------
# Helpers
# -------------------------
def _normalize_feature_key(s: str) -> str:
    """Normalize feature names to snake_case."""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

def _try_parse_yaml(path: str) -> Dict[str, Any]:
    """Parse YAML file safely."""
    if not os.path.exists(path):
        return {}
    if yaml is None:
        raise RuntimeError(
            f"PyYAML is required to parse {path}, but not installed. "
            f"Install via 'pip install pyyaml' or provide --guideline_json."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _load_guidelines_from_config(config_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load central guidelines.yaml from config_dir, falling back to dataset-specific files.
    Returns: dict dataset_name -> {feature_key -> anchor_value}
    """
    anchors: Dict[str, Dict[str, float]] = {}
    # Preferred: central guidelines.yaml
    central = os.path.join(config_dir, "guidelines.yaml")
    if os.path.exists(central):
        data = _try_parse_yaml(central)
        global_map = { _normalize_feature_key(k): float(v)
                       for k,v in (data.get("global") or {}).items() if _is_number(v) }
        datasets = data.get("datasets") or {}
        for ds_name, feats in datasets.items():
            ds = {}
            if isinstance(feats, dict):
                for k, v in feats.items():
                    if _is_number(v):
                        ds[_normalize_feature_key(k)] = float(v)
            # Merge globals (dataset-specific overrides global)
            for k, v in global_map.items():
                ds.setdefault(k, v)
            anchors[str(ds_name)] = ds
    else:
        # Fallback: per-dataset guideline files
        for ds in DEFAULT_DATASETS:
            guess = os.path.join(config_dir, f"guidelines_{_dataset_tag(ds)}.yaml")
            if os.path.exists(guess):
                data = _try_parse_yaml(guess)
                global_map = { _normalize_feature_key(k): float(v)
                               for k,v in (data.get("global") or {}).items() if _is_number(v) }
                datasets = data.get("datasets") or {}
                feats = datasets.get(ds) or {}
                ds_map = {}
                if isinstance(feats, dict):
                    for k, v in feats.items():
                        if _is_number(v): 
                            ds_map[_normalize_feature_key(k)] = float(v)
                for k, v in global_map.items():
                    ds_map.setdefault(k, v)
                anchors[ds] = ds_map
    return anchors

def _dataset_tag(ds_name: str) -> str:
    """Helper for filenames like guidelines_ICU.yaml."""
    if "ICU" in ds_name: return "ICU"
    if "NHANES" in ds_name: return "NHANES"
    if "Cleveland" in ds_name: return "Cleveland"
    if "Hydraulic" in ds_name: return "Hydraulic"
    if "CTG" in ds_name: return "CTG"
    return ds_name

def _is_number(x) -> bool:
    """Check if value can be converted to float."""
    try:
        float(x)
        return True
    except Exception:
        return False

def _collect_thresholds(threshold_csv: List[str], dataset_dirs: List[str]) -> pd.DataFrame:
    """
    Collect threshold data from CSV files and dataset directories.
    """
    frames: List[pd.DataFrame] = []
    # (1) explicit threshold CSVs
    for p in (threshold_csv or []):
        if os.path.isdir(p):
            # recursively find any *threshold*.csv inside
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith(".csv") and "threshold" in fn.lower():
                        frames.append(_read_threshold_csv(os.path.join(root, fn)))
        else:
            if os.path.exists(p):
                frames.append(_read_threshold_csv(p))
            else:
                print(f"[WARN] thresholds CSV not found: {p}", file=sys.stderr)
    
    # (2) dataset dirs
    for d in (dataset_dirs or []):
        p = os.path.join(d, "aggregated", "thresholds_units.csv")
        if os.path.exists(p):
            frames.append(_read_threshold_csv(p))
        else:
            print(f"[WARN] Missing aggregated thresholds at: {p}", file=sys.stderr)
    
    if not frames:
        raise SystemExit("[ERROR] No threshold CSVs found. Provide --threshold_csv or --dataset_dirs.]")
    
    df = pd.concat(frames, ignore_index=True)
    return df

def _read_threshold_csv(path: str) -> pd.DataFrame:
    """Read and normalize threshold CSV."""
    df = pd.read_csv(path)
    # Normalize column names
    rename_map = {}
    for c in list(df.columns):
        lc = c.lower()
        if lc == "feature_norm": rename_map[c] = "feature_norm"
        if lc == "feature":      rename_map[c] = "feature"
        if lc == "unit":         rename_map[c] = "unit"
        if lc == "dataset":      rename_map[c] = "dataset"
        if lc == "method":       rename_map[c] = "method"
        if lc == "experiment":   rename_map[c] = "experiment"
        if lc == "gate_type":    rename_map[c] = "gate_type"
        if lc == "b_raw_median": rename_map[c] = "b_raw_median"
        if lc == "braw_median":  rename_map[c] = "b_raw_median"
        if lc == "b_raw":        rename_map[c] = "b_raw"
    
    if rename_map: 
        df = df.rename(columns=rename_map)

    # Check for required columns
    if "b_raw_median" not in df.columns and "b_raw" not in df.columns:
        raise ValueError(f"{path} missing 'b_raw_median' or 'b_raw'.")
    if "unit" not in df.columns:
        df["unit"] = ""

    # Unify feature key
    if "feature_norm" in df.columns:
        df["feature_key"] = df["feature_norm"].astype(str).map(_normalize_feature_key)
    elif "feature" in df.columns:
        df["feature_key"] = df["feature"].astype(str).map(_normalize_feature_key)
    else:
        raise ValueError(f"{path} missing 'feature' or 'feature_norm' column.")
    
    # Convert to strings
    for col in ["dataset","method","experiment","gate_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df

def _prefer_gate_rows(df: pd.DataFrame, gate_preference: List[str]) -> pd.DataFrame:
    """
    For each (dataset, feature_key), keep the first available gate_type according to preference.
    """
    if "gate_type" not in df.columns:
        return df
    
    keep_rows = []
    for (ds, feat), sub in df.groupby(["dataset","feature_key"], as_index=False):
        # Choose best gate available
        chosen = None
        for g in gate_preference:
            cand = sub[sub["gate_type"] == g]
            if not cand.empty:
                chosen = cand
                break
        if chosen is None:
            chosen = sub  # fallback to all
        keep_rows.append(chosen.iloc[[0]])  # one row
    
    return pd.concat(keep_rows, ignore_index=True) if keep_rows else df

def _resolve_median(series: pd.Series) -> float:
    """Calculate median of a series."""
    vals = series.dropna().values.astype(float) if len(series) else np.array([])
    if vals.size == 0: 
        return np.nan
    return float(np.nanmedian(vals))

def _aggregate_to_feature_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure one row per (dataset, feature_key): pick/compute the median in natural units.
    """
    have_bagg = "b_raw_median" in df.columns
    ag = (df.groupby(["dataset","feature_key","unit"], dropna=False)
            .agg(median=("b_raw_median", "median") if have_bagg else ("b_raw", _resolve_median))
            .reset_index()
            .rename(columns={"median": "b_raw_median"}))
    return ag

def _match_anchor(anchors_for_ds: Dict[str,float], feature_key: str) -> Tuple[float, bool]:
    """
    Try to match feature to anchor with various fallback strategies.
    """
    key = feature_key
    if key in anchors_for_ds:
        return float(anchors_for_ds[key]), True
    
    # Strip suffixes
    base = key
    for suf in ["_min","_max","_mean","_std","_q50","_median"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if base in anchors_for_ds:
        return float(anchors_for_ds[base]), True
    
    # Common synonyms
    alias_map = {
        "sbp": "systolic_bp",
        "sbp_min": "systolic_bp",
        "dbp": "diastolic_bp",
        "dbp_min": "diastolic_bp",
        "spo2": "spo2_min",
        "age": "age_years",
        "cholesterol": "chol",
        "glucose": "fasting_glucose",
    }
    if key in alias_map and alias_map[key] in anchors_for_ds:
        return float(anchors_for_ds[alias_map[key]]), True
    if base in alias_map and alias_map[base] in anchors_for_ds:
        return float(anchors_for_ds[alias_map[base]]), True
    
    return np.nan, False

def _format_pct(x: float) -> str:
    """Format percentage."""
    if not np.isfinite(x): 
        return "NA"
    return f"{x*100:.0f}%"

def _build_and_save_figure_seaborn(tab: pd.DataFrame,
                                   dataset: str,
                                   outdir: str,
                                   title_suffix: str,
                                   green_band: float,
                                   yellow_band: float,
                                   annotate: bool,
                                   dpi: int) -> str:
    """
    Draw an enhanced heatmap using seaborn.
    """
    # Sort rows by category then by name
    tab = tab.sort_values(["category","feature_key"]).reset_index(drop=True)
    
    # Prepare data for heatmap
    n_features = len(tab)
    
    # Create figure
    fig_h = max(4, 0.4 * n_features + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h), dpi=dpi)
    
    # Create matrix for heatmap (features x 1 dataset)
    matrix = tab["category"].values.reshape(-1, 1)
    
    # Custom colormap
    colors = [COLOR_OK, COLOR_WARN, COLOR_BAD, COLOR_NA]
    n_colors = len(colors)
    cmap = sns.color_palette(colors, n_colors=n_colors, as_cmap=False)
    
    # Create annotations if requested
    annot_matrix = None
    if annotate:
        annot_matrix = []
        for i in range(n_features):
            med = tab.iloc[i]["b_raw_median"]
            rel = tab.iloc[i]["rel_err"]
            txt = "NA"
            if np.isfinite(med):
                txt = f"{med:.1f}"
                if np.isfinite(rel): 
                    txt += f"\n({_format_pct(rel)})"
            annot_matrix.append([txt])
        annot_matrix = np.array(annot_matrix)
    
    # Draw heatmap using seaborn
    sns.heatmap(matrix,
                annot=annot_matrix if annotate else False,
                fmt='',
                cmap=cmap,
                vmin=-0.5,
                vmax=3.5,
                cbar=False,
                linewidths=0.5,
                linecolor='white',
                square=False,
                ax=ax,
                xticklabels=[dataset],
                yticklabels=[fk.replace("_"," ").title() for fk in tab["feature_key"].tolist()])
    
    # Title
    ttl = f"Discovered Thresholds vs. Anchors – {dataset}"
    if title_suffix: 
        ttl += f" – {title_suffix}"
    ax.set_title(ttl, fontsize=14, pad=15, fontweight='bold')
    
    # Labels
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    
    # Rotate y-axis labels for better readability
    plt.setp(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLOR_OK,   edgecolor='black', linewidth=0.5, 
              label=f"≤ {int(green_band*100)}%"),
        Patch(facecolor=COLOR_WARN, edgecolor='black', linewidth=0.5,
              label=f"≤ {int(yellow_band*100)}%"),
        Patch(facecolor=COLOR_BAD,  edgecolor='black', linewidth=0.5,
              label=f"> {int(yellow_band*100)}%"),
        Patch(facecolor=COLOR_NA,   edgecolor='black', linewidth=0.5,
              label="No anchor / NA"),
    ]
    
    # Position legend outside the plot
    ax.legend(handles=legend_elems, 
             title="|median−anchor|/anchor",
             loc='center left',
             bbox_to_anchor=(1.05, 0.5),
             frameon=True,
             fancybox=True,
             shadow=True,
             fontsize=10,
             title_fontsize=11)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f"fig4_thresholds_{dataset}.png")
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    return fpath

def build_heatmap_for_dataset(df: pd.DataFrame,
                              dataset: str,
                              anchors_all: Dict[str, Dict[str,float]],
                              green_band: float,
                              yellow_band: float,
                              outdir: str,
                              title_suffix: str,
                              annotate: bool,
                              dpi: int,
                              only_anchored: bool,
                              whitelist_keys: List[str],
                              gate_preference: List[str]) -> Tuple[str, pd.DataFrame]:
    """Compute per-feature medians, match to anchors, classify categories, apply filters, save figure."""
    # Subset by dataset
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        raise ValueError(f"No threshold rows for dataset={dataset}")

    # If multiple gate types exist for same feature, choose preferred
    if "gate_type" in sub.columns:
        sub = _prefer_gate_rows(sub, gate_preference)

    # Reduce to one row per feature (median in natural units)
    sub = _aggregate_to_feature_median(sub)

    # Anchors for this dataset
    anchors_for_ds = anchors_all.get(dataset, {})

    # Build summary table
    feature_keys = sorted(set(sub["feature_key"].tolist()) | set(anchors_for_ds.keys()))
    rows = []
    for fk in feature_keys:
        rec = sub[sub["feature_key"] == fk]
        med = float(rec["b_raw_median"].iloc[0]) if not rec.empty else np.nan
        unit = str(rec["unit"].iloc[0]) if not rec.empty else ""
        anchor, found = _match_anchor(anchors_for_ds, fk)
        
        if np.isfinite(anchor) and np.isfinite(med) and anchor != 0.0:
            rel = abs(med - anchor) / abs(anchor)
            if rel <= green_band: 
                cat = 0
            elif rel <= yellow_band: 
                cat = 1
            else: 
                cat = 2
        else:
            rel = np.nan
            cat = 3  # N/A
        
        rows.append({
            "dataset": dataset,
            "feature_key": fk,
            "unit": unit,
            "b_raw_median": med,
            "anchor": anchor if np.isfinite(anchor) else np.nan,
            "rel_err": rel,
            "category": cat,
        })
    
    tab = pd.DataFrame(rows)

    # Apply whitelist filter
    if whitelist_keys:
        mask = tab["feature_key"].isin(whitelist_keys)
        before = len(tab)
        tab = tab[mask].copy()
        if tab.empty:
            print(f"[WARN] After whitelist filtering, no features remain for dataset={dataset}.", file=sys.stderr)
        else:
            print(f"[INFO] Whitelist filter kept {len(tab)}/{before} features for dataset={dataset}.")

    # Apply only_anchored filter
    if only_anchored:
        before = len(tab)
        tab = tab[tab["category"] != 3].copy()
        if tab.empty:
            print(f"[WARN] After --only_anchored, no anchored features remain for dataset={dataset}.", file=sys.stderr)
        else:
            print(f"[INFO] --only_anchored kept {len(tab)}/{before} features for dataset={dataset}.")

    # If still empty, skip figure
    if tab.empty:
        return "", pd.DataFrame(columns=["dataset","feature_key","unit","b_raw_median","anchor","rel_err","category"])

    # Draw figure with seaborn
    fpath = _build_and_save_figure_seaborn(
        tab=tab, dataset=dataset, outdir=outdir, title_suffix=title_suffix,
        green_band=green_band, yellow_band=yellow_band, annotate=annotate, dpi=dpi
    )
    return fpath, tab

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Figure 4 (v3.7): Heatmap of discovered thresholds (median b_raw) vs. domain anchors\n"
            "Enhanced with seaborn visualization.\n"
            "Reads guidelines from CONFIG and thresholds from each dataset's aggregated CSV."
        )
    )
    ap.add_argument("--threshold_csv", nargs="+", default=[],
                    help="Optional: paths to thresholds CSV(s). If empty, will read from --dataset_dirs.")
    ap.add_argument("--dataset_dirs", nargs="+", default=[],
                    help="List of dataset result dirs (e.g., overall_ICU_composite_risk_score).")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="Datasets to include (must match 'dataset' in CSV/guidelines).")
    ap.add_argument("--config_dir", default="config",
                    help="Directory containing guidelines.yaml.")
    ap.add_argument("--method", default="lgu",
                    help="Filter by method (default: lgu).")
    ap.add_argument("--experiment", default="lgu_hard",
                    help="Filter by experiment (default: lgu_hard).")
    ap.add_argument("--gate_types", nargs="*", default=None,
                    help=f"Preferred gate types order (default: {DEFAULT_GATE_PREFERENCE}).")
    ap.add_argument("--green_band", type=float, default=0.10, 
                    help="Green threshold (≤10%%).")
    ap.add_argument("--yellow_band", type=float, default=0.20, 
                    help="Yellow threshold (≤20%%).")
    ap.add_argument("--annotate", action="store_true", 
                    help="Annotate each cell with median and Δ%%.")
    ap.add_argument("--dpi", type=int, default=180, 
                    help="Figure DPI.")
    ap.add_argument("--outdir", default="utility_plots/figs", 
                    help="Output directory for threshold heatmaps.")
    ap.add_argument("--only_anchored", action="store_true",
                    help="Keep only features with anchors and valid medians.")
    ap.add_argument("--feature_whitelist", nargs="+", default=None,
                    help="Optional list of feature names to keep.")

    args = ap.parse_args()

    # Set gate preference (fix for global declaration error)
    gate_preference = args.gate_types if args.gate_types else DEFAULT_GATE_PREFERENCE

    # Load thresholds
    df = _collect_thresholds(args.threshold_csv, args.dataset_dirs)

    # Filter by method/experiment
    if args.method and "method" in df.columns:
        df = df[df["method"].astype(str) == args.method]
    if args.experiment and "experiment" in df.columns:
        df = df[df["experiment"].astype(str) == args.experiment]

    # Check for required columns
    if "b_raw_median" not in df.columns and "b_raw" not in df.columns:
        raise SystemExit("[ERROR] Threshold CSVs missing 'b_raw_median' or 'b_raw'.")

    # Load guidelines (anchors)
    anchors_all = _load_guidelines_from_config(args.config_dir)
    if not anchors_all:
        print(f"[WARN] No guidelines loaded from {args.config_dir}. "
              f"Run 08_convert_ground_truth_to_guidelines.py first.", file=sys.stderr)

    # Prepare whitelist
    whitelist_keys: List[str] = []
    if args.feature_whitelist:
        whitelist_keys = [_normalize_feature_key(x) for x in args.feature_whitelist]

    # Generate figures per dataset
    os.makedirs(args.outdir, exist_ok=True)
    all_summaries = []
    
    for ds in args.datasets:
        if ds not in df["dataset"].unique():
            print(f"[WARN] Dataset '{ds}' not found in thresholds; skipping.", file=sys.stderr)
            continue
        
        title_suffix = ""
        if args.method or args.experiment:
            title_suffix = f"{args.method}:{args.experiment}".strip(":")
        
        try:
            fpath, summary = build_heatmap_for_dataset(
                df=df, dataset=ds, anchors_all=anchors_all,
                green_band=args.green_band, yellow_band=args.yellow_band,
                outdir=args.outdir, title_suffix=title_suffix,
                annotate=args.annotate, dpi=args.dpi,
                only_anchored=args.only_anchored,
                whitelist_keys=whitelist_keys,
                gate_preference=gate_preference
            )
            if fpath:
                print(f"[OK] Saved figure: {fpath}")
            else:
                print(f"[INFO] No figure produced for dataset={ds} (empty after filters).")
            all_summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] Failed to process dataset={ds}: {e}", file=sys.stderr)

    # Save combined summary
    if any(len(s) for s in all_summaries):
        out_summary = pd.concat(all_summaries, ignore_index=True)
        csv_path = os.path.join(args.outdir, "fig4_thresholds_summary.csv")
        out_summary.to_csv(csv_path, index=False)
        print(f"[OK] Saved summary CSV: {csv_path}")
    else:
        print("[WARN] No figures generated (check filters, datasets, and inputs).")

if __name__ == "__main__":
    main()