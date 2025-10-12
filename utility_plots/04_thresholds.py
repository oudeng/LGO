# -*- coding: utf-8 -*-
"""
04_thresholds_v3_7.py – Enhanced threshold audit v3.7 (publication-ready)

Key improvements in v3.7:
- Completely removes grey rows when using --only_anchored
- Requires BOTH guideline AND median threshold for feature inclusion
- Proper quality score fallback when coverage is missing
- Clean top-k selection for publication figures

Core features:
- Automatic filtering of operator pseudo-features
- Consistent method/experiment filtering
- Absolute error band for near-zero anchors
- Enhanced alias mapping
- Coverage calculation support (when data available)
- Top-k feature selection with smart quality scoring
- Unit preservation and display in outputs
"""
import argparse, os, sys, math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except:
    yaml = None

# ---------------- Configuration ----------------
COLOR_OK   = "#2ca02c"   # green
COLOR_WARN = "#ffbf00"   # yellow
COLOR_BAD  = "#d62728"   # red
COLOR_NA   = "#d9d9d9"   # grey
COV_CMAP   = "Greys"     # coverage heatmap

DEFAULT_DATASETS = [
    "ICU_composite_risk_score",
    "NHANES_metabolic_score",
    "UCI_Heart_Cleveland_num",
    "UCI_HydraulicSys_fault_score",
    "UCI_CTG_NSPbin",
]

# Default tokens to drop (operator pseudo-features)
DEFAULT_DROP_TOKENS = [
    'add', 'sub', 'mul', 'div', 'id', 'idF', 
    'gate_expr', 'lgu_thre', 'zero', 'one', 'unknown'
]

# Extended alias mapping for feature name variations
ALIAS_MAP = {
    # Blood pressure
    'systolic_bp': ['sbp', 'sbp_min', 'systolicbp'],
    'diastolic_bp': ['dbp', 'dbp_min', 'diastolicbp'],
    # Heart rate
    'heart_rate': ['hr', 'hr_max', 'hrmax'],
    # Respiratory
    'respiratory_rate': ['resprate', 'resprate_max', 'respratemax'],
    # Glucose
    'fasting_glucose': ['glucose', 'fastingglucose'],
    'glucose': ['fasting_glucose', 'fastingglucose'],
    # Cholesterol
    'cholesterol': ['chol', 'total_cholesterol'],
    'hdl_cholesterol': ['hdl', 'hdlcholesterol'],
    # Other vitals
    'lactate': ['lactate_mmol_l', 'lactatemmoll'],
    'creatinine': ['creatinine_mg_dl', 'creatininemgdl'],
    'map': ['map_mmhg', 'mapmmhg', 'mean_arterial_pressure'],
    'spo2': ['spo2_min', 'oxygen_saturation'],
    # Demographics
    'age': ['age_years', 'age_band'],
    'waist': ['waist_circumference', 'waistcircumference'],
}

# ------------- Utilities -------------
def _norm(s: str) -> str:
    """Normalize for comparison - preserve underscores"""
    return str(s).strip().lower()

def _isnum(x) -> bool:
    try:
        float(x)
        return True
    except:
        return False

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    if yaml is None:
        print(f"[WARN] PyYAML required for {path}. pip install pyyaml")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _load_guidelines(config_dir: str) -> Dict[str, Dict[str, float]]:
    """Load guidelines from YAML - preserve exact feature names"""
    anchors = {}
    central = os.path.join(config_dir, "guidelines.yaml")
    if os.path.exists(central):
        data = _load_yaml(central)
        
        # Global guidelines
        gmap = {}
        if 'global' in data and isinstance(data['global'], dict):
            for k, v in data['global'].items():
                if _isnum(v):
                    gmap[str(k)] = float(v)
        
        # Dataset-specific guidelines
        dsets = data.get('datasets') or {}
        for ds, feats in dsets.items():
            mm = {}
            if isinstance(feats, dict):
                for k, v in feats.items():
                    if _isnum(v): 
                        mm[str(k)] = float(v)
            # Add global defaults
            for k, v in gmap.items():
                mm.setdefault(k, v)
            anchors[str(ds)] = mm
    
    return anchors

def _read_csv(path: str, drop_tokens: List[str]) -> pd.DataFrame:
    """Read CSV and clean data"""
    df = pd.read_csv(path)
    
    # Normalize column names
    ren = {}
    for c in df.columns:
        lc = c.lower()
        mapping = {
            "dataset": "dataset",
            "method": "method",
            "experiment": "experiment",
            "feature": "feature",
            "feature_norm": "feature_norm",
            "unit": "unit",
            "median": "median",
            "q1": "q1",
            "q3": "q3",
            "guideline": "guideline",
            "rel_error": "rel_error",
            "abs_error": "abs_error",
            "b_raw_median": "b_raw_median",
            "gate_type": "gate_type",
            "n_features": "n_features",
            "hit10_rate": "hit10_rate", 
            "hit20_rate": "hit20_rate",
            "median_rel_error": "median_rel_error"
        }
        if lc in mapping:
            ren[c] = mapping[lc]
    
    if ren:
        df = df.rename(columns=ren)
    
    # Create feature_key - preserve underscores
    if "feature_norm" in df.columns:
        df["feature_key"] = df["feature_norm"].astype(str).str.lower().str.strip()
    elif "feature" in df.columns:
        df["feature_key"] = df["feature"].astype(str).str.lower().str.strip()
    
    # Filter out operator tokens
    if "feature_key" in df.columns and drop_tokens:
        df = df[~df["feature_key"].isin([t.lower() for t in drop_tokens])]
    if "feature" in df.columns and drop_tokens:
        df = df[~df["feature"].str.lower().isin([t.lower() for t in drop_tokens])]
    
    return df

def _lookup_guideline(feature: str, guidelines: Dict[str, float]) -> float:
    """Enhanced guideline lookup with alias support"""
    feature_lower = feature.lower().strip()
    
    # Direct match
    if feature in guidelines:
        return guidelines[feature]
    
    # Case-insensitive match
    for guide_key, value in guidelines.items():
        if guide_key.lower() == feature_lower:
            return value
    
    # Check alias mappings
    for base_name, aliases in ALIAS_MAP.items():
        if feature_lower in [a.lower() for a in aliases]:
            if base_name in guidelines:
                return guidelines[base_name]
            # Try lowercase
            for guide_key, value in guidelines.items():
                if guide_key.lower() == base_name.lower():
                    return value
    
    # Reverse check - is the guideline key an alias?
    for guide_key, value in guidelines.items():
        guide_lower = guide_key.lower()
        for base_name, aliases in ALIAS_MAP.items():
            if guide_lower in [a.lower() for a in aliases] and feature_lower == base_name.lower():
                return value
    
    # Try without underscores as last resort
    feature_no_underscore = feature_lower.replace('_', '')
    for guide_key, value in guidelines.items():
        if guide_key.lower().replace('_', '') == feature_no_underscore:
            return value
    
    return np.nan

def _classify_error(rel_error: float, abs_error: float, guideline: float,
                    green: float = 0.10, yellow: float = 0.20, 
                    abs_band: float = 0.0) -> int:
    """
    Classify error with absolute band support for near-zero anchors
    Returns: 0=green, 1=yellow, 2=red, 3=NA
    """
    # No guideline = NA
    if not np.isfinite(guideline):
        return 3
    
    # Near-zero anchor with absolute band
    if abs_band > 0 and abs(guideline) < 0.1:
        if np.isfinite(abs_error):
            if abs_error <= abs_band:
                return 0  # Green
            elif abs_error <= abs_band * 2:
                return 1  # Yellow
            else:
                return 2  # Red
    
    # Standard relative error classification
    if not np.isfinite(rel_error):
        return 3
    if rel_error <= green:
        return 0
    if rel_error <= yellow:
        return 1
    return 2

def _plot_agreement_heatmap(df: pd.DataFrame, dataset: str, outdir: str, 
                            title_suffix: str, annotate: bool, dpi: int) -> str:
    """Plot agreement heatmap"""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    # Sort by error category and feature name
    df = df.sort_values(["error_cat", "feature_key"]).reset_index(drop=True)
    
    cats = df["error_cat"].values.astype(float).reshape(-1, 1)
    cmap = ListedColormap([COLOR_OK, COLOR_WARN, COLOR_BAD, COLOR_NA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    n = len(df)
    fig_w, fig_h = 6.8, max(3.0, 0.38 * n + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    
    ax.imshow(cats, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(np.arange(n))
    
    # Format feature names with units if available
    labels = []
    for _, row in df.iterrows():
        label = row["feature_key"].replace("_", " ")
        if pd.notna(row.get("unit")) and row["unit"]:
            label += f" ({row['unit']})"
        labels.append(label)
    
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks([0])
    ax.set_xticklabels([dataset], fontsize=10)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Feature")
    
    title = f"Threshold Agreement with Guidelines – {dataset}"
    if title_suffix:
        title += f" – {title_suffix}"
    ax.set_title(title, fontsize=12, pad=10)
    
    # Annotate cells
    if annotate:
        for i, row in df.iterrows():
            med = row.get("median", np.nan)
            err = row.get("rel_error", np.nan)
            cov = row.get("coverage", np.nan)
            text = ""
            if np.isfinite(med):
                text += f"{med:.2f}"
            if np.isfinite(err):
                text += f"\n{int(err*100)}%"
            if np.isfinite(cov):
                text += f"\nC:{int(cov*100)}%"
            if text:
                color = "white" if row["error_cat"] < 3 else "black"
                ax.text(0, i, text, ha="center", va="center", fontsize=9, color=color)
    
    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR_OK, label="≤10%"),
        Patch(facecolor=COLOR_WARN, label="≤20%"),
        Patch(facecolor=COLOR_BAD, label=">20%"),
        Patch(facecolor=COLOR_NA, label="No guideline"),
    ]
    ax.legend(handles=handles, title="Relative Error", loc="lower right",
              bbox_to_anchor=(1.3, 0.02), frameon=False, fontsize=9)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"v3_agreement_{dataset}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_coverage_heatmap(df: pd.DataFrame, dataset: str, outdir: str,
                          title_suffix: str, annotate: bool, dpi: int) -> str:
    """Plot coverage heatmap showing feature stability across seeds/models"""
    if 'coverage' not in df.columns or df['coverage'].isna().all():
        return ""
    
    df = df.sort_values(['coverage', 'feature_key'], ascending=[False, True]).reset_index(drop=True)
    vals = df['coverage'].fillna(0).values.reshape(-1, 1)
    
    n = vals.shape[0]
    fig_w, fig_h = 6.8, max(3.0, 0.38 * n + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    
    im = ax.imshow(vals, cmap=COV_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(np.arange(n))
    
    # Format labels
    labels = []
    for _, row in df.iterrows():
        label = row["feature_key"].replace("_", " ")
        if pd.notna(row.get("unit")) and row["unit"]:
            label += f" ({row['unit']})"
        labels.append(label)
    
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks([0])
    ax.set_xticklabels([dataset], fontsize=10)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Feature")
    
    title = f"Gate Coverage (seeds/models) – {dataset}"
    if title_suffix:
        title += f" – {title_suffix}"
    ax.set_title(title, fontsize=12, pad=10)
    
    if annotate:
        for i, row in df.iterrows():
            cov = row.get('coverage', np.nan)
            n_feat = row.get('n_seeds_feat', row.get('n_models_feat', np.nan))
            n_ds = row.get('n_seeds_ds', row.get('n_models_ds', np.nan))
            if np.isfinite(cov):
                text = f"{int(round(100*cov))}%"
                if np.isfinite(n_feat) and np.isfinite(n_ds):
                    text += f"\n{int(n_feat)}/{int(n_ds)}"
                color = "white" if cov > 0.5 else "black"
                ax.text(0, i, text, ha="center", va="center", fontsize=9, color=color)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coverage", rotation=270, labelpad=12)
    
    plt.tight_layout()
    path = os.path.join(outdir, f"v3_coverage_{dataset}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_error_distribution(df: pd.DataFrame, dataset: str, outdir: str,
                            title_suffix: str, dpi: int) -> str:
    """Plot IQR bars showing threshold variability"""
    df_valid = df[df["has_iqr"]].sort_values("rel_error")
    if df_valid.empty:
        return ""
    
    n = len(df_valid)
    fig_w = 10
    fig_h = max(4, 0.3 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    
    y_pos = np.arange(n)
    
    for i, (_, row) in enumerate(df_valid.iterrows()):
        ax.barh(i, row["q3"] - row["q1"], left=row["q1"], height=0.6,
                color="lightblue", edgecolor="black", linewidth=1)
        ax.plot(row["median"], i, 'ko', markersize=6)
        if np.isfinite(row["guideline"]):
            ax.plot(row["guideline"], i, 'r|', markersize=10, markeredgewidth=2)
    
    ax.set_yticks(y_pos)
    labels = []
    for _, row in df_valid.iterrows():
        label = row["feature_key"].replace("_", " ")
        if pd.notna(row.get("unit")) and row["unit"]:
            label += f" ({row['unit']})"
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=9)
    
    ax.set_xlabel("Threshold Value")
    ax.set_ylabel("Feature")
    
    title = f"Threshold Distribution (Q1-Median-Q3) – {dataset}"
    if title_suffix:
        title += f" – {title_suffix}"
    ax.set_title(title, fontsize=12)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightblue', linewidth=8, label='IQR (Q1-Q3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', 
               markersize=6, label='Median'),
        Line2D([0], [0], marker='|', color='w', markerfacecolor='r', 
               markersize=10, markeredgewidth=2, label='Guideline'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(outdir, f"v3_error_dist_{dataset}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def process_dataset(dataset: str, df_units: pd.DataFrame, df_audit: pd.DataFrame,
                   df_summary: pd.DataFrame, anchors: Dict[str, Dict[str, float]], 
                   args) -> pd.DataFrame:
    """Process a single dataset with all enhancements"""
    
    # Filter by dataset
    U = df_units[df_units["dataset"] == dataset].copy() if not df_units.empty else pd.DataFrame()
    A = df_audit[df_audit["dataset"] == dataset].copy() if not df_audit.empty else pd.DataFrame()
    S = df_summary[df_summary["dataset"] == dataset].copy() if not df_summary.empty else pd.DataFrame()
    
    # FIXED: Apply consistent method/experiment filtering - properly update dataframes
    if not U.empty:
        if "method" in U.columns and args.method:
            U = U[U["method"] == args.method].copy()
        if "experiment" in U.columns and args.experiment:
            U = U[U["experiment"] == args.experiment].copy()
    
    if not A.empty:
        if "method" in A.columns and args.method:
            A = A[A["method"] == args.method].copy()
        if "experiment" in A.columns and args.experiment:
            A = A[A["experiment"] == args.experiment].copy()
    
    if not S.empty:
        if "method" in S.columns and args.method:
            S = S[S["method"] == args.method].copy()
        if "experiment" in S.columns and args.experiment:
            S = S[S["experiment"] == args.experiment].copy()
    
    if A.empty and U.empty:
        return pd.DataFrame()
    
    # Get guidelines for this dataset
    dataset_guidelines = anchors.get(dataset, {})
    
    # Build unit mapping from units table
    unit_map = {}
    if not U.empty and "feature" in U.columns and "unit" in U.columns:
        for _, row in U.iterrows():
            feat = row.get("feature", "")
            unit = row.get("unit", "")
            if feat and unit:
                unit_map[feat.lower()] = unit
                # Also map normalized version
                if "feature_norm" in row:
                    unit_map[row["feature_norm"].lower()] = unit
    
    # Calculate coverage if summary data available
    coverage_map = {}
    if not S.empty and 'hit10_rate' in S.columns:
        # Use hit rate as proxy for coverage
        for _, row in S.iterrows():
            if 'n_features' in row:
                # Estimate coverage from hit rates
                coverage_map['_global'] = row.get('hit10_rate', 0) + row.get('hit20_rate', 0)
    
    results = []
    
    # Process audit data (has quartiles and guidelines)
    if not A.empty:
        for _, row in A.iterrows():
            feature = row.get("feature", "")
            feature_norm = row.get("feature_norm", feature)
            feature_key = feature
            
            # Skip operator tokens
            if feature_key.lower() in [t.lower() for t in args.drop_tokens]:
                continue
            
            # Get threshold values
            median = row.get("median", np.nan)
            q1 = row.get("q1", np.nan)
            q3 = row.get("q3", np.nan)
            
            # Get guideline with enhanced lookup
            guideline = row.get("guideline", np.nan)
            if not np.isfinite(guideline):
                guideline = _lookup_guideline(feature, dataset_guidelines)
            if not np.isfinite(guideline) and feature_norm:
                guideline = _lookup_guideline(feature_norm, dataset_guidelines)
            
            # Calculate errors
            rel_error = row.get("rel_error", np.nan)
            abs_error = row.get("abs_error", np.nan)
            
            if np.isfinite(median) and np.isfinite(guideline):
                if guideline != 0:
                    rel_error = abs(median - guideline) / abs(guideline)
                abs_error = abs(median - guideline)
            
            # Classify error with absolute band support
            error_cat = _classify_error(rel_error, abs_error, guideline,
                                       args.green_band, args.yellow_band, args.abs_band)
            
            # FIXED v3.7: only_anchored requires BOTH guideline AND median
            if args.only_anchored and (not np.isfinite(guideline) or not np.isfinite(median)):
                continue
            
            # Get unit - first from row, then from unit_map
            unit = row.get("unit", "")
            if not unit:
                unit = unit_map.get(feature.lower(), "")
                if not unit:
                    unit = unit_map.get(feature_norm.lower() if feature_norm else "", "")
            
            # Estimate coverage
            coverage = coverage_map.get(feature_key, coverage_map.get('_global', np.nan))
            
            # FIXED: Better quality score calculation with fallback
            cov_for_score = coverage if np.isfinite(coverage) and coverage > 0 else 1.0 if np.isfinite(rel_error) else 0.0
            quality_score = (1 - rel_error) * cov_for_score if np.isfinite(rel_error) else 0.0
            
            results.append({
                "dataset": dataset,
                "feature": feature,
                "feature_key": feature_key,
                "median": median,
                "q1": q1,
                "q3": q3,
                "guideline": guideline,
                "rel_error": rel_error,
                "abs_error": abs_error,
                "error_cat": error_cat,
                "has_iqr": np.isfinite(q1) and np.isfinite(q3),
                "unit": unit,
                "coverage": coverage,
                "quality_score": quality_score
            })
    
    # Add features from units that aren't in audit
    elif not U.empty:
        for _, row in U.iterrows():
            feature = row.get("feature", "")
            feature_key = feature
            
            if feature_key.lower() in [t.lower() for t in args.drop_tokens]:
                continue
            
            median = row.get("b_raw_median", np.nan)
            guideline = _lookup_guideline(feature, dataset_guidelines)
            
            rel_error = np.nan
            abs_error = np.nan
            if np.isfinite(median) and np.isfinite(guideline):
                if guideline != 0:
                    rel_error = abs(median - guideline) / abs(guideline)
                abs_error = abs(median - guideline)
            
            error_cat = _classify_error(rel_error, abs_error, guideline,
                                       args.green_band, args.yellow_band, args.abs_band)
            
            # FIXED v3.7: only_anchored requires BOTH guideline AND median
            if args.only_anchored and (not np.isfinite(guideline) or not np.isfinite(median)):
                continue
            
            # FIXED v3.7: Better quality score with proper fallback
            if np.isfinite(rel_error):
                base_score = 1.0 - rel_error
            else:
                base_score = 0.0
            
            # Use coverage if available, otherwise just base_score
            if np.isfinite(coverage_map.get(feature_key, np.nan)) and coverage_map.get(feature_key, 0) > 0:
                quality_score = base_score * coverage_map.get(feature_key, 0)
            else:
                quality_score = base_score
            
            results.append({
                "dataset": dataset,
                "feature": feature,
                "feature_key": feature_key,
                "median": median,
                "q1": np.nan,
                "q3": np.nan,
                "guideline": guideline,
                "rel_error": rel_error,
                "abs_error": abs_error,
                "error_cat": error_cat,
                "has_iqr": False,
                "unit": row.get("unit", ""),
                "coverage": np.nan,
                "quality_score": quality_score
            })
    
    df_results = pd.DataFrame(results)
    
    # Apply top-k filtering if requested
    if args.topk and args.topk > 0 and len(df_results) > args.topk:
        # Sort by quality score (coverage * accuracy)
        df_results = df_results.nlargest(args.topk, 'quality_score')
    
    return df_results

def main():
    ap = argparse.ArgumentParser(
        description="Enhanced threshold audit v3 with comprehensive improvements"
    )
    ap.add_argument("--dataset_dirs", nargs="+", default=[],
                    help="Dataset directories containing aggregated/*.csv")
    ap.add_argument("--config_dir", default="config",
                    help="Directory containing guidelines.yaml")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="Datasets to process")
    ap.add_argument("--method", default="lgu",
                    help="Filter by method")
    ap.add_argument("--experiment", default="lgu_hard",
                    help="Filter by experiment")
    ap.add_argument("--only_anchored", action="store_true",
                    help="Only show features with guidelines")
    
    # Error thresholds
    ap.add_argument("--green_band", type=float, default=0.10,
                    help="Green threshold (≤10% default)")
    ap.add_argument("--yellow_band", type=float, default=0.20,
                    help="Yellow threshold (≤20% default)")
    ap.add_argument("--abs_band", type=float, default=0.0,
                    help="Absolute error band for near-zero anchors (0=disabled)")
    
    # Token filtering
    ap.add_argument("--drop_tokens", nargs="*", default=DEFAULT_DROP_TOKENS,
                    help="Tokens to filter out (operators, etc)")
    
    # Display options
    ap.add_argument("--topk", type=int, default=0,
                    help="Show only top-k features by quality score")
    ap.add_argument("--annotate", action="store_true",
                    help="Annotate heatmap cells")
    ap.add_argument("--draw_coverage", action="store_true",
                    help="Draw coverage heatmap")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--outdir", default="figs/thresholds_v3")
    
    args = ap.parse_args()
    
    # Load guidelines
    anchors = _load_guidelines(args.config_dir)
    print(f"[INFO] Loaded guidelines for {len(anchors)} datasets")
    print(f"[INFO] Filtering tokens: {args.drop_tokens}")
    
    # Collect all CSVs
    all_units = []
    all_audit = []
    all_summary = []
    
    for dir_path in args.dataset_dirs:
        if os.path.isdir(dir_path):
            units_path = os.path.join(dir_path, "aggregated", "thresholds_units.csv")
            audit_path = os.path.join(dir_path, "aggregated", "threshold_audit.csv")
            summary_path = os.path.join(dir_path, "aggregated", "threshold_audit_summary.csv")
            
            if os.path.exists(units_path):
                all_units.append(_read_csv(units_path, args.drop_tokens))
            if os.path.exists(audit_path):
                all_audit.append(_read_csv(audit_path, args.drop_tokens))
            if os.path.exists(summary_path):
                all_summary.append(_read_csv(summary_path, []))
    
    df_units = pd.concat(all_units, ignore_index=True) if all_units else pd.DataFrame()
    df_audit = pd.concat(all_audit, ignore_index=True) if all_audit else pd.DataFrame()
    df_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    
    if df_units.empty and df_audit.empty:
        print("[ERROR] No data files found")
        return
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Process each dataset
    all_results = []
    
    for dataset in args.datasets:
        # Check if dataset exists in data
        has_data = False
        if not df_units.empty and (df_units["dataset"] == dataset).any():
            has_data = True
        if not df_audit.empty and (df_audit["dataset"] == dataset).any():
            has_data = True
        
        if not has_data:
            print(f"[INFO] Dataset '{dataset}' not found in data")
            continue
        
        print(f"\n[INFO] Processing {dataset}...")
        
        results = process_dataset(dataset, df_units, df_audit, df_summary, anchors, args)
        
        if results.empty:
            print(f"[WARN] No data for {dataset} after filters")
            continue
        
        # Create plots
        title_suffix = f"{args.method}:{args.experiment}" if args.method else ""
        
        # Agreement heatmap
        p1 = _plot_agreement_heatmap(results, dataset, args.outdir, 
                                     title_suffix, args.annotate, args.dpi)
        print(f"  ✓ Created: {p1}")
        
        # Coverage heatmap if requested
        if args.draw_coverage and 'coverage' in results.columns:
            p2 = _plot_coverage_heatmap(results, dataset, args.outdir,
                                       title_suffix, args.annotate, args.dpi)
            if p2:
                print(f"  ✓ Created: {p2}")
        
        # Error distribution
        if results["has_iqr"].any():
            p3 = _plot_error_distribution(results, dataset, args.outdir, 
                                         title_suffix, args.dpi)
            if p3:
                print(f"  ✓ Created: {p3}")
        
        all_results.append(results)
    
    # Save combined summary
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        summary_path = os.path.join(args.outdir, "v3_thresholds_summary.csv")
        combined.to_csv(summary_path, index=False)
        print(f"\n[OK] Summary saved: {summary_path}")
        
        # Print enhanced summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        total_stats = {
            'total_features': 0,
            'features_with_guidelines': 0,
            'features_with_thresholds': 0,
            'green': 0,
            'yellow': 0,
            'red': 0
        }
        
        for dataset in combined["dataset"].unique():
            ds_data = combined[combined["dataset"] == dataset]
            n_features = len(ds_data)
            n_anchored = ds_data["guideline"].notna().sum()
            n_with_threshold = ds_data["median"].notna().sum()
            
            print(f"\n{dataset}:")
            print(f"  Features: {n_features} total")
            print(f"  Guidelines: {n_anchored} features with guidelines")
            print(f"  Thresholds: {n_with_threshold} features with computed thresholds")
            
            if n_anchored > 0:
                anchored = ds_data[ds_data["guideline"].notna()]
                n_green = (anchored["error_cat"] == 0).sum()
                n_yellow = (anchored["error_cat"] == 1).sum()
                n_red = (anchored["error_cat"] == 2).sum()
                
                print(f"  Agreement: {n_green} green, {n_yellow} yellow, {n_red} red")
                
                if n_with_threshold > 0:
                    success_10 = n_green / n_anchored * 100
                    success_20 = (n_green + n_yellow) / n_anchored * 100
                    print(f"  Success rate: {success_10:.1f}% within 10%, {success_20:.1f}% within 20%")
                
                # Update totals
                total_stats['total_features'] += n_features
                total_stats['features_with_guidelines'] += n_anchored
                total_stats['features_with_thresholds'] += n_with_threshold
                total_stats['green'] += n_green
                total_stats['yellow'] += n_yellow
                total_stats['red'] += n_red
                
                # Coverage info if available
                if 'coverage' in anchored.columns:
                    avg_coverage = anchored['coverage'].mean()
                    if np.isfinite(avg_coverage):
                        print(f"  Average coverage: {avg_coverage*100:.1f}%")
        
        # Overall summary
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE")
        print("="*60)
        print(f"Total features analyzed: {total_stats['total_features']}")
        print(f"Features with guidelines: {total_stats['features_with_guidelines']}")
        print(f"Features with thresholds: {total_stats['features_with_thresholds']}")
        
        if total_stats['features_with_guidelines'] > 0:
            total_assessed = total_stats['green'] + total_stats['yellow'] + total_stats['red']
            if total_assessed > 0:
                print(f"\nAgreement breakdown:")
                print(f"  Green (≤10%): {total_stats['green']} ({total_stats['green']/total_assessed*100:.1f}%)")
                print(f"  Yellow (≤20%): {total_stats['yellow']} ({total_stats['yellow']/total_assessed*100:.1f}%)")
                print(f"  Red (>20%): {total_stats['red']} ({total_stats['red']/total_assessed*100:.1f}%)")
                print(f"  Success rate: {(total_stats['green']+total_stats['yellow'])/total_assessed*100:.1f}% within 20%")

if __name__ == "__main__":
    main()