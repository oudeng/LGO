"""
utility_analysis.07_gen_thresholds_units

python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU_composite_risk_score \
  --method lgo --topk 10 --experiments lgo_hard

(Dec 11, 2025) add def normalize_experiment_name and others to solve the naming problem.
"""

import argparse, pandas as pd, numpy as np, json, re
from pathlib import Path

def parse_units_yaml(path):
    units = {}
    if not path.exists(): return units
    for line in path.read_text().splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k,v = line.split(":",1)
            units[k.strip()] = v.strip()
    return units

def normalize_experiment_name(exp: str) -> str:
    """
    统一实验名：
    - CLI: base, lgo_soft, lgo_hard
    - candidates: base, LGOsoft, LGOhard
    """
    if exp is None:
        return ""
    t = str(exp).strip().lower()
    t = t.replace("-", "_")
    t_compact = t.replace("_", "")

    # LGO hard threshold（原 lgo_thre / LGOhard）
    if t_compact in {"lgohard", "lgo_thre", "lgothre", "hard", "thre"}:
        return "lgo_hard"

    # LGO soft gate（原 lgo / LGOsoft）
    if t_compact in {"lgosoft", "lgo", "soft"}:
        return "lgo_soft"

    # baseline
    if t_compact in {"base", "none"}:
        return "base"

    return t

def iter_calls(expr, fname="lgo_thre"):
    """Yields (start_idx, end_idx, inside_content) for each fname(...) call.
    Supports multiple gate functions: lgo_thre, lgo_and2, lgo_or2, gate_expr, lgo_gate"""
    if not isinstance(expr, str): return
    i = 0
    L = len(expr)
    while i < L:
        j = expr.find(fname+"(", i)
        if j < 0: break
        # find matching ')'
        depth = 0; k = j+len(fname)
        # skip to first '('
        while k < L and expr[k] != '(':
            k += 1
        start = k+1
        depth = 1; k = start
        while k < L and depth > 0:
            if expr[k] == '(':
                depth += 1
            elif expr[k] == ')':
                depth -= 1
            k += 1
        end = k-1
        yield (j, end+1, expr[start:end])
        i = end+1

def guess_feature(token):
    """Enhanced feature extraction supporting various patterns"""
    # Clean the token first
    token = token.strip()
    
    # Try id() or idF() patterns first
    m = re.search(r"idF?\(\s*([A-Za-z0-9_]+)\s*\)", token)
    if m: return m.group(1)
    
    # Try as_pos(id(...)) or as_thr(id(...)) wrappers
    m = re.search(r"as_(?:pos|thr)\s*\(\s*idF?\(\s*([A-Za-z0-9_]+)\s*\)\s*\)", token)
    if m: return m.group(1)
    
    # Try nested as_pos(as_thr(...)) or as_thr(as_pos(...))
    m = re.search(r"as_(?:pos|thr)\s*\(\s*as_(?:pos|thr)\s*\(\s*idF?\(\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*\)", token)
    if m: return m.group(1)
    
    # Try wrapped in as_pos() or as_thr() without id()
    m = re.search(r"as_(?:pos|thr)\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)", token)
    if m: return m.group(1)
    
    # Try nested as_pos(as_thr(var)) or as_thr(as_pos(var))
    m = re.search(r"as_(?:pos|thr)\s*\(\s*as_(?:pos|thr)\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)\s*\)", token)
    if m: return m.group(1)
    
    # Try ARGi or Xi patterns (in case remapping didn't happen)
    m = re.search(r"\b(ARG\d+|X\d+)\b", token)
    if m: return m.group(1)
    
    # If it's just a variable name without function call
    if re.match(r"^[A-Za-z][A-Za-z0-9_]*$", token):
        return token
    
    # Try extracting from mathematical expressions (e.g., "2*chol + 1")
    # Look for variable names that aren't function names
    m = re.search(r"\b([A-Za-z][A-Za-z0-9_]*)\b", token)
    if m:
        candidate = m.group(1)
        # Exclude common function names
        if candidate not in {"as_pos", "as_thr", "id", "idF", "add", "sub", "mul", "div", 
                            "sqrt", "log", "exp", "sin", "cos", "tan", "abs", "pow"}:
            return candidate
    
    return None

def safe_eval_num(expr):
    """Try to evaluate numeric literal or simple expressions"""
    try:
        # Allow simple math operations
        safe_dict = {
            "__builtins__": {},
            "np": np,
            "pi": np.pi,
            "e": np.e
        }
        val = float(eval(expr, safe_dict))
        if np.isfinite(val): 
            return float(val)
    except Exception:
        pass
    return None

def extract_threshold_from_gate(inside, gate_type):
    """Extract threshold parameter based on gate function type.
    For lgo_thre(x, a, b): returns b (3rd param)
    For lgo_and2/lgo_or2/gate_expr/lgo_gate(x, a, b): returns b (3rd param)"""
    
    # Split top level commas in 'inside'
    parts = []; buf=""; depth=0
    for ch in inside:
        if ch == "," and depth==0:
            parts.append(buf.strip())
            buf=""
            continue
        buf += ch
        if ch=="(":
            depth += 1
        elif ch==")":
            depth -= 1
    if buf.strip(): 
        parts.append(buf.strip())
    
    # All gate functions expect 3 parameters (x, a, b)
    if len(parts) != 3: 
        return None, None, None
    
    x_tok, a_tok, b_tok = parts
    feat = guess_feature(x_tok)
    
    # If feature is still not found, try harder
    if not feat or feat == "unknown":
        # Try stripping more aggressively
        x_tok_clean = re.sub(r"[^\w]", " ", x_tok)
        for word in x_tok_clean.split():
            if re.match(r"^[A-Za-z][A-Za-z0-9_]*$", word):
                if word not in {"as", "pos", "thr", "id", "idF"}:
                    feat = word
                    break
    
    if not feat:
        feat = "unknown"
    
    # Try to get numeric b_z
    # Allow wrappers like as_thr(...)
    b_tok_inner = b_tok
    
    # Strip as_thr wrapper if present
    mthr = re.match(r"as_thr\((.*)\)$", b_tok)
    if mthr:
        b_tok_inner = mthr.group(1)
    
    # Strip as_pos wrapper if present
    mpos = re.match(r"as_pos\((.*)\)$", b_tok_inner)
    if mpos:
        b_tok_inner = mpos.group(1)
    
    b_z = safe_eval_num(b_tok_inner)
    
    return feat, b_z, b_tok

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--dataset", default="CUSTOM")
    p.add_argument("--method", default="lgo")
    p.add_argument("--experiments", default="lgo_hard", 
                   help="Comma-separated list of experiments to include (e.g., 'lgo_hard,lgo_soft'). Default: lgo_hard")
    p.add_argument("--topk", type=int, default=10, help="use top-K per seed if cv_loss is present; else all")
    p.add_argument("--debug", action="store_true", help="Enable debug output")
    args = p.parse_args()

    d = Path(args.dataset_dir); ag = d/"aggregated"; ag.mkdir(parents=True, exist_ok=True)
    
    # Parse experiments filter and normalize names (CLI -> canonical tokens)
    raw_exps = [e.strip() for e in str(args.experiments).split(",") if e.strip()]
    if not raw_exps:
        raw_exps = ["lgo_hard"]  # default

    # 统一到：{"base", "lgo_soft", "lgo_hard"}
    exp_filter = {normalize_experiment_name(e) for e in raw_exps}
    
    # Robust candidate discovery
    method = (args.method or "lgo").strip().lower()
    cand_dir = d / "candidates"
    cand_files = []
    if cand_dir.exists():
        # Try most specific patterns first, including experiment in filename
        patterns = [
            f"candidates_{method}_*_seed*.csv",  # New pattern with experiment
            f"candidates_{method}_seed*.csv",
            f"candidates_{method}_*.csv",
            f"candidates_{method}.csv",
        ]
        for pat in patterns:
            cand_files.extend(sorted(cand_dir.glob(pat)))
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in cand_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        cand_files = unique_files
        
        # If still empty: fallback to scan all and filter method substring (case-insensitive)
        if not cand_files:
            for fp in sorted(cand_dir.glob("candidates_*.csv")):
                nm = fp.name.lower()
                if f"candidates_{method}_" in nm or nm == f"candidates_{method}.csv":
                    cand_files.append(fp)
    
    if not cand_files:
        # Helpful diagnostics
        existing = []
        if cand_dir.exists():
            existing = [q.name for q in sorted(cand_dir.glob("candidates_*.csv"))]
        print(f"[WARN] no candidate files for method='{method}' under: {cand_dir}")
        if existing:
            print(f"       Found files: {existing}")
        (ag/"thresholds_units.csv").write_text("")
        return

    # Scaler + units
    scaler_json = json.loads((d/"config/scaler.json").read_text()) if (d/"config/scaler.json").exists() else {}
    mus = scaler_json.get("means", {}) or scaler_json.get("feature_means", {})
    sigs = scaler_json.get("stds", {}) or scaler_json.get("feature_stds", {})
    units = parse_units_yaml(d/"config/units.yaml")

    # Define gate functions to search for (expanded list)
    gate_functions = ["lgo_thre", "lgo_and2", "lgo_or2", "gate_expr", "lgo_gate"]
    
    rows = []
    for cf in cand_files:
        print(f"Processing: {cf.name}")
        c = pd.read_csv(cf)
        
        # Check for experiment column and filter
        if "experiment" in c.columns:
            # 先把文件里的实验名统一一下（比如 LGOhard -> lgo_hard）
            exp_norm = c["experiment"].astype(str).map(normalize_experiment_name)

            # 只保留命中 exp_filter 的行
            mask = exp_norm.isin(exp_filter)
            c_filtered = c.loc[mask].copy()

            if len(c_filtered) == 0:
                print(f"  Skipped (no matching experiments in {exp_filter})")
                continue

            # 为了后续脚本一致，直接把 experiment 列改成规范名
            c_filtered["experiment"] = exp_norm[mask].values

            c = c_filtered
            current_experiments = sorted(c["experiment"].astype(str).unique())

        else:
            # If no experiment column, infer from filename or use default
            # Updated inference logic for new naming
            if "_lgo_hard_" in cf.name or "_lgo_thre_" in cf.name:
                current_experiments = ["lgo_hard"]
            elif "_lgo_soft_" in cf.name or ("_lgo_" in cf.name and "_lgo_hard_" not in cf.name and "_lgo_thre_" not in cf.name):
                current_experiments = ["lgo_soft"]
            else:
                current_experiments = ["base"]
            
            # Check if inferred experiment is in filter
            if not any(exp in exp_filter for exp in current_experiments):
                print(f"  Skipped (inferred experiment {current_experiments} not in {exp_filter})")
                continue
        
        if "expr_str" not in c.columns:
            if "expr" in c.columns:
                c = c.rename(columns={"expr":"expr_str"})
            else:
                continue
        
        # Sort by cv_loss and take top-K
        if "cv_loss" in c.columns:
            c = c.sort_values("cv_loss").head(args.topk)
        
        # Process expressions
        for idx, row in c.iterrows():
            expr = row.get("expr_str", "")
            experiment = row.get("experiment", current_experiments[0] if current_experiments else "unknown")
            
            if pd.isna(expr) or not expr:
                continue
            
            if args.debug and idx < 3:  # Debug first few expressions
                print(f"  Expression {idx}: {expr[:100]}...")
                
            # Search for all gate functions
            found_any = False
            for gate_func in gate_functions:
                for (s, eidx, inside) in iter_calls(expr, gate_func):
                    feat, b_z, b_tok_raw = extract_threshold_from_gate(inside, gate_func)
                    
                    if feat is None:
                        continue
                    
                    found_any = True
                    if args.debug:
                        print(f"    Found {gate_func}({feat}, ..., {b_z})")
                    
                    if b_z is None:
                        # Dynamic threshold that couldn't be evaluated
                        rows.append({
                            "dataset": args.dataset, 
                            "method": args.method, 
                            "experiment": experiment,
                            "gate_type": gate_func,
                            "feature": feat,
                            "unit": units.get(feat, ""),  # Preserve empty string if no unit
                            "mu": mus.get(feat, np.nan), 
                            "sigma": sigs.get(feat, np.nan),
                            "b_z": np.nan, 
                            "b_raw": np.nan, 
                            "note": "dynamic_b_unresolved"
                        })
                    else:
                        # Static threshold
                        mu = float(mus.get(feat, np.nan))
                        sig = float(sigs.get(feat, np.nan))
                        b_raw = (mu + b_z*sig) if np.isfinite(mu) and np.isfinite(sig) else np.nan
                        rows.append({
                            "dataset": args.dataset, 
                            "method": args.method, 
                            "experiment": experiment,
                            "gate_type": gate_func,
                            "feature": feat,
                            "unit": units.get(feat, ""),  # Preserve empty string if no unit
                            "mu": mu, 
                            "sigma": sig,
                            "b_z": b_z, 
                            "b_raw": b_raw, 
                            "note": ""
                        })
            
            if args.debug and not found_any and idx < 3:
                print(f"    No gates found in expression")
    
    df = pd.DataFrame(rows)
    if len(df)==0:
        (ag/"thresholds_units.csv").write_text("")
        print(f"[WARN] No thresholds extracted for experiments {exp_filter}")
        return
    
    print(f"\nExtracted {len(df)} threshold instances")
    print(f"Experiments found: {df['experiment'].unique()}")
    print(f"Gate types found: {df['gate_type'].unique()}")
    unique_features = df['feature'].unique()
    print(f"Features found ({len(unique_features)} unique): {unique_features[:15]}...")  # Show first 15
    
    # Aggregate per (method, experiment, feature, gate_type)
    agg = df.groupby(["dataset","method","experiment","gate_type","feature","unit","mu","sigma"], 
                     dropna=False).agg(
        count=("b_z","size"),
        b_z_median=("b_z","median"),
        b_z_q1=("b_z", lambda s: s.quantile(0.25)),
        b_z_q3=("b_z", lambda s: s.quantile(0.75)),
        b_raw_median=("b_raw","median"),
        b_raw_q1=("b_raw", lambda s: s.quantile(0.25)),
        b_raw_q3=("b_raw", lambda s: s.quantile(0.75)),
        n_dynamic=("note", lambda s: int((s=="dynamic_b_unresolved").sum()))
    ).reset_index()
    
    # Sort for better readability
    agg = agg.sort_values(["experiment", "feature", "gate_type"])
    
    agg.to_csv(ag/"thresholds_units.csv", index=False)
    print(f"\n[OK] Wrote thresholds_units.csv with {len(agg)} aggregated rows")
    
    # Print summary statistics for key Cleveland features if available
    cleveland_features = ["chol", "thalach", "trestbps", "oldpeak", "age"]
    for feat in cleveland_features:
        if feat in agg["feature"].values:
            feat_stats = agg[agg["feature"] == feat]
            print(f"\n{feat} threshold statistics:")
            for _, row in feat_stats.iterrows():
                print(f"  Experiment: {row['experiment']}, Gate: {row['gate_type']}")
                if np.isfinite(row['b_raw_median']):
                    print(f"    Median (raw): {row['b_raw_median']:.2f} {row['unit']}")
                    print(f"    IQR (raw): [{row['b_raw_q1']:.2f}, {row['b_raw_q3']:.2f}]")
                else:
                    print(f"    Median (z-score): {row['b_z_median']:.2f}")
                print(f"    Count: {row['count']}")
        else:
            print(f"\n[INFO] Feature '{feat}' not found in thresholds")

if __name__ == "__main__":
    main()