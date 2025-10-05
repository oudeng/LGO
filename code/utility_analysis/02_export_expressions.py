import argparse
import pandas as pd
from pathlib import Path

def _discover_methods(dataset_dir):
    """
    Auto-discover available methods from candidates filenames and overall_metrics.csv.
    Returns a sorted list of method names.
    """
    d = Path(dataset_dir)
    methods = set()
    
    # Try to discover from candidates directory
    cand_dir = d / "candidates"
    if cand_dir.exists():
        for p in cand_dir.glob("candidates_*.csv"):
            name = p.stem  # Remove .csv extension
            # Try multiple patterns
            if name.startswith("candidates_"):
                remaining = name[len("candidates_"):]
                # Handle: candidates_{method}_seed{N}, candidates_{method}_{anything}, candidates_{method}
                if "_seed" in remaining:
                    method = remaining.split("_seed")[0]
                elif "_" in remaining:
                    method = remaining.split("_")[0]
                else:
                    method = remaining
                if method:
                    methods.add(method)
    
    # Also try overall_metrics.csv as fallback
    om = d / "aggregated" / "overall_metrics.csv"
    if om.exists():
        try:
            m = pd.read_csv(om)
            if "method" in m.columns:
                for v in m["method"].dropna().unique():
                    methods.add(str(v))
        except Exception:
            pass
    
    return sorted(methods)

def _find_candidate_files(dataset_dir, method):
    """
    Find candidate files for a given method with multiple pattern fallbacks.
    """
    d = Path(dataset_dir) / "candidates"
    if not d.exists():
        return []
    
    files = []
    
    # Priority 1: candidates_{method}_seed*.csv
    files = list(d.glob(f"candidates_{method}_seed*.csv"))
    if files:
        return sorted(files)
    
    # Priority 2: candidates_{method}_*.csv
    files = list(d.glob(f"candidates_{method}_*.csv"))
    if files:
        return sorted(files)
    
    # Priority 3: candidates_{method}.csv
    exact_file = d / f"candidates_{method}.csv"
    if exact_file.exists():
        return [exact_file]
    
    # Priority 4: Case-insensitive substring match
    all_candidates = list(d.glob("candidates_*.csv"))
    method_lower = method.lower()
    for f in all_candidates:
        name_lower = f.stem.lower()
        if f"candidates_{method_lower}_" in name_lower or name_lower == f"candidates_{method_lower}":
            files.append(f)
    
    return sorted(files)

def _pick_expr_column(df):
    """
    Find the expression column with fallback options.
    """
    expr_columns = ["expr_str", "expr", "equation", "model", "expression", "Expression", "Equation"]
    for col in expr_columns:
        if col in df.columns:
            return col
    return None

def _pick_loss_column(df):
    """
    Find the loss column for sorting with fallback options.
    """
    loss_columns = ["cv_loss", "loss", "test_loss", "val_loss", "error", "mse", "rmse"]
    for col in loss_columns:
        if col in df.columns:
            return col
    # If no loss column, try complexity as fallback
    if "complexity" in df.columns:
        return "complexity"
    return None

def _parse_methods(method_str, dataset_dir):
    """
    Parse method string which can be 'all', single method, or comma-separated list.
    """
    if method_str.lower() == "all":
        return _discover_methods(dataset_dir)
    elif "," in method_str:
        return [m.strip() for m in method_str.split(",")]
    else:
        return [method_str]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--method", required=True, help="Method name, 'all', or comma-separated list")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--dataset", default="CUSTOM")
    
    args = p.parse_args()
    d = Path(args.dataset_dir)
    
    # Parse methods
    methods = _parse_methods(args.method, args.dataset_dir)
    if not methods:
        print("[WARN] No methods found in dataset directory.")
        return
    
    # Collect all rows across all methods
    all_rows = []
    top1_entries = []
    
    for method in methods:
        cand_files = _find_candidate_files(args.dataset_dir, method)
        if not cand_files:
            print(f"[WARN] No candidate files found for method: {method}")
            continue
        
        method_rows = []
        for cf in cand_files:
            try:
                c = pd.read_csv(cf)
                if len(c) == 0:
                    continue
                
                # Find expression column
                expr_col = _pick_expr_column(c)
                if not expr_col:
                    print(f"[WARN] No expression column found in {cf.name}")
                    continue
                
                # Find loss column for sorting
                loss_col = _pick_loss_column(c)
                if loss_col and loss_col in c.columns:
                    c = c.sort_values(loss_col).head(args.topk)
                else:
                    c = c.head(args.topk)
                
                # Extract seed from filename if possible
                seed = 0
                if "_seed" in cf.stem:
                    try:
                        seed = int(cf.stem.split("_seed")[-1])
                    except:
                        seed = 0
                
                # Get experiment if exists
                experiment = c["experiment"].iloc[0] if "experiment" in c.columns else ""
                
                for idx, row in c.iterrows():
                    method_rows.append({
                        "dataset": args.dataset,
                        "method": method,
                        "experiment": experiment,
                        "rank": len(method_rows) + 1,
                        "expr": row.get(expr_col, ""),
                        "complexity": row.get("complexity", ""),
                        "cv_metric": row.get(loss_col, "") if loss_col else "",
                        "seed": seed
                    })
            except Exception as e:
                print(f"[WARN] Error processing {cf.name}: {e}")
                continue
        
        if method_rows:
            all_rows.extend(method_rows)
            # Store top1 for this method
            top1_entries.append({
                "method": method,
                "experiment": method_rows[0].get("experiment", ""),
                "expr": method_rows[0]["expr"]
            })
            print(f"[OK] Processed {len(method_rows)} expressions for method: {method}")
    
    # Save combined results
    if all_rows:
        outdir = d / "expressions"
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Save all topk expressions in one file
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(outdir / "topk_expressions.csv", index=False)
        print(f"[OK] topk_expressions.csv saved with {len(all_rows)} total expressions.")
        
        # Save top1 expressions text file with all methods
        if top1_entries:
            with open(outdir / "top1_expressions.txt", "w") as f:
                for entry in top1_entries:
                    method_tag = f"[{entry['method'].upper()}"
                    if entry['experiment']:
                        method_tag += f"|{entry['experiment'].upper()}"
                    method_tag += "]"
                    f.write(f"{method_tag}\n")
                    f.write(f"{entry['expr']}\n\n")
            print(f"[OK] top1_expressions.txt saved with {len(top1_entries)} methods.")
    else:
        print("[WARN] No expressions found to save.")

if __name__ == "__main__":
    main()