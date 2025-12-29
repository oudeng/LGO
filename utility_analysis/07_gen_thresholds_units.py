"""
Usage:
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir overall_ICU_composite_risk_score \
  --dataset ICU_composite_risk_score \
  --method lgo \
  --topk 10 \
  --experiments LGOhard,LGOsoft
  --experiments LGOhard

如果之后还想看 soft gate 或一起看，可用：
# 只看 soft gate
--experiments LGOsoft

# hard + soft 一起
--experiments LGOhard,LGOsoft
"""


import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------- helpers ---------------------------

def parse_units_yaml(path: Path):
    """Very small YAML reader for 'units.yaml' (key: value per line)."""
    units = {}
    if not path.exists():
        return units
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        if line.strip().startswith("#"):
            continue
        k, v = line.split(":", 1)
        units[k.strip()] = v.strip()
    return units


def iter_calls(expr: str, fname: str):
    """Yield (start_idx, end_idx, inside) for each fname(...) in expr."""
    if not isinstance(expr, str):
        return
    needle = fname + "("
    i = 0
    L = len(expr)
    while i < L:
        j = expr.find(needle, i)
        if j < 0:
            break
        # find matching ')'
        k = j + len(needle)
        depth = 1
        start = k
        while k < L and depth > 0:
            ch = expr[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            k += 1
        end = k - 1
        inside = expr[start:end]
        yield j, k, inside
        i = k


def guess_feature(token: str) -> str:
    """Extract feature name from first argument of gate."""
    token = str(token).strip()
    # id()/idF()
    m = re.search(r"idF?\(\s*([A-Za-z0-9_]+)\s*\)", token)
    if m:
        return m.group(1)
    # as_pos(id(...)) / as_thr(id(...))
    m = re.search(r"as_(?:pos|thr)\s*\(\s*idF?\(\s*([A-Za-z0-9_]+)\s*\)\s*\)", token)
    if m:
        return m.group(1)
    # nested as_pos(as_thr(id(...)))
    m = re.search(
        r"as_(?:pos|thr)\s*\(\s*as_(?:pos|thr)\s*\(\s*idF?\(\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*\)",
        token,
    )
    if m:
        return m.group(1)
    # as_pos(var) / as_thr(var)
    m = re.search(r"as_(?:pos|thr)\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)", token)
    if m:
        return m.group(1)
    # plain variable
    if re.match(r"^[A-Za-z][A-Za-z0-9_]*$", token):
        return token
    # fall back
    return "unknown"


def safe_eval_num(expr: str):
    """Evaluate simple numeric expression; return float or None."""
    expr = str(expr).strip()
    if not expr:
        return None
    # plain number
    if re.match(r"^[+-]?\d+(\.\d+)?$", expr):
        try:
            return float(expr)
        except Exception:
            return None
    try:
        safe_dict = {"__builtins__": {}, "np": np, "pi": np.pi, "e": np.e}
        val = float(eval(expr, safe_dict))
        if np.isfinite(val):
            return val
    except Exception:
        return None
    return None


def extract_threshold_from_gate(inside: str, gate_type: str):
    """From 'x,a,b' inside gate(x,a,b) return (feature, b_z, raw_b_token)."""
    parts = []
    buf = ""
    depth = 0
    for ch in inside:
        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
            continue
        buf += ch
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
    if buf.strip():
        parts.append(buf.strip())

    # lgo_thre / gate_expr / lgo_and2 / lgo_or2 / lgo_gate 都是 (x,a,b)
    if len(parts) != 3:
        return None, None, None

    x_tok, a_tok, b_tok = parts
    feat = guess_feature(x_tok)
    if not feat:
        feat = "unknown"

    # unwrap as_thr/as_pos
    b_inner = b_tok
    m = re.match(r"as_thr\((.*)\)$", b_inner)
    if m:
        b_inner = m.group(1)
    m = re.match(r"as_pos\((.*)\)$", b_inner)
    if m:
        b_inner = m.group(1)

    b_z = safe_eval_num(b_inner)
    return feat, b_z, b_tok


def pick_expr_column(df: pd.DataFrame):
    """Choose expression column name, matching run_v3_9_3/save_candidates_file."""
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    for cand in ["expr_str", "expr_simplified", "expr", "expression", "raw_expr", "tree_str"]:
        if cand in df.columns:
            return cand
    return None


# --------------------------- main ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--dataset", default="CUSTOM")
    p.add_argument("--method", default="lgo")
    # 这里直接用 candidates 里的 experiment 字段/文件名：base / LGOsoft / LGOhard
    p.add_argument(
        "--experiments",
        default="LGOhard",
        help="Comma-separated experiment names as in candidates (e.g. 'LGOhard,LGOsoft,base'). "
             "Default: LGOhard",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Use top-K rows per seed (sorted by cv_loss if present); if <=0, use all rows.",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    d = Path(args.dataset_dir)
    cand_dir = d / "candidates"
    ag = d / "aggregated"
    ag.mkdir(parents=True, exist_ok=True)

    # load scaler + units
    scaler_json = {}
    scaler_path = d / "config" / "scaler.json"
    if scaler_path.exists():
        scaler_json = json.loads(scaler_path.read_text())
    mus = scaler_json.get("means", {}) or scaler_json.get("feature_means", {})
    sigs = scaler_json.get("stds", {}) or scaler_json.get("feature_stds", {})
    units = parse_units_yaml(d / "config" / "units.yaml")

    # experiments filter
    exp_filter = {e.strip() for e in str(args.experiments).split(",") if e.strip()}
    if not exp_filter:
        exp_filter = {"LGOhard"}

    # 允许老版本的小写别名（不改变输出里到底写什么）
    if "LGOhard" in exp_filter:
        exp_filter.add("lgo_hard")
    if "LGOsoft" in exp_filter:
        exp_filter.add("lgo_soft")

    # candidate files for method
    cand_files = []
    if cand_dir.exists():
        patterns = [
            f"candidates_{args.method}_*_seed*.csv",
            f"candidates_{args.method}_seed*.csv",
            f"candidates_{args.method}_*.csv",
            f"candidates_{args.method}.csv",
        ]
        for pat in patterns:
            cand_files.extend(sorted(cand_dir.glob(pat)))

    if not cand_files:
        existing = []
        if cand_dir.exists():
            existing = [p.name for p in sorted(cand_dir.glob("candidates_*.csv"))]
        print(f"[WARN] no candidate files for method='{args.method}' under: {cand_dir}")
        if existing:
            print("[INFO] existing files:", ", ".join(existing[:20]))
        (ag / "thresholds_units.csv").write_text("")
        return

    # 去重
    seen = set()
    unique_files = []
    for cf in cand_files:
        if cf not in seen:
            seen.add(cf)
            unique_files.append(cf)
    cand_files = unique_files

    gate_functions = ["lgo_thre", "lgo_and2", "lgo_or2", "gate_expr", "lgo_gate"]

    rows = []
    for cf in cand_files:
        print(f"Processing: {cf.name}")
        c = pd.read_csv(cf)

        current_experiments = []

        # experiment filter
        if "experiment" in c.columns:
            # 只保留 exp 在 exp_filter 里的
            c_filtered = c[c["experiment"].isin(exp_filter)].copy()
            if len(c_filtered) == 0:
                print(f"  Skipped (no matching experiments in {exp_filter})")
                continue
            c = c_filtered
            current_experiments = sorted(c["experiment"].astype(str).unique())
        else:
            # 从文件名推断：candidates_{method}_{experiment}_seed{seed}.csv
            stem = cf.stem
            exp_name = None
            prefix = f"candidates_{args.method}_"
            if stem.startswith(prefix):
                tail = stem[len(prefix):]
                if "_seed" in tail:
                    exp_name = tail.split("_seed", 1)[0]
                else:
                    exp_name = tail
            if not exp_name:
                exp_name = "base"
            current_experiments = [exp_name]

            if not any(e in exp_filter for e in current_experiments):
                print(f"  Skipped (inferred experiment {current_experiments} not in {exp_filter})")
                continue
            # 补上 experiment 列
            c = c.copy()
            c["experiment"] = exp_name

        # 选择表达式列
        expr_col = pick_expr_column(c)
        if not expr_col:
            print(f"  Skipped (no expression column in {cf.name})")
            continue

        # 按 cv_loss 排序再取 top-K
        if "cv_loss" in c.columns and c["cv_loss"].notna().any():
            c = c.sort_values("cv_loss")
        if args.topk > 0:
            c = c.head(args.topk)

        # 扫描表达式
        for idx, row in c.iterrows():
            expr = row.get(expr_col, "")
            experiment = row.get(
                "experiment",
                current_experiments[0] if current_experiments else "unknown",
            )

            if pd.isna(expr) or not expr:
                continue
            expr = str(expr)

            if args.debug and idx < 3:
                print(f"  Expression {idx}: {expr[:100]}...")

            found_any = False
            for gate_func in gate_functions:
                for (s, eidx, inside) in iter_calls(expr, gate_func):
                    feat, b_z, b_tok_raw = extract_threshold_from_gate(inside, gate_func)
                    if feat is None:
                        continue
                    found_any = True

                    if args.debug:
                        print(f"    Found {gate_func}({feat}, ..., {b_z})")

                    mu = float(mus.get(feat, np.nan)) if feat in mus else np.nan
                    sig = float(sigs.get(feat, np.nan)) if feat in sigs else np.nan

                    if b_z is None:
                        rows.append(
                            {
                                "dataset": args.dataset,
                                "method": args.method,
                                "experiment": experiment,
                                "gate_type": gate_func,
                                "feature": feat,
                                "unit": units.get(feat, ""),
                                "mu": mu,
                                "sigma": sig,
                                "b_z": np.nan,
                                "b_raw": np.nan,
                                "note": "dynamic_b_unresolved",
                            }
                        )
                    else:
                        b_raw = (
                            mu + b_z * sig
                            if np.isfinite(mu) and np.isfinite(sig)
                            else np.nan
                        )
                        rows.append(
                            {
                                "dataset": args.dataset,
                                "method": args.method,
                                "experiment": experiment,
                                "gate_type": gate_func,
                                "feature": feat,
                                "unit": units.get(feat, ""),
                                "mu": mu,
                                "sigma": sig,
                                "b_z": b_z,
                                "b_raw": b_raw,
                                "note": "",
                            }
                        )

            if args.debug and not found_any and idx < 3:
                print("    No gates found in expression")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        (ag / "thresholds_units.csv").write_text("")
        print(f"[WARN] No thresholds extracted for experiments {exp_filter}")
        return

    print(f"\nExtracted {len(df)} threshold instances")
    print(f"Experiments found: {df['experiment'].unique()}")
    print(f"Gate types found: {df['gate_type'].unique()}")
    unique_features = df["feature"].unique()
    print(
        f"Features found ({len(unique_features)} unique): {unique_features[:15]}..."
    )

    agg = (
        df.groupby(
            [
                "dataset",
                "method",
                "experiment",
                "gate_type",
                "feature",
                "unit",
                "mu",
                "sigma",
            ],
            dropna=False,
        )
        .agg(
            count=("b_z", "size"),
            b_z_median=("b_z", "median"),
            b_z_q1=("b_z", lambda s: s.quantile(0.25)),
            b_z_q3=("b_z", lambda s: s.quantile(0.75)),
            b_raw_median=("b_raw", "median"),
            b_raw_q1=("b_raw", lambda s: s.quantile(0.25)),
            b_raw_q3=("b_raw", lambda s: s.quantile(0.75)),
            n_dynamic=("note", lambda s: int((s == "dynamic_b_unresolved").sum())),
        )
        .reset_index()
    )

    agg = agg.sort_values(["experiment", "feature", "gate_type"])
    outpath = ag / "thresholds_units.csv"
    agg.to_csv(outpath, index=False)
    print(f"[OK] Wrote thresholds_units.csv with {len(agg)} aggregated rows to {outpath}")


if __name__ == "__main__":
    main()