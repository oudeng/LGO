#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_v3_8_2.py
on Dec 31, 2025

"""

import sys, argparse, json, time, inspect, re, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# =============================================================================
# 将 exp_engins 子文件夹添加到 Python 路径
# 所有引擎模块（LGO_v2_1, PySR_v2, Operon_v2, lgo_v3 等）都在 exp_engins 目录中
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXP_ENGINES_DIR = _SCRIPT_DIR / "exp_engins"

# 确保 exp_engins 在 sys.path 最前面
if str(_EXP_ENGINES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_ENGINES_DIR))

# 诊断输出（可通过环境变量关闭）
import os
if os.environ.get("LGO_DEBUG", "0") == "1":
    print(f"[DEBUG] Script dir: {_SCRIPT_DIR}")
    print(f"[DEBUG] Engines dir: {_EXP_ENGINES_DIR}")
    print(f"[DEBUG] Engines dir exists: {_EXP_ENGINES_DIR.exists()}")
    if _EXP_ENGINES_DIR.exists():
        print(f"[DEBUG] Contents: {list(_EXP_ENGINES_DIR.iterdir())[:10]}")
    print(f"[DEBUG] sys.path[0]: {sys.path[0]}")

# project helpers - 从 exp_engins/lgo_v3 导入
try:
    from lgo_v3.instrumentation import save_scaler, save_units, save_splits, save_hparams, append_runtime_profile
    from lgo_v3.metrics import classification_metrics
except ImportError as e:
    print(f"[WARN] Failed to import lgo_v3 helpers: {e}")
    print(f"[WARN] Expected location: {_EXP_ENGINES_DIR / 'lgo_v3'}")
    # 定义占位函数，防止脚本完全崩溃
    def save_scaler(*args, **kwargs): pass
    def save_units(*args, **kwargs): pass
    def save_splits(*args, **kwargs): pass
    def save_hparams(*args, **kwargs): pass
    def append_runtime_profile(*args, **kwargs): pass
    def classification_metrics(y_true, y_prob):
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except: auroc = 0.5
        try:
            auprc = average_precision_score(y_true, y_prob)
        except: auprc = float(np.mean(y_true))
        try:
            brier = brier_score_loss(y_true, y_prob)
        except: brier = 0.25
        return {"AUROC": auroc, "AUPRC": auprc, "Brier": brier}

# ====================== Global constants ======================
EPS = 1e-12
CONSIST_TOL = 1e-2     # internal vs external RMSE tolerance
SENTINEL_ABS = 1e6     # anomaly for raw predictions
HUGE_MAG = 1e4         # anomaly magnitude

# ====================== Safe math primitives ======================
def _as_arr(x): return np.asarray(x, dtype=float)

def sdiv(a, b):
    a = _as_arr(a); b = _as_arr(b)
    b = np.where(np.abs(b) > EPS, b, np.sign(b) * EPS + (b == 0) * EPS)
    with np.errstate(all="ignore"):
        out = a / b
    return np.where(np.isfinite(out), out, 0.0)

def ssqrt(a):
    a = _as_arr(a)
    with np.errstate(all="ignore"):
        out = np.sqrt(np.maximum(a, 0.0))
    return np.where(np.isfinite(out), out, 0.0)

def slog(a):
    a = _as_arr(a)
    with np.errstate(all="ignore"):
        out = np.log(np.clip(np.abs(a), EPS, None))
    return np.where(np.isfinite(out), out, 0.0)

def spow(a, b):
    a = _as_arr(a); b = _as_arr(b)
    with np.errstate(all="ignore"):
        out = np.sign(a) * (np.abs(a) ** np.clip(b, -5, 5))
    return np.where(np.isfinite(out), out, 0.0)

def ssigmoid(z):
    z = _as_arr(z); z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def srelu(x): return np.maximum(_as_arr(x), 0.0)
def sclip(x, lo, hi): return np.clip(_as_arr(x), _as_arr(lo), _as_arr(hi))
def swhere(cond, x, y):
    c = _as_arr(cond)
    if c.dtype != bool: c = c > 0
    return np.where(c, _as_arr(x), _as_arr(y))
def ssign(x): return np.sign(_as_arr(x))
def sneg(x):  return -_as_arr(x)
def sif_else(c, a, b):
    c = _as_arr(c)
    if c.dtype != bool: c = c > 0
    return np.where(c, _as_arr(a), _as_arr(b))

# ====================== LGO family (LGO_v2 semantics) ======================
def lgo(x, a, b): return x * ssigmoid(a * (x - b))
def lgo_thre(x, a, b): return ssigmoid(a * (x - b))
def lgo_pair(x, y, a, b): return (x * y) * ssigmoid(a * ((x - y) - b))
def lgo_and2(x, y, a, b):
    s1 = ssigmoid(a * (x - b)); s2 = ssigmoid(a * (y - b))
    return (x * y) * (s1 * s2)
def lgo_or2(x, y, a, b):
    s1 = ssigmoid(a * (x - b)); s2 = ssigmoid(a * (y - b))
    gate = 1.0 - (1.0 - s1) * (1.0 - s2)
    return (x + y) * gate
def lgo_and3(x, y, z, a, b):
    s1 = ssigmoid(a * (x - b)); s2 = ssigmoid(a * (y - b)); s3 = ssigmoid(a * (z - b))
    return (x * y * z) * (s1 * s2 * s3)
def gate_expr(expr, a, b): return expr * ssigmoid(a * (expr - b))

# typed helpers consistent with LGO_v2
def as_pos_func(z):
    z = _as_arr(z); z = np.clip(z, -60.0, 60.0)
    return np.log1p(np.exp(z))
def as_thr_func(z):
    z = _as_arr(z); return np.clip(z, -3.0, 3.0)

# -------------------------- Utilities --------------------------
def _clip_to_train_range(yhat, y_tr):
    lo, hi = float(np.nanmin(y_tr)), float(np.nanmax(y_tr))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan_to_num(yhat, nan=float(np.nanmean(y_tr)))
    return np.clip(yhat, lo, hi)

def _metrics_from_raw(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]
    y_pred = np.nan_to_num(y_pred, nan=float(np.nanmean(y_true)))
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae  = float(np.mean(np.abs(diff)))
    ybar = float(np.mean(y_true))
    sst  = float(np.sum((y_true - ybar)**2))
    r2   = 0.0 if sst <= 1e-12 else float(1.0 - np.sum(diff**2)/sst)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def _remap_vars_any(expr, feature_names):
    """Map ARGi/Xi/xi/x_i -> real feature names for readability and parser robustness."""
    if not isinstance(expr, str) or not feature_names: return expr
    out = expr
    for i in range(len(feature_names)-1, -1, -1):
        nm = feature_names[i]
        patterns = [rf'\\bARG{i}\\b', rf'(?<![A-Za-z0-9_])X{i}(?![A-Za-z0-9_])', rf'\\bx{i}\\b', rf'\\bx_{i}\\b']
        for pat in patterns:
            out = re.sub(pat, nm, out)
    return out

def _build_namespace(X_mat, feature_names, zscore=False, scaler_map=None):
    ns = {}
    means = (scaler_map or {}).get("feature_means", {})
    stds  = (scaler_map or {}).get("feature_stds", {})
    for i, name in enumerate(feature_names):
        col = X_mat[:, i]
        if zscore and name in means and name in stds:
            sd = stds.get(name, 1.0) or 1.0
            col = (col - means.get(name, 0.0)) / sd
        ns[name] = col
    return ns

def _make_ns_for_lgo(X_mat, feature_names, zscore=True, scaler_map=None):
    ns = _build_namespace(X_mat, feature_names, zscore=zscore, scaler_map=scaler_map)
    # LGO_v2 primitives + typed helpers
    ns.update({
        "add2": lambda a,b: a+b, "sub2": lambda a,b: a-b, "mul2": lambda a,b: a*b, "pow2": spow,
        "add":  lambda a,b: a+b, "sub":  lambda a,b: a-b, "mul":  lambda a,b: a*b,
        "div":  sdiv, "sdiv": sdiv,
        "ssqrt": ssqrt, "sqrt": ssqrt, "slog": slog, "log": slog, "pow": spow,
        "id": lambda x: x, "idF": lambda x: x,
        "as_pos": as_pos_func, "as_thr": as_thr_func,
        "tf2feat": lambda x: x, "epi2feat": lambda x: x, "env2feat": lambda x: x,
        "tf_to_feat": lambda x: x, "epi_to_feat": lambda x: x, "env_to_feat": lambda x: x,
        "lgo": lgo, "lgo_thre": lgo_thre, "lgo_pair": lgo_pair,
        "lgo_and2": lgo_and2, "lgo_or2": lgo_or2, "lgo_and3": lgo_and3,
        "gate_expr": gate_expr,
        "lgo_and2_tf_epi": lgo_and2, "lgo_or2_tf_epi": lgo_or2,
        "lgo_and2_tf_env": lgo_and2, "lgo_or2_tf_env": lgo_or2,
        "lgo_and2_epi_env": lgo_and2, "lgo_or2_epi_env": lgo_or2,
        "lgo_cross": lambda x, y, a, b: x * ssigmoid(a * (y - b)),
        "zero": 0.0, "one": 1.0,
        "ssigmoid": ssigmoid, "np": np, "math": math,
    })
    return ns

def _make_ns_generic(X_mat, feature_names):
    ns = _build_namespace(X_mat, feature_names, zscore=False, scaler_map=None)
    ns.update({
        "add": lambda a,b: a+b, "sub": lambda a,b: a-b, "mul": lambda a,b: a*b, "div": sdiv,
        "sqrt": ssqrt, "log": slog, "pow": spow, "abs": lambda x: np.abs(_as_arr(x)),
        "exp": lambda x: np.exp(np.clip(_as_arr(x), -60, 60)),
        "expm1": lambda x: np.expm1(np.clip(_as_arr(x), -60, 60)), "log1p": lambda x: np.log1p(np.clip(_as_arr(x), -1+EPS, None)),
        "sin": lambda x: np.sin(_as_arr(x)), "cos": lambda x: np.cos(_as_arr(x)), "tan": lambda x: np.tan(_as_arr(x)),
        "tanh": lambda x: np.tanh(_as_arr(x)), "sinh": lambda x: np.sinh(_as_arr(x)), "cosh": lambda x: np.cosh(_as_arr(x)),
        "erf": lambda x: math.erf(float(x)) if np.ndim(x)==0 else np.vectorize(math.erf)(_as_arr(x)),
        "sign": ssign, "relu": srelu, "clip": sclip, "where": swhere, "max": np.maximum, "min": np.minimum,
        "gt": lambda a,b: _as_arr(a) > _as_arr(b), "lt": lambda a,b: _as_arr(a) < _as_arr(b),
        "ge": lambda a,b: _as_arr(a) >= _as_arr(b), "le": lambda a,b: _as_arr(a) <= _as_arr(b),
        "eq": lambda a,b: _as_arr(a) == _as_arr(b), "ne": lambda a,b: _as_arr(a) != _as_arr(b),
        "neg": sneg, "if_else": sif_else, "if_then_else": sif_else,
        "np": np, "math": math,
        "zero": 0.0, "one": 1.0,
    })
    return ns

def _make_ns_for_pstree_rils(X_mat, feature_names, scaler_map):
    """Namespace for PSTree/RILS: features standardized by TRAIN-only scaler."""
    ns = _build_namespace(X_mat, feature_names, zscore=True, scaler_map=scaler_map)
    ops = _make_ns_generic(np.zeros((1, len(feature_names))), feature_names)
    for k, v in ops.items():
        if k not in feature_names:  # don't overwrite the feature arrays
            ns[k] = v
    return ns

def _clean_expr_text(expr):
    s = str(expr).replace("^", "**").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick_expr_column(cand: pd.DataFrame):
    if not isinstance(cand, pd.DataFrame) or len(cand)==0: return None
    preferred = ["expr","expr_str","equation","Expression","model","model_str","expr_stripped"]
    for col in preferred:
        if col in cand.columns and cand[col].notna().any():
            vals = cand[col].dropna().astype(str)
            if len(vals)>0: return col
    obj_cols = [c for c in cand.columns if cand[c].dtype=="O"]
    return obj_cols[0] if obj_cols else None

def _parse_predictions_from_engine(raw_data):
    """Parse y_pred_test from engine output (handles list/array/json formats)."""
    if raw_data is None:
        return None
    if isinstance(raw_data, (list, np.ndarray)):
        return np.asarray(raw_data, dtype=float).ravel()
    if isinstance(raw_data, str):
        try:
            parsed = json.loads(raw_data)
            return np.asarray(parsed, dtype=float).ravel()
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(raw_data)
                return np.asarray(parsed, dtype=float).ravel()
            except Exception:
                return None
    return None

# ------------------------ Engines ------------------------
# 所有引擎模块都在 exp_engins 目录中
# LGO 主引擎：仍然用 v2_1（验证过稳定）
try:
    from LGO_v2_1 import run_lgo_sr_v2 as run_lgo_sr
except Exception as e:
    run_lgo_sr = None
    if os.environ.get("LGO_DEBUG", "0") == "1":
        print(f"[DEBUG] LGO_v2_1 import failed: {e}")
        print(f"[DEBUG] Expected at: {_EXP_ENGINES_DIR / 'LGO_v2_1.py'}")

# 使用 SRBench 对齐的 v2 版本（约 500k evaluations）
try:
    from PySR_v2 import run_pysr_sr_v2 as run_pysr_sr
except Exception as e:
    run_pysr_sr = None
    if os.environ.get("LGO_DEBUG", "0") == "1":
        print(f"[DEBUG] PySR_v2 import failed: {e}")

try:
    from Operon_v2 import run_operon_sr_v2 as run_operon_sr
except Exception as e:
    run_operon_sr = None
    if os.environ.get("LGO_DEBUG", "0") == "1":
        print(f"[DEBUG] Operon_v2 import failed: {e}")

# 可选：PSTree / RILS-ROLS（保留原来的 try/except，这样即使没装也只是打印 warning）
try:
    from PSTree_v2_2 import run_pstree_once as _pst_run
    from PSTree_v2_2 import PSTreeConfig as _PSTCfg
except Exception:
    _pst_run, _PSTCfg = None, None

try:
    from RILS_ROLS_v2_1 import run_rils_rols_once as _rils_run
    from RILS_ROLS_v2_1 import RILSROLSConfig as _RILSCfg
except Exception:
    _rils_run, _RILSCfg = None, None

ENGINES = {
    "lgo":      run_lgo_sr,
    "pysr":     run_pysr_sr,
    "operon":   run_operon_sr,
    "pstree":   _pst_run,
    "rils_rols": _rils_run,
}

# ---------------------- Engine wrappers ----------------------
def _call_lgo(X, y, feature_names, experiment, hparams):
    if run_lgo_sr is None:
        print(f"[ERROR] lgo engine not importable in this env.")
        print(f"[ERROR] Expected LGO_v2_1.py at: {_EXP_ENGINES_DIR / 'LGO_v2_1.py'}")
        print(f"[ERROR] To debug, run with: LGO_DEBUG=1 python run_v3_8_2.py ...")
        return pd.DataFrame()
    candidate_kwargs = {
        "X": X, "y": y, "feature_names": feature_names, "experiment": experiment,
        "use_zscore": True,
        "pop_size": int(hparams.get("pop_size", 600)),
        "ngen": int(hparams.get("ngen", hparams.get("generations", 80))),
        "tournament_size": int(hparams.get("tournament_size", 7)),
        "cx_pb": float(hparams.get("cx_pb", 0.8)),
        "mut_pb": float(hparams.get("mut_pb", 0.2)),
        "max_height": int(hparams.get("max_height", 10)),
        "hof_size": int(hparams.get("hof_size", 20)),
        "topk_cv": int(hparams.get("topk_cv", 12)),
        "topk_local_opt": int(hparams.get("topk_local_opt", 6)),
        "local_opt_steps": int(hparams.get("local_opt_steps", 60)),
        "micro_mutation_prob": float(hparams.get("micro_mutation_prob", 0.10)),
        "cv_proxy_weight": float(hparams.get("cv_proxy_weight", 0.0)),
        "cv_proxy_weight_final": hparams.get("cv_proxy_weight_final", None),
        "cv_proxy_warmup_frac": float(hparams.get("cv_proxy_warmup_frac", 0.8)),
        "cv_proxy_subsample": float(hparams.get("cv_proxy_subsample", 0.30)),
        "cv_proxy_folds": int(hparams.get("cv_proxy_folds", 2)),
        "typed_mode": str(hparams.get("typed_mode", "light")),
        "typed_grouping": str(hparams.get("typed_grouping", "none")),
        "include_lgo_pair": bool(hparams.get("include_lgo_pair", False)),
        "include_lgo_cross_old": bool(hparams.get("include_lgo_cross_old", False)),
        "include_lgo_multi": bool(hparams.get("include_lgo_multi", True)),
        "include_lgo_and3": bool(hparams.get("include_lgo_and3", True)),
        "enable_gate_expr": bool(hparams.get("gate_expr_enable", False)),
        "random_state": int(hparams.get("rand_state", hparams.get("seed", 0))),
        "X_test": hparams.get("X_test", None),"y_test": hparams.get("y_test", None),
    }
    sig = inspect.signature(run_lgo_sr).parameters
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig}
    dropped = [k for k in candidate_kwargs if k not in sig]
    print(f"[LGO] accepted kwargs: {sorted(kwargs.keys())}")
    if dropped: print(f"[LGO] dropped kwargs (not in signature): {sorted(dropped)}")
    try:
        return run_lgo_sr(**kwargs)
    except Exception as e:
        print(f"[ERROR] lgo call failed: {e}")
        return pd.DataFrame()

def _call_general_engine(engine_fn, X, y, feature_names, experiment):
    if engine_fn is None: return pd.DataFrame()
    try:
        sig = inspect.signature(engine_fn).parameters
    except Exception:
        sig = {}
    kwargs = {}
    if "feature_names" in sig: kwargs["feature_names"] = feature_names
    if "experiment" in sig: kwargs["experiment"] = experiment
    try:
        return engine_fn(X, y, **kwargs)
    except TypeError:
        try: return engine_fn(X, y, feature_names=feature_names)
        except Exception: return pd.DataFrame()

def _call_pstree(X_tr, y_tr, X_te, y_te, seed, hparams, feature_names):
    if _pst_run is None or _PSTCfg is None:
        print("[ERROR] PSTree modules not importable. Activate pstree env.")
        return pd.DataFrame()
    # Signature-safe PSTreeConfig
    try:
        cfg_sig = inspect.signature(_PSTCfg).parameters
    except Exception:
        cfg_sig = {}
    candidate = {
        "seed": seed,
        "max_depth": int(hparams.get("pst_max_depth", hparams.get("max_depth", 6))),
        "max_leaf_nodes": int(hparams.get("pst_max_leaf_nodes", hparams.get("max_leaf_nodes", 4))),
        "min_samples_leaf": int(hparams.get("pst_min_samples_leaf", hparams.get("min_samples_leaf", 20))),
        "pop_size": int(hparams.get("pst_pop_size", hparams.get("pop_size", 100))),
        "ngen": int(hparams.get("pst_ngen", hparams.get("ngen", 60))),
        "timeout_sec": hparams.get("pst_timeout_sec", hparams.get("timeout_sec", None)),
    }
    cfg = _PSTCfg(**{k:v for k,v in candidate.items() if k in cfg_sig})
    try:
        # FIX: pass seed, cfg, feature_names in correct order
        row = _pst_run(X_tr, y_tr, X_te, y_te, seed, cfg, feature_names)
        return pd.DataFrame([row]) if isinstance(row, dict) else (row if isinstance(row, pd.DataFrame) else pd.DataFrame())
    except Exception as e:
        print(f"[ERROR] PSTree call failed: {e}"); return pd.DataFrame()

def _call_rils(X_tr, y_tr, X_te, y_te, seed, hparams, feature_names):
    if _rils_run is None or _RILSCfg is None:
        print("[ERROR] RILS-ROLS modules not importable. Activate rils env.")
        return pd.DataFrame()
    # Signature-safe RILSROLSConfig (v2)
    try:
        cfg_sig = inspect.signature(_RILSCfg).parameters
    except Exception:
        cfg_sig = {}
    candidate = {
        "seed": seed,
        "max_fit_calls": int(hparams.get("max_fit_calls", 100000)),
        "max_time": int(hparams.get("max_time", 100)),
        "complexity_penalty": float(hparams.get("complexity_penalty", 1e-3)),
        "max_complexity": int(hparams.get("max_complexity", 50)),
        "sample_size": float(hparams.get("sample_size", 1.0)),
        "verbose": bool(hparams.get("verbose", False)),
        "timeout_sec": hparams.get("timeout_sec", None),
    }
    cfg = _RILSCfg(**{k:v for k,v in candidate.items() if k in cfg_sig})
    try:
        row = _rils_run(X_tr, y_tr, X_te, y_te, seed, cfg, feature_names)
        return pd.DataFrame([row]) if isinstance(row, dict) else (row if isinstance(row, pd.DataFrame) else pd.DataFrame())
    except Exception as e:
        print(f"[ERROR] RILS call failed: {e}"); return pd.DataFrame()

# --------------------------- Self-check logging ---------------------------
def _append_selfcheck(outdir: Path, rows: list):
    sc_path = outdir / "aggregated" / "selfcheck_report.csv"
    outdir.joinpath("aggregated").mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(sc_path, mode="a", header=not sc_path.exists(), index=False)

# ====================== NEW: Candidates saving functions (v3.7) ======================
def save_candidates_file(cand, method, experiment, seed, outdir, feature_names):
    """
    Save candidates file for compatibility with analysis pipeline.
    Returns the top expression if available.
    """
    cdir = outdir / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"candidates_{method}_{experiment}_seed{seed}.csv"
    
    top_expr = None
    expr_col = None
    
    if isinstance(cand, pd.DataFrame) and len(cand) > 0:
        # Find expression column
        expr_col = _pick_expr_column(cand)
        
        # Remap variables for lgo
        if method == "lgo" and expr_col and expr_col in cand.columns:
            cand[expr_col] = cand[expr_col].apply(lambda x: _remap_vars_any(str(x), feature_names) if pd.notna(x) else "")
        
        # Add required metadata columns
        if "experiment" not in cand.columns:
            cand["experiment"] = experiment
        if "seed" not in cand.columns:
            cand["seed"] = seed
        if "method" not in cand.columns:
            cand["method"] = method
            
        # Extract top expression before saving
        if expr_col:
            ctmp = cand.copy()
            # Sort by test_loss or cv_loss
            if "test_loss" in ctmp.columns and ctmp["test_loss"].notna().any():
                ctmp = ctmp.sort_values("test_loss")
            elif "cv_loss" in ctmp.columns and ctmp["cv_loss"].notna().any():
                ctmp = ctmp.sort_values("cv_loss")
            
            vals = ctmp[expr_col].dropna().astype(str)
            if len(vals) > 0:
                top_expr = str(vals.iloc[0])
        
        # Save candidates
        cand.to_csv(cpath, index=False)
        print(f"[v3.8] Saved {len(cand)} candidates to {cpath}")
    else:
        # Create minimal candidates file
        empty_df = pd.DataFrame({
            "expr_str": [""],
            "experiment": [experiment],
            "seed": [seed],
            "method": [method],
            "complexity": [np.nan],
            "cv_loss": [np.nan],
            "test_loss": [np.nan]
        })
        empty_df.to_csv(cpath, index=False)
        print(f"[v3.8] Saved empty candidates file to {cpath}")
    
    return top_expr, expr_col

def save_expressions_summary(cand, top_expr, method, experiment, seed, outdir, args):
    """
    Save expression summary to aggregated/expressions.csv for analysis scripts.
    """
    ag = outdir / "aggregated"
    ag.mkdir(parents=True, exist_ok=True)
    expr_path = ag / "expressions.csv"
    
    # Prepare expression row
    expr_row = {
        "dataset": args.dataset,
        "method": method,
        "experiment": experiment,
        "seed": seed,
        "expr": top_expr if top_expr else "",
        "complexity": np.nan,
        "cv_loss": np.nan,
        "test_loss": np.nan,
        "cv_r2": np.nan,
        "test_r2": np.nan
    }
    
    # Extract metrics from candidates if available
    if isinstance(cand, pd.DataFrame) and len(cand) > 0:
        if "complexity" in cand.columns:
            expr_row["complexity"] = cand["complexity"].iloc[0]
        if "cv_loss" in cand.columns:
            expr_row["cv_loss"] = cand["cv_loss"].iloc[0]
        if "test_loss" in cand.columns:
            expr_row["test_loss"] = cand["test_loss"].iloc[0]
        if "cv_r2" in cand.columns:
            expr_row["cv_r2"] = cand["cv_r2"].iloc[0]
        if "test_r2" in cand.columns:
            expr_row["test_r2"] = cand["test_r2"].iloc[0]
    
    # Save to expressions.csv
    pd.DataFrame([expr_row]).to_csv(expr_path, mode="a", header=not expr_path.exists(), index=False)
    print(f"[v3.8] Updated expressions summary: {expr_path}")

# --------------------------- Main runner ---------------------------
def run_once(args, method, experiment, seed):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    assert args.target in df.columns, "target not in CSV"
    feature_names = [c for c in df.columns if c != args.target]
    X = df[feature_names].values.astype(float)
    y = df[args.target].values.astype(float)

    # Save splits/scaler/units (metadata only) once
    if seed == args.seeds_list[0]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) if args.task!="regression" \
              else KFold(n_splits=5, shuffle=True, random_state=seed)
        folds = [{"train_idx": tr.tolist(), "test_idx": te.tolist()} for tr,te in skf.split(X, y if args.task!="regression" else None)]
        save_splits(outdir, "KFold" if args.task=="regression" else "StratifiedKFold", None, folds)
        mu_meta = {c: float(df[c].mean()) for c in feature_names}
        sd_meta = {c: float(df[c].std(ddof=0)) for c in feature_names}
        save_scaler(outdir, mu_meta, sd_meta, True)
        unit_map = json.loads(args.unit_map_json) if args.unit_map_json else {c: "" for c in feature_names}
        save_units(outdir, unit_map)

    hparams = json.loads(args.hparams_json) if args.hparams_json.strip() else {}
    save_hparams(outdir, method, seed, 0, hparams)

    # Holdout split
    if args.task == "binary_classification":
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=seed, stratify=y)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=seed)

    # TRAIN-ONLY scaler for LGO/PSTree/RILS external eval
    scaler_map_train = {
        "feature_means": {feature_names[i]: float(X_tr[:, i].mean()) for i in range(len(feature_names))},
        "feature_stds":  {feature_names[i]: float(X_tr[:, i].std(ddof=0)) for i in range(len(feature_names))},
    }

    print(f"[{method}|{experiment}|seed={seed}] Starting engine call...")
    t0 = time.time()
    if method == "lgo":
        hparams["X_test"] = X_te; hparams["y_test"] = y_te
        cand = _call_lgo(X_tr, y_tr, feature_names, experiment, hparams)
    elif method == "pstree":
        cand = _call_pstree(X_tr, y_tr, X_te, y_te, seed, hparams, feature_names)
    elif method == "rils_rols":
        cand = _call_rils(X_tr, y_tr, X_te, y_te, seed, hparams, feature_names)
    else:
        cand = _call_general_engine(ENGINES[method], X_tr, y_tr, feature_names, experiment)
    t1 = time.time()
    print(f"[{method}|{experiment}|seed={seed}] Engine returned {len(cand) if isinstance(cand, pd.DataFrame) else 0} candidates in {t1-t0:.2f}s")
    append_runtime_profile(outdir, method, seed, 0, [{"phase":"fit","duration_s": t1-t0}])

    # ====================== v3.7: SAVE CANDIDATES ======================
    #top_expr, expr_col = save_candidates_file(cand, method, experiment, seed, outdir, feature_names)
    #save_expressions_summary(cand, top_expr, method, experiment, seed, outdir, args)
    # ==================================================================

    # ====================== v3.8: SAVE CANDIDATES ======================
    try:
        top_expr, expr_col = save_candidates_file(cand, method, experiment, seed, outdir, feature_names)
        save_expressions_summary(cand, top_expr, method, experiment, seed, outdir, args)
    except Exception as e:
        print(f"[WARN] saving candidates/expressions failed for {method}|{experiment}|seed={seed}: {e}")
        top_expr, expr_col = None, None  # 让后续走安全兜底

    ag = outdir/"aggregated"; ag.mkdir(parents=True, exist_ok=True)
    preds_dir = outdir/"predictions"; preds_dir.mkdir(parents=True, exist_ok=True)
    om_rows = []; sc_rows = []

    if args.task == "regression":
        # Re-determine top expression from saved candidates (for consistency)
        if top_expr is None and isinstance(cand, pd.DataFrame) and len(cand) > 0:
            expr_col = _pick_expr_column(cand)
            if expr_col:
                ctmp = cand.copy()
                sort_key = "test_loss" if "test_loss" in ctmp.columns and ctmp["test_loss"].notna().any() else ("cv_loss" if "cv_loss" in ctmp.columns else None)
                if sort_key: ctmp = ctmp.sort_values(sort_key)
                vals = ctmp[expr_col].dropna().astype(str)
                if len(vals) > 0: 
                    top_expr = str(vals.iloc[0])
                    chosen_row = ctmp.iloc[0:1]
                else:
                    chosen_row = None
            else:
                chosen_row = None
        elif isinstance(cand, pd.DataFrame) and len(cand) > 0:
            chosen_row = cand.iloc[0:1]
        else:
            chosen_row = None

        if top_expr is None:
            print(f"[WARN|{method}|{experiment}|seed={seed}] No expression; fallback to mean predictor.")
            y_pred_raw = np.full_like(y_te, float(np.mean(y_tr)), dtype=float)
        else:
            # CRITICAL: Try native predictions first for ALL methods
            used_native_pred = False
            if isinstance(chosen_row, pd.DataFrame) and "y_pred_test" in chosen_row.columns:
                raw = chosen_row["y_pred_test"].iloc[0]
                arr = _parse_predictions_from_engine(raw)
                if arr is not None and arr.shape[0] == y_te.shape[0]:
                    y_pred_raw = arr
                    used_native_pred = True
                    print(f"[{method}|{experiment}|seed={seed}] Using native predictions from engine")

            if not used_native_pred:
                # Fallback to expression evaluation
                expr_clean = _clean_expr_text(top_expr)
                expr_mapped = _remap_vars_any(expr_clean, feature_names)
                try:
                    if method == "lgo":
                        # Prefer internal scaler if present
                        eval_scaler_map = None
                        if isinstance(cand, pd.DataFrame) and "scaler_json" in cand.columns and len(cand)>0:
                            try:
                                sj = str(cand["scaler_json"].iloc[0]).strip()
                                if sj.startswith("{"):
                                    scaler_info = json.loads(sj)
                                    means = scaler_info.get("mean", []); stds = scaler_info.get("std", [])
                                    if len(means) == len(feature_names) and len(stds) == len(feature_names):
                                        eval_scaler_map = {
                                            "feature_means": {feature_names[i]: float(means[i]) for i in range(len(feature_names))},
                                            "feature_stds":  {feature_names[i]: float(stds[i])  for i in range(len(feature_names))},
                                        }
                                        print(f"[{method}|{experiment}|seed={seed}] Using LGO internal scaler (train-only)")
                            except Exception:
                                pass
                        if eval_scaler_map is None: eval_scaler_map = scaler_map_train
                        ns = _make_ns_for_lgo(X_te, feature_names, zscore=True, scaler_map=eval_scaler_map)
                    elif method in ("pstree","rils_rols"):
                        ns = _make_ns_for_pstree_rils(X_te, feature_names, scaler_map_train)
                    else:
                        ns = _make_ns_generic(X_te, feature_names)

                    y_pred_raw = eval(expr_mapped, {}, ns)

                    # RILS-ROLS linear refit (alpha,beta) if provided - only when using expression replay
                    if method == "rils_rols" and isinstance(chosen_row, pd.DataFrame):
                        try:
                            alpha = float(chosen_row.get("alpha", np.nan).iloc[0])
                            beta  = float(chosen_row.get("beta",  np.nan).iloc[0])
                            if np.isfinite(alpha) and np.isfinite(beta):
                                y_pred_raw = alpha * y_pred_raw + beta
                                print(f"[{method}|{experiment}|seed={seed}] Applied linear refit: α={alpha:.3f}, β={beta:.3f}")
                        except Exception:
                            pass

                except Exception as e:
                    print(f"[EVAL-ERR|{method}|{experiment}|seed={seed}] {e}; fallback to mean predictor.")
                    y_pred_raw = np.full_like(y_te, float(np.mean(y_tr)), dtype=float)

        # Self-check C: anomaly scan on RAW
        y_raw = np.asarray(y_pred_raw, dtype=float).ravel()
        has_nan = bool(np.isnan(y_raw).any())
        has_inf = bool(np.isinf(y_raw).any())
        is_const = bool(np.allclose(np.nanvar(y_raw), 0.0, atol=1e-12))
        has_sentinel = bool(np.nanmax(np.abs(y_raw)) >= SENTINEL_ABS) if y_raw.size>0 else False
        has_huge = bool(np.nanmax(np.abs(y_raw)) >= HUGE_MAG) if y_raw.size>0 else False
        status_c = "OK" if (not has_nan and not has_inf and not is_const and not has_sentinel and not has_huge) else "WARN"
        sc_rows.append({
            "dataset": args.dataset, "task": args.task, "method": method, "experiment": experiment,
            "seed": seed, "split": 0, "check": "anomaly_scan_raw_pred", "status": status_c,
            "value": None, "threshold": "", "notes": f"nan={has_nan}, inf={has_inf}, const={is_const}, sentinel>={SENTINEL_ABS}={has_sentinel}, huge>={HUGE_MAG}={has_huge}"
        })
        if status_c != "OK":
            print(f"[SELFCHK|{method}|{experiment}|seed={seed}] anomaly_scan: nan={has_nan}, inf={has_inf}, const={is_const}, sentinel>={SENTINEL_ABS}={has_sentinel}, huge>={HUGE_MAG}={has_huge}")

        # finalize predictions for metrics
        y_pred = _clip_to_train_range(y_pred_raw, y_tr)
        y_pred = np.nan_to_num(y_pred, nan=float(np.mean(y_tr)))
        
        # ALIGNED EVALUATION: Calculate all metrics from same predictions
        m = _metrics_from_raw(y_te, y_pred)

        # Self-check A: RMSE >= MAE
        status_a = "OK" if (m["RMSE"] + 1e-12) >= m["MAE"] else "FAIL"
        if status_a != "OK":
            print(f"[SELFCHK|{method}|{experiment}|seed={seed}] RMSE<MAE anomaly: RMSE={m['RMSE']:.6f}, MAE={m['MAE']:.6f}")
        sc_rows.append({
            "dataset": args.dataset, "task": args.task, "method": method, "experiment": experiment,
            "seed": seed, "split": 0, "check": "metrics_guard_RMSE_ge_MAE", "status": status_a,
            "value": m["RMSE"] - m["MAE"], "threshold": ">= 0", "notes": ""
        })

        # Self-check B: Compare with native metrics if available (informational only)
        if method in ("pstree","rils_rols") and isinstance(chosen_row, pd.DataFrame):
            if "test_loss" in chosen_row.columns:
                try:
                    rmse_int = float(np.sqrt(float(chosen_row["test_loss"].iloc[0])))
                    delta = abs(rmse_int - m["RMSE"])
                    status_b = "OK" if delta <= CONSIST_TOL else "INFO"
                    print(f"[CHECK|{method}|{experiment}|seed={seed}] native_RMSE={rmse_int:.6f} vs computed_RMSE={m['RMSE']:.6f} (Δ={delta:.6f}) [{status_b}]")
                    sc_rows.append({
                        "dataset": args.dataset, "task": args.task, "method": method, "experiment": experiment,
                        "seed": seed, "split": 0, "check": "native_vs_computed_RMSE", "status": status_b,
                        "value": delta, "threshold": f"<= {CONSIST_TOL}", "notes": f"native={rmse_int:.6f}, computed={m['RMSE']:.6f}"
                    })
                except Exception:
                    pass

        # Write unified metrics (all from same predictions)
        om_rows = [
            {"dataset":args.dataset,"task":args.task,"method":method,"experiment":experiment,"seed":seed,"split":0,"metric":"RMSE","value":m["RMSE"]},
            {"dataset":args.dataset,"task":args.task,"method":method,"experiment":experiment,"seed":seed,"split":0,"metric":"MAE","value":m["MAE"]},
            {"dataset":args.dataset,"task":args.task,"method":method,"experiment":experiment,"seed":seed,"split":0,"metric":"R2","value":m["R2"]},
        ]

        if args.save_predictions:
            pd.DataFrame({
                "row_id": np.arange(len(y_te)), "y_true": y_te,
                "y_pred": np.asarray(y_pred).ravel(),
                "y_pred_raw": np.asarray(y_pred_raw).ravel(),
                "method": method, "experiment": experiment, "seed": seed,
            }).to_csv((Path(args.outdir)/"predictions"/f"test_predictions_{method}_{experiment}_seed{seed}.csv"), index=False)

    else:
        # binary classification
        # Re-determine top expression
        if top_expr is None and isinstance(cand, pd.DataFrame) and len(cand) > 0:
            expr_col = _pick_expr_column(cand)
            if expr_col:
                ctmp = cand.copy()
                sort_key = "test_loss" if "test_loss" in ctmp.columns and ctmp["test_loss"].notna().any() else ("cv_loss" if "cv_loss" in ctmp.columns else None)
                if sort_key: ctmp = ctmp.sort_values(sort_key)
                vals = ctmp[expr_col].dropna().astype(str)
                if len(vals) > 0: 
                    top_expr = str(vals.iloc[0])
                    chosen_row = ctmp.iloc[0:1]
                else:
                    chosen_row = None
            else:
                chosen_row = None
        elif isinstance(cand, pd.DataFrame) and len(cand) > 0:
            chosen_row = cand.iloc[0:1]
        else:
            chosen_row = None

        if top_expr:
            expr_clean = _clean_expr_text(top_expr)
            expr_mapped = _remap_vars_any(expr_clean, feature_names)
            if method == "lgo":
                ns = _make_ns_for_lgo(X_te, feature_names, zscore=True, scaler_map=scaler_map_train)
            else:
                ns = _make_ns_generic(X_te, feature_names)
            try:
                y_score_raw = eval(expr_mapped, {}, ns)
            except Exception as e:
                print(f"[EVAL-ERR|{method}|{experiment}|seed={seed}] {e}; fallback to logit(mean).")
                p_base = float(np.clip(np.mean(y_tr), 1e-6, 1-1e-6))
                y_score_raw = np.full_like(y_te, float(np.log(p_base/(1.0-p_base))), dtype=float)
        else:
            p_base = float(np.clip(np.mean(y_tr), 1e-6, 1-1e-6))
            y_score_raw = np.full_like(y_te, float(np.log(p_base/(1.0-p_base))), dtype=float)

        # anomaly scan for classification raw scores
        y_raw = np.asarray(y_score_raw, dtype=float).ravel()
        has_nan = bool(np.isnan(y_raw).any()); has_inf = bool(np.isinf(y_raw).any())
        is_const = bool(np.allclose(np.nanvar(y_raw), 0.0, atol=1e-12))
        has_sentinel = bool(np.nanmax(np.abs(y_raw)) >= SENTINEL_ABS) if y_raw.size>0 else False
        has_huge = bool(np.nanmax(np.abs(y_raw)) >= HUGE_MAG) if y_raw.size>0 else False
        status_c = "OK" if (not has_nan and not has_inf and not is_const and not has_sentinel and not has_huge) else "WARN"
        sc_rows.append({
            "dataset": args.dataset, "task": args.task, "method": method, "experiment": experiment,
            "seed": seed, "split": 0, "check": "anomaly_scan_raw_score", "status": status_c,
            "value": None, "threshold": "", "notes": f"nan={has_nan}, inf={has_inf}, const={is_const}, sentinel>={SENTINEL_ABS}={has_sentinel}, huge>={HUGE_MAG}={has_huge}"
        })
        y_prob = 1.0/(1.0+np.exp(-y_score_raw))
        if (not np.all(np.isfinite(y_prob))) or y_prob.shape[0]!=y_te.shape[0]:
            p_base = float(np.clip(np.mean(y_tr), 1e-6, 1-1e-6))
            y_prob = np.full_like(y_te, p_base, dtype=float)
            y_score_raw = np.full_like(y_te, float(np.log(p_base/(1.0-p_base))), dtype=float)

        if args.save_predictions:
            pd.DataFrame({
                "row_id": np.arange(len(y_te)), "y_true": y_te,
                "y_prob": y_prob, "y_score_raw": y_score_raw,
                "method": method, "experiment": experiment, "seed": seed,
            }).to_csv((Path(args.outdir)/"predictions"/f"test_predictions_{method}_{experiment}_seed{seed}.csv"), index=False)

        m = classification_metrics(y_te, y_prob)
        om_rows = [
            {"dataset":args.dataset,"task":args.task,"method":method,"experiment":experiment,"seed":seed,"split":0,"metric":"AUROC","value":m["AUROC"]},
            {"dataset":args.dataset,"task":args.task,"method":method,"experiment":experiment,"seed":seed,"split":0,"metric":"AUPRC","value":m["AUPRC"]},
            {"dataset":args.dataset,"task":args.task,"method":method,"experiment":experiment,"seed":seed,"split":0,"metric":"Brier","value":m["Brier"]},
        ]

    # write aggregated metrics
    if om_rows:
        om_path = Path(args.outdir)/"aggregated"/"overall_metrics.csv"
        Path(args.outdir,"aggregated").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(om_rows).to_csv(om_path, mode="a", header=not om_path.exists(), index=False)

    # write selfcheck
    if sc_rows:
        sc_path = Path(args.outdir)/"aggregated"/"selfcheck_report.csv"
        Path(args.outdir,"aggregated").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sc_rows).to_csv(sc_path, mode="a", header=not sc_path.exists(), index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True); p.add_argument("--target", required=True)
    p.add_argument("--task", required=True, choices=["regression","binary_classification"])
    p.add_argument("--method", default="lgo", choices=list(ENGINES.keys()))
    p.add_argument("--experiments", default="", help="Comma list: base,lgo_soft,lgo_hard,pysr,operon,pstree,rils_rols")
    p.add_argument("--experiment", default="base")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default="")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--unit_map_json", default="")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--hparams_json", default="{}")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--dataset", default="CUSTOM")
    args = p.parse_args()

    # NOTE: avoid "python python script.py" typo; call as: python run_v3_7.py
    args.seeds_list = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds.strip() else [args.seed]

    #v3.7
    #exp_tokens = [t.strip() for t in (args.experiments.split(",") if args.experiments.strip() else [args.experiment]) if t.strip()]
    
    # v3.8
    exp_tokens = [t.strip() for t in (args.experiments.split(",") if args.experiments.strip() else [args.experiment]) if t.strip()]
    mapping = {}
    for tok in exp_tokens:
        if tok in ["pysr","operon","pstree","rils_rols"]:
            mapping[tok] = (tok, "base")
        else:
            # Updated mapping for v3.7
            if tok == "base":
                mapping[tok] = ("lgo", "base")
            elif tok == "lgo_soft":  # New name for old "lgo"
                mapping[tok] = ("lgo", "lgo_soft")
            elif tok == "lgo_hard":  # New name for old "lgo_thre"
                mapping[tok] = ("lgo", "lgo_hard")
            else:
                raise SystemExit(f"Unknown experiment token: {tok}")

    for seed in args.seeds_list:
        for tok in exp_tokens:
            method, experiment = mapping[tok]
            try:
                run_once(args, method, experiment, seed)
            except Exception as e:
                from pathlib import Path
                import traceback, time
                ag = Path(args.outdir) / "aggregated"
                ag.mkdir(parents=True, exist_ok=True)
                err_path = ag / "errors_log.csv"

                import csv
                with open(err_path, "a", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([int(time.time()), args.dataset, method, experiment, seed,
                                str(e), traceback.format_exc().replace("\n", " | ")])

                print(f"[ERROR] {method}|{experiment}|seed={seed} crashed; logged to {err_path}.")
                continue  # 继续后续实验/种子

    print(f"[OK] LGO v3.8 finished - LGO experiments renamed (lgo_soft/lgo_hard).",
          {"experiments": exp_tokens, "seeds": args.seeds_list, "outdir": args.outdir})

if __name__ == "__main__":
    main()
