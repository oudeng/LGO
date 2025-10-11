# -*- coding: utf-8 -*-
"""
LGO_v2_1.py — DEAP engine with renamed LGO experiments

Sep 29, 2025

- Enhance v2.0, to avoid KeyError (interrupt) by empty result return. 
  e.g.: HydraulicSys dataset on seed 34 only.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

try:
    from deap import base, creator, tools, gp, algorithms
except Exception:  # pragma: no cover
    base = None; creator = None; tools = None; gp = None; algorithms = None

# ==========================================================
# Safe math primitives
# ==========================================================
EPS = 1e-12

def _as_arr(x):
    return np.asarray(x, dtype=float)

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

# ==========================================================
# LGO family
# ==========================================================
def lgo(x, a, b):
    """Soft gating: x * sigmoid(a * (x - b))"""
    return x * ssigmoid(a * (x - b))

def lgo_thre(x, a, b):
    """Hard threshold: sigmoid(a * (x - b))"""
    return ssigmoid(a * (x - b))

def lgo_pair(x, y, a, b):
    return (x * y) * ssigmoid(a * ((x - y) - b))

def lgo_and2(x, y, a, b):
    s1 = ssigmoid(a * (x - b))
    s2 = ssigmoid(a * (y - b))
    return (x * y) * (s1 * s2)

def lgo_or2(x, y, a, b):
    s1 = ssigmoid(a * (x - b))
    s2 = ssigmoid(a * (y - b))
    gate = 1.0 - (1.0 - s1) * (1.0 - s2)  # OR-like
    return (x + y) * gate

def lgo_and3(x, y, z, a, b):
    s1 = ssigmoid(a * (x - b))
    s2 = ssigmoid(a * (y - b))
    s3 = ssigmoid(a * (z - b))
    return (x * y * z) * (s1 * s2 * s3)

# ---  gate over ANY expression ---
def gate_expr(expr, a, b):
    return expr * ssigmoid(a * (expr - b))

# ==========================================================
# Metrics and helpers
# ==========================================================
DEFAULT_Y_CLIP = None

def clip_and_sanitize(yhat, y_clip=DEFAULT_Y_CLIP):
    yhat = _as_arr(yhat)
    if y_clip is not None and len(y_clip) == 2:
        yhat = np.clip(yhat, y_clip[0], y_clip[1])
    return np.where(np.isfinite(yhat), yhat, 0.0)

def mse(y, yhat):
    y = _as_arr(y); yhat = _as_arr(yhat)
    with np.errstate(over="ignore"):
        return float(np.mean((y - yhat) ** 2))

def r2(y, yhat):
    y = _as_arr(y); yhat = _as_arr(yhat)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - yhat) ** 2)
    if ss_tot < EPS:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

# ==========================================================
# Z-Score standardization
# ==========================================================
@dataclass
class ZScaler:
    mean_: np.ndarray
    std_: np.ndarray
    @staticmethod
    def fit(X: np.ndarray) -> "ZScaler":
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0, ddof=0)
        sd = np.where(sd > 0, sd, 1.0)
        return ZScaler(mean_=mu, std_=sd)
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_
    def to_json(self) -> str:
        return json.dumps({"mean": self.mean_.tolist(), "std": self.std_.tolist()})

# ==========================================================
# Typed GP grammar
# ==========================================================
class Feat: pass
class Expr: pass
class Pos:  pass
class Thr:  pass

class TF:  pass
class EPI: pass
class ENV: pass

# v2: optional light grouping for Gene-like tasks
class FeatTF:  pass
class FeatEpi: pass
class FeatEnv: pass

def as_pos_func(z):  # softplus; accepts scalar or numpy array, reduces→scalar
    try:
        arr = np.asarray(z, dtype=float)
        zval = float(np.nanmean(arr)) if arr.ndim > 0 else float(arr)
    except Exception:
        zval = float(z)
    zval = max(-60.0, min(60.0, zval))
    return float(np.log1p(np.exp(zval)))

def as_thr_func(z):  # clip to plausible range for z-scored features; accepts array
    try:
        arr = np.asarray(z, dtype=float)
        zval = float(np.nanmean(arr)) if arr.ndim > 0 else float(arr)
    except Exception:
        zval = float(z)
    return float(max(-3.0, min(3.0, zval)))

# generators/primitives (no lambda, pickling-safe)
def _erc(): return random.gauss(0.0, 1.0)
def _id_feat(x): return x
def _id_pos(a):  return a
def _id_thr(b):  return b
def id_expr(x):  return x
def add2(a,b): return a+b
def sub2(a,b): return a-b
def mul2(a,b): return a*b
def pow2(a,b): return spow(a,b)

# v2: group-to-Feat bridges (to keep arithmetic unchanged)
def tf_to_feat(x):  return x
def epi_to_feat(x): return x
def env_to_feat(x): return x

def _rand_pos():
    # softplus^-1 around [0.3, 8] mass
    u = random.uniform(-2.0, 2.0)
    return math.log1p(math.exp(3.0 * u)) + 0.25  # ~[0.25,12]

def _rand_thr():
    return random.gauss(0.0, 1.0)  # for z-scored features

def infer_feature_groups(feature_names: Sequence[str]):
    """Heuristic grouping for Gene-like tasks. Returns dict: idx->group ('tf'|'epi'|'env'|None)."""
    groups = {}
    tf_pat   = re.compile(r"(creb|nf.?k.?b|stat3|p53)", re.I)
    epi_pat  = re.compile(r"(h3k27ac|h3k4me3|h3k9me3|methyl|chromatin|access)", re.I)
    env_pat  = re.compile(r"(oxid|nutrient|depriv|stress|pO2|pH|dose|vent)", re.I)
    for i, nm in enumerate(feature_names):
        s = str(nm)
        if tf_pat.search(s):   groups[i] = "tf"
        elif epi_pat.search(s):groups[i] = "epi"
        elif env_pat.search(s):groups[i] = "env"
        else: groups[i] = None
    return groups

import re

def build_typed_pset(n_features: int,
                     experiment: str,
                     feature_names: Optional[Sequence[str]] = None,
                     typed_mode: str = 'strict',
                     include_lgo_pair: bool = False,
                     include_lgo_cross_old: bool = False,
                     prior_mode: str = "lgo",
                     include_lgo_multi: bool = True,
                     include_lgo_and3: bool = True,
                     enable_gate_expr: bool = False,
                     typed_grouping: str = "none"):
    """
    Typed primitive set with LGO family:
    - experiment == "lgo_soft": uses lgo (soft gating)
    - experiment == "lgo_hard": uses lgo_thre (hard threshold)
    - experiment == "base": no LGO operators
    - Optional gate_expr over any sub-expression (Expr,Pos,Thr)->Expr, controlled by enable_gate_expr.
    - Optional light typed grouping ('auto_gene'): encourages cross-group pairs (TF×Epi/TF×Env/Epi×Env).
    """
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    use_grouping = (typed_grouping != "none")
    group_map = infer_feature_groups(feature_names) if use_grouping else {}

    # If grouping enabled but no matches at all, fall back to no grouping
    if use_grouping and not any(v in ("tf","epi","env") for v in group_map.values()):
        use_grouping = False

    # --- Build pset ---
    if use_grouping:
        # Each argument gets a specific group type
        arg_types = []
        for i in range(n_features):
            g = group_map.get(i, None)
            if g == "tf":   arg_types.append(FeatTF)
            elif g == "epi":arg_types.append(FeatEpi)
            elif g == "env":arg_types.append(FeatEnv)
            else:           arg_types.append(Feat)  # unknown stays general Feat
        pset = gp.PrimitiveSetTyped("MAIN", arg_types, Expr, prefix="ARG")
        # Bridges TF/EPI/ENV -> Feat
        pset.addPrimitive(tf_to_feat,  [FeatTF],  Feat, name="tf2feat")
        pset.addPrimitive(epi_to_feat, [FeatEpi], Feat, name="epi2feat")
        pset.addPrimitive(env_to_feat, [FeatEnv], Feat, name="env2feat")
        # Identity
        pset.addPrimitive(_id_feat,    [Feat],    Feat, name="idF")
        pset.addPrimitive(id_expr,     [Feat],    Expr, name="id")
    else:
        pset = gp.PrimitiveSetTyped("MAIN", [Feat] * n_features, Expr, prefix="ARG")
        pset.addPrimitive(id_expr, [Feat], Expr, name="id")
        pset.addPrimitive(_id_feat, [Feat], Feat, name="idF")

    # base arithmetic
    pset.addPrimitive(add2, [Expr,Expr], Expr, name="add")
    pset.addPrimitive(sub2, [Expr,Expr], Expr, name="sub")
    pset.addPrimitive(mul2, [Expr,Expr], Expr, name="mul")
    pset.addPrimitive(sdiv, [Expr,Expr], Expr, name="div")
    pset.addPrimitive(ssqrt,[Expr],      Expr, name="sqrt")
    pset.addPrimitive(slog, [Expr],      Expr, name="log")
    pset.addPrimitive(pow2, [Expr,Expr], Expr, name="pow")

    # Expr terminals
    pset.addTerminal(0.0, Expr, name="zero")
    pset.addTerminal(1.0, Expr, name="one")
    pset.addEphemeralConstant("erc_expr", _erc, Expr)

    # LGO family - v2: updated experiment names
    if experiment in ("lgo_soft", "lgo_hard"):
        # Choose the appropriate LGO function based on experiment
        if experiment == "lgo_soft":
            pset.addPrimitive(lgo, [Expr, Pos, Thr], Expr, name="lgo")
        else:  # lgo_hard
            pset.addPrimitive(lgo_thre, [Expr, Pos, Thr], Expr, name="lgo_thre")
        
        # Additional LGO operators only for lgo_soft (preserves v1 behavior)
        if experiment == "lgo_soft":
            if include_lgo_pair:
                pset.addPrimitive(lgo_pair, [Expr, Expr, Pos, Thr], Expr, name="lgo_pair")

            # multi-input (general)
            if include_lgo_multi:
                pset.addPrimitive(lgo_and2, [Expr, Expr, Pos, Thr], Expr, name="lgo_and2")
                pset.addPrimitive(lgo_or2,  [Expr, Expr, Pos, Thr], Expr, name="lgo_or2")
                if include_lgo_and3 and n_features >= 3:
                    pset.addPrimitive(lgo_and3, [Expr, Expr, Expr, Pos, Thr], Expr, name="lgo_and3")

            # optional gate over ANY expression
            if enable_gate_expr:
                pset.addPrimitive(gate_expr, [Expr, Pos, Thr], Expr, name="gate_expr")

            # optional cross-group typed gates (light)
            if use_grouping and include_lgo_multi:
                # Encourage TF×Epi, TF×Env, Epi×Env
                def _lgo_and2_tf_epi(x, y, a, b): return lgo_and2(x, y, a, b)
                def _lgo_or2_tf_epi(x, y, a, b):  return lgo_or2(x, y, a, b)
                def _lgo_and2_tf_env(x, y, a, b): return lgo_and2(x, y, a, b)
                def _lgo_or2_tf_env(x, y, a, b):  return lgo_or2(x, y, a, b)
                def _lgo_and2_epi_env(x, y, a, b):return lgo_and2(x, y, a, b)
                def _lgo_or2_epi_env(x, y, a, b): return lgo_or2(x, y, a, b)

                pset.addPrimitive(_lgo_and2_tf_epi, [FeatTF, FeatEpi, Pos, Thr], Expr, name="lgo_and2_tf_epi")
                pset.addPrimitive(_lgo_or2_tf_epi,  [FeatTF, FeatEpi, Pos, Thr], Expr, name="lgo_or2_tf_epi")
                pset.addPrimitive(_lgo_and2_tf_env, [FeatTF, FeatEnv, Pos, Thr], Expr, name="lgo_and2_tf_env")
                pset.addPrimitive(_lgo_or2_tf_env,  [FeatTF, FeatEnv, Pos, Thr], Expr, name="lgo_or2_tf_env")
                pset.addPrimitive(_lgo_and2_epi_env,[FeatEpi, FeatEnv, Pos, Thr], Expr, name="lgo_and2_epi_env")
                pset.addPrimitive(_lgo_or2_epi_env, [FeatEpi, FeatEnv, Pos, Thr], Expr, name="lgo_or2_epi_env")

            if include_lgo_cross_old:
                def lgo_cross_old(x, y, a, b):
                    return x * ssigmoid(a * (y - b))
                pset.addPrimitive(lgo_cross_old, [Expr, Expr, Pos, Thr], Expr, name="lgo_cross")

        # typed priors for a/b (respect prior_mode)
        if prior_mode == "baseline":
            def _rand_pos_base(): return abs(random.uniform(-1.0, 1.0)) + 1e-3
            def _rand_thr_base(): return random.uniform(-1.0, 1.0)
            pset.addEphemeralConstant("apos", _rand_pos_base, Pos)
            pset.addEphemeralConstant("bthr", _rand_thr_base, Thr)
        else:
            pset.addEphemeralConstant("apos", _rand_pos, Pos)
            pset.addEphemeralConstant("bthr", _rand_thr, Thr)

        if typed_mode == "light":
            pset.addPrimitive(as_pos_func, [Expr], Pos, name="as_pos")
            pset.addPrimitive(as_thr_func, [Expr], Thr, name="as_thr")

    # rename features
    for i, nm in enumerate(feature_names):
        pset.renameArguments(**{f"ARG{i}": nm})

    # safety guards
    if Expr not in pset.terminals or len(pset.terminals[Expr]) == 0:
        pset.addTerminal(0.0, Expr, name="zero_guard")
    if Feat not in pset.primitives or len(pset.primitives[Feat]) == 0:
        pset.addPrimitive(_id_feat, [Feat], Feat, name="idF_guard")

    return pset

# ==========================================================
# Complexity map
# ==========================================================
DEFAULT_COMPLEXITY = {
    "add": 1.0, "sub": 1.0, "mul": 1.0, "div": 1.5,
    "sqrt": 1.5, "log": 1.5, "pow": 2.0,
    "id": 0.0, "idF": 0.0, "idPos": 0.0, "idThr": 0.0,
    "tf2feat": 0.0, "epi2feat": 0.0, "env2feat": 0.0,
    "lgo": 2.0, "lgo_thre": 2.0, "lgo_pair": 3.0, "lgo_cross": 3.2,
    "lgo_and2": 3.0, "lgo_or2": 3.0, "lgo_and3": 3.8,
    "lgo_and2_tf_epi": 3.0, "lgo_or2_tf_epi": 3.0,
    "lgo_and2_tf_env": 3.0, "lgo_or2_tf_env": 3.0,
    "lgo_and2_epi_env": 3.0, "lgo_or2_epi_env": 3.0,
    "gate_expr": 2.0, "as_TF": 0.0, "as_EPI": 0.0, "as_ENV": 0.0,
    "as_pos": 1.0, "as_thr": 1.0,
}

def calc_complexity(ind: "gp.PrimitiveTree", comp_map: Dict[str, float] = DEFAULT_COMPLEXITY) -> float:
    c = 0.0
    for node in ind:
        if hasattr(node, "name"):
            c += comp_map.get(node.name, 0.0)
    return float(c)

# ==========================================================
# Micro-mutation for LGO parameters (a,b)
# ==========================================================
def mutate_lgo_params(ind: "gp.PrimitiveTree",
                      jitter_pos_sigma: float = 0.25,
                      jitter_thr_sigma: float = 0.20,
                      p: float = 0.3) -> Tuple["gp.PrimitiveTree"]:
    new = copy.deepcopy(ind)
    for i, node in enumerate(new):
        # In DEAP, ephemerals/constants are Terminals with attributes .ret and .value
        if hasattr(node, "ret") and hasattr(node, "value"):
            if node.ret is Pos and node.value is not None and random.random() < p:
                eps = random.gauss(0.0, jitter_pos_sigma)
                new[i] = gp.Terminal(node.value * math.exp(eps), False, node.ret)
            elif node.ret is Thr and node.value is not None and random.random() < p:
                eps = random.gauss(0.0, jitter_thr_sigma)
                new[i] = gp.Terminal(node.value + eps, False, node.ret)
    return (new,)

# ==========================================================
# Train-time evaluation (with optional CV proxy; allows annealing via context)
# ==========================================================
def _compile_func(toolbox, ind):
    try:
        return toolbox.compile(expr=ind)
    except Exception:
        return None

def _safe_kfold(n_samples: int, requested: int) -> int:
    if n_samples < 2:
        return 2
    return max(2, min(requested, n_samples))

def clip_and_eval(func, Xz, y, y_clip):
    cols = [Xz[:, i] for i in range(Xz.shape[1])]
    with np.errstate(all="ignore"):
        yhat = func(*cols)
    return clip_and_sanitize(yhat, y_clip)

def train_eval_with_proxy(ind, toolbox, Xz, y,
                          cv_proxy_weight: float = 0.0,
                          cv_proxy_subsample: float = 0.3,
                          cv_proxy_folds: int = 2,
                          y_clip=DEFAULT_Y_CLIP,
                          eval_calls: Dict[str, int] = None,
                          rnd: Optional[random.Random] = None):
    if eval_calls is not None:
        eval_calls["n"] += 1
    func = _compile_func(toolbox, ind)
    if func is None:
        return (1e9,)
    yhat = clip_and_eval(func, Xz, y, y_clip)
    train_mse = mse(y, yhat)

    if cv_proxy_weight > 0.0 and Xz.shape[0] >= 4:
        n = Xz.shape[0]
        idx = np.arange(n)
        if rnd is None:
            rnd = random.Random(0)
        rnd.shuffle(idx)
        m = max(20, int(cv_proxy_subsample * n))
        m = min(m, n)
        sub_idx = idx[:m]
        Xs = Xz[sub_idx]; ys = y[sub_idx]
        k = _safe_kfold(len(Xs), cv_proxy_folds)
        kf = KFold(n_splits=k, shuffle=True, random_state=rnd.randint(0, 10**6))
        mses = []
        for tr, va in kf.split(Xs):
            yhat_va = clip_and_eval(func, Xs[va], ys[va], y_clip)
            mses.append(mse(ys[va], yhat_va))
        proxy = float(np.mean(mses)) if mses else train_mse
        fit = train_mse + cv_proxy_weight * proxy
    else:
        fit = train_mse
    return (fit,)

# ==========================================================
# Post-hoc CV & test evaluation
# ==========================================================
def crossval_deap_func(toolbox, ind, Xz, y, n_splits=5, y_clip=DEFAULT_Y_CLIP, random_state=0):
    func = _compile_func(toolbox, ind)
    if func is None:
        return 1e9, 1.0, 0.0
    k = _safe_kfold(len(Xz), n_splits)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    mses, r2s = [], []
    for tr, va in kf.split(Xz):
        yhat_va = clip_and_eval(func, Xz[va], y[va], y_clip)
        mses.append(mse(y[va], yhat_va))
        r2s.append(r2(y[va], yhat_va))
    cv_loss = float(np.median(mses)) if mses else 1e9
    instability = float(np.std(mses) / max(np.mean(mses), EPS)) if mses else 1.0
    cv_r2 = float(np.mean(r2s)) if r2s else 0.0
    return cv_loss, instability, cv_r2

def test_eval(toolbox, ind, Xz_test, y_test, y_clip=DEFAULT_Y_CLIP):
    func = _compile_func(toolbox, ind)
    if func is None:
        return 1e9, 0.0
    yhat_te = clip_and_eval(func, Xz_test, y_test, y_clip)
    return mse(y_test, yhat_te), r2(y_test, yhat_te)

# ==========================================================
# Local optimization over LGO parameters
# ==========================================================
def has_lgo(ind: "gp.PrimitiveTree") -> bool:
    s = str(ind)
    return any(tag in s for tag in ("lgo(", "lgo_thre(", "lgo_pair(", "lgo_and2(", "lgo_or2(", "lgo_and3(", "gate_expr("))

def _scale_pos_params(ind: "gp.PrimitiveTree", scale: float) -> "gp.PrimitiveTree":
    new = copy.deepcopy(ind)
    for i, node in enumerate(new):
        if hasattr(node, "ret") and hasattr(node, "value"):
            if node.ret is Pos and node.value is not None and np.isfinite(node.value):
                new[i] = gp.Terminal(node.value * scale, False, node.ret)
    return new

def _jitter_thr_params(ind: "gp.PrimitiveTree", sigma: float = 0.25, p: float = 0.8) -> "gp.PrimitiveTree":
    new = copy.deepcopy(ind)
    for i, node in enumerate(new):
        if hasattr(node, "ret") and hasattr(node, "value"):
            if node.ret is Thr and node.value is not None and np.isfinite(node.value) and random.random() < p:
                eps = random.gauss(0.0, sigma)
                new[i] = gp.Terminal(node.value + eps, False, node.ret)
    return new

def local_optimize_lgo_params_v2(ind: "gp.PrimitiveTree",
                                 toolbox,
                                 Xz, y,
                                 steps: int = 60,
                                 pos_scales: Sequence[float] = (0.8, 1.2, 2.0, 3.0),
                                 jitter_thr_sigma: float = 0.25,
                                 patience: int = 15,
                                 random_state: int = 0) -> "gp.PrimitiveTree":
    """Coordinate search: alternates between 'b-jitter' and a few 'a-scale' proposals."""
    if not has_lgo(ind):
        return ind
    rnd = random.Random(random_state)
    best = copy.deepcopy(ind)
    best_loss = crossval_deap_func(toolbox, best, Xz, y, n_splits=5, random_state=random_state)[0]
    no_improve = 0
    for t in range(steps):
        proposals = []
        if t % 2 == 0:
            # b-first: strong focus
            cand = _jitter_thr_params(best, sigma=jitter_thr_sigma, p=0.9)
            proposals.append(cand)
        else:
            # a-scale grid
            for sc in pos_scales:
                proposals.append(_scale_pos_params(best, sc))

        improved = False
        for cand in proposals:
            loss, _, _ = crossval_deap_func(toolbox, cand, Xz, y, n_splits=5, random_state=random_state)
            if loss + 1e-12 < best_loss:
                best, best_loss = cand, loss
                improved = True
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    return best

# ==========================================================
# Evolution loop with tqdm progress bar (semantics == eaSimple)
# ==========================================================
def eaSimple_with_progress(pop, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=False, eval_ctx: Optional[dict]=None):
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats is not None else [])

    # Evaluate invalid individuals
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    if eval_ctx is not None: eval_ctx["gen"] = 0; eval_ctx["ngen"] = ngen
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) if hasattr(toolbox, "map") else map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(pop)
    record = stats.compile(pop) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    pbar = tqdm(total=ngen, desc="DEAP(EA)", leave=False) if tqdm else None
    for gen in range(1, ngen + 1):
        if eval_ctx is not None: eval_ctx["gen"] = gen
        offspring = toolbox.select(pop, len(pop))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) if hasattr(toolbox, "map") else map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(offspring)
        pop[:] = offspring
        record = stats.compile(pop) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if pbar:
            pbar.update(1)
            try:
                best = min(ind.fitness.values[0] for ind in pop)
                pbar.set_postfix_str(f"best={best:.4g}")
            except Exception:
                pass
    if pbar:
        pbar.close()
    return pop, logbook

# ==========================================================
# High-level SR entry (v2: renamed function)
# ==========================================================
def run_lgo_sr_v2(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    experiment: str = "base",           # v2: "base", "lgo_soft", "lgo_hard"
    typed_mode: str = "light",          # 'none'|'light'|'strict'
    include_lgo_pair: bool = False,
    include_lgo_cross_old: bool = False,
    include_lgo_multi: bool = True,
    include_lgo_and3: bool = True,
    enable_gate_expr: bool = False,     # small switch
    typed_grouping: str = "none",       # 'none' | 'auto_gene'
    use_zscore: bool = True,
    prior_mode: str = "lgo",
    pop_size: int = 600,
    ngen: int = 80,
    tournament_size: int = 5,
    cx_pb: float = 0.85,
    mut_pb: float = 0.15,
    max_height: int = 10,
    hof_size: int = 20,
    topk_cv: int = 12,
    topk_local_opt: int = 6,
    local_opt_steps: int = 60,
    micro_mutation_prob: float = 0.10,
    cv_proxy_weight: float = 0.0,
    cv_proxy_subsample: float = 0.30,
    cv_proxy_folds: int = 2,
    # v2 annealing (optional; default keeps constant weight)
    cv_proxy_weight_final: Optional[float] = None,
    cv_proxy_warmup_frac: float = 0.8,
    random_state: int = 0,
    y_clip=DEFAULT_Y_CLIP,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    assert gp is not None, "DEAP is required."
    t0 = time.time()
    rnd = random.Random(random_state)
    np.random.seed(random_state); random.seed(random_state)

    # Standardize
    scaler = ZScaler.fit(X) if use_zscore else ZScaler(mean_=np.zeros(X.shape[1]), std_=np.ones(X.shape[1]))
    Xz = scaler.transform(X); Xz_test = scaler.transform(X_test) if X_test is not None else None

    # Primitive set
    if typed_mode in ("strict", "light"):
        pset = build_typed_pset(X.shape[1], experiment=experiment, feature_names=feature_names,
                                typed_mode=typed_mode, include_lgo_pair=include_lgo_pair,
                                include_lgo_cross_old=include_lgo_cross_old, prior_mode=prior_mode,
                                include_lgo_multi=include_lgo_multi, include_lgo_and3=include_lgo_and3,
                                enable_gate_expr=enable_gate_expr, typed_grouping=typed_grouping)
    else:
        # Untyped fallback (ablation/debug)
        pset = gp.PrimitiveSet("MAIN", X.shape[1], prefix="ARG")
        pset.addPrimitive(add2, 2); pset.addPrimitive(sub2, 2); pset.addPrimitive(mul2, 2)
        pset.addPrimitive(sdiv, 2); pset.addPrimitive(ssqrt, 1); pset.addPrimitive(slog, 1); pset.addPrimitive(pow2, 2)
        if experiment in ("lgo_soft", "lgo_hard"):  # v2: renamed experiments
            if experiment == "lgo_soft":
                pset.addPrimitive(lgo, 3)
                if include_lgo_pair: pset.addPrimitive(lgo_pair, 4)
                if include_lgo_multi:
                    pset.addPrimitive(lgo_and2, 4); pset.addPrimitive(lgo_or2,  4)
                    if include_lgo_and3 and X.shape[1] >= 3: pset.addPrimitive(lgo_and3, 5)
                if enable_gate_expr:
                    pset.addPrimitive(gate_expr, 3)
            else:  # lgo_hard
                pset.addPrimitive(lgo_thre, 3)
        pset.addTerminal(0.0, name="zero"); pset.addTerminal(1.0, name="one")
        if feature_names is not None:
            for i, nm in enumerate(feature_names):
                pset.renameArguments(**{f"ARG{i}": nm})

    # Fitness/Individual
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_height//2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate_base", gp.mutUniform, expr=toolbox.expr, pset=pset)

    def mutate_with_micro(ind):
        ind2, = toolbox.mutate_base(ind)
        if typed_mode in ("strict", "light") and (rnd.random() < micro_mutation_prob):
            ind2, = mutate_lgo_params(ind2, p=0.5)
        return (ind2,)

    toolbox.register("mutate", mutate_with_micro)
    toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=max_height))

    # evaluation wrapper with optional annealing
    eval_calls = {"n": 0}
    eval_ctx = {
        "gen": 0, "ngen": ngen,
        "w0": cv_proxy_weight,
        "wf": cv_proxy_weight_final if (cv_proxy_weight_final is not None) else cv_proxy_weight,
        "warm": cv_proxy_warmup_frac,
        "rnd": rnd,
    }
    def _effective_weight():
        # Linear ramp from w0 to wf after warmup frac
        g = eval_ctx["gen"]; G = max(1, eval_ctx["ngen"])
        frac = g / G
        w0 = eval_ctx["w0"]; wf = eval_ctx["wf"]; warm = eval_ctx["warm"]
        if wf == w0 or frac <= warm:
            return w0
        alpha = (frac - warm) / max(1e-9, 1.0 - warm)
        return (1 - alpha) * w0 + alpha * wf

    def _evaluate(ind):
        w = _effective_weight()
        return train_eval_with_proxy(ind, toolbox, Xz, y,
                                     cv_proxy_weight=w,
                                     cv_proxy_subsample=cv_proxy_subsample,
                                     cv_proxy_folds=cv_proxy_folds,
                                     y_clip=y_clip,
                                     eval_calls=eval_calls,
                                     rnd=eval_ctx["rnd"])

    toolbox.register("evaluate", _evaluate)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(hof_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean); stats.register("min", np.min); stats.register("max", np.max)

    # Progress-enabled EA (semantics identical to eaSimple)
    pop, log = eaSimple_with_progress(pop, toolbox, cxpb=cx_pb, mutpb=mut_pb, ngen=ngen,
                                      stats=stats, halloffame=hof, verbose=False, eval_ctx=eval_ctx)

    # Post-hoc eval
    rows: List[Dict[str, Any]] = []
    K = min(topk_cv, len(hof))
    for i in range(K):
        ind = hof[i]
        ind_opt = copy.deepcopy(ind)
        if i < topk_local_opt:
            ind_opt = local_optimize_lgo_params_v2(ind_opt, toolbox, Xz, y, steps=local_opt_steps, random_state=random_state+i)

        cv_loss, instability, cv_r2 = crossval_deap_func(toolbox, ind_opt, Xz, y, n_splits=5, random_state=random_state)

        test_loss, test_r2 = (np.nan, np.nan)
        if X_test is not None and y_test is not None:
            test_loss, test_r2 = test_eval(toolbox, ind_opt, Xz_test, y_test)

        expr_str = str(ind_opt)
        rows.append({
            "rank": i,
            "expr": expr_str,
            "cv_loss": float(cv_loss),
            "cv_r2": float(cv_r2),
            "instability": float(instability),
            "test_loss": float(test_loss) if np.isfinite(test_loss) else np.nan,
            "test_r2": float(test_r2) if np.isfinite(test_r2) else np.nan,
            "complexity": float(calc_complexity(ind_opt)),
            "height": int(ind_opt.height),
            "size": int(len(ind_opt)),
            "eval_calls": int(eval_calls["n"]),
            "runtime_sec": float(time.time() - t0),
            "used_lgo": ("lgo(" in expr_str),
            "used_lgo_thre": ("lgo_thre(" in expr_str),
            "used_lgo_pair": ("lgo_pair(" in expr_str),
            "used_lgo_and2": ("lgo_and2(" in expr_str),
            "used_lgo_or2":  ("lgo_or2(" in expr_str),
            "used_lgo_and3": ("lgo_and3(" in expr_str),
            "used_gate_expr": ("gate_expr(" in expr_str),
            "scaler_json": ZScaler(mean_=scaler.mean_, std_=scaler.std_).to_json(),
            "experiment": experiment,
            "typed_mode": typed_mode,
            "use_zscore": use_zscore,
            "prior_mode": prior_mode,
            "typed_grouping": typed_grouping,
            "enable_gate_expr": bool(enable_gate_expr),
        })

    # v2.0    
    #df = pd.DataFrame(rows).sort_values(["cv_loss","complexity","instability"], ascending=[True,True,True]).reset_index(drop=True)
    #return df

    # v2.1
    # 替换为（安全版）, 在 sort_values 前做空判断，并容忍部分列缺失。
    # 作用：就算某次（例如 seed=34）最终候选为空，也会正常返回空表，
    #      主控会用“均值基线”产出预测并把指标写入 overall_metrics.csv，避免中断。
    df = pd.DataFrame(rows)
    if df.empty:
        # 直接返回空 DF，让上层走“空候选→基线预测”的兜底路径
        return df

    # 仅对存在的列排序，避免因某列缺失导致 KeyError
    sort_cols = [c for c in ["cv_loss","complexity","instability"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True]*len(sort_cols)).reset_index(drop=True)
    return df
