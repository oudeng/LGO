import math, re
import numpy as np
def as_pos(x): return np.log1p(np.exp(x))
def as_thr(x, lo=-3.0, hi=3.0): return np.clip(x, lo, hi)
def sigm(u): return 1.0/(1.0+np.exp(-u))
def lgo_thre(x, a, b): return sigm(a*(x - b))
def id(x): return x
def idF(x): return x
one = 1.0
zero = 0.0
def add(x,y): return np.asarray(x)+np.asarray(y)
def sub(x,y): return np.asarray(x)-np.asarray(y)
def mul(x,y): return np.asarray(x)*np.asarray(y)
def div(x,y):
    y = np.asarray(y)
    return np.asarray(x)/np.where(y==0, np.finfo(float).eps, y)
def sdiv(x,y): return div(x,y)
def ssqrt(x): return np.sqrt(np.maximum(x, 0.0))
def slog(x): return np.log(np.maximum(x, 1e-12))
def spow(x,y): 
    try: return np.power(x,y)
    except Exception: return np.sign(x)*np.power(np.abs(x), y)
SAFE_FUNCS = {
    "sqrt": np.sqrt, "log": np.log, "exp": np.exp, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "abs": np.abs, "pow": np.power,
    "as_pos": as_pos, "as_thr": as_thr, "sigm": sigm, "lgo_thre": lgo_thre,
    "id": id, "idF": idF, "one": one, "zero": zero,
    "add": add, "sub": sub, "mul": mul, "div": div,
    "sdiv": sdiv, "ssqrt": ssqrt, "slog": slog, "spow": spow
}
def safe_eval_expr(expr, namespace):
    try:
        expr2 = expr.replace("^","**").replace("×","*").replace("÷","/").replace("−","-")
        res = eval(expr2, {"__builtins__": {}}, {**SAFE_FUNCS, **namespace})
        arr = np.asarray(res, dtype=float)
        if not np.all(np.isfinite(arr)):
            # FIXED: Reduced extreme values from 1e6 to 0.0 to work better with clipping
            # The outer clipping function will handle proper bounds
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception:
        n = len(next(iter(namespace.values()))) if namespace else 0
        return np.zeros(n, dtype=float)