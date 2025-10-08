
"""\subsection*{Minimal audit script (unit-aware thresholds)}
\noindent\textbf{Usage:}
\texttt{python tools/audit\_thresholds\_min.py \textbackslash \\
\hspace*{1.5em} --thresholds outputs/thresholds\_units.csv \textbackslash \\
\hspace*{1.5em} --guidelines config/guidelines.yaml \textbackslash \\
\hspace*{1.5em} --out outputs/threshold\_audit.csv \textbackslash \\
\hspace*{1.5em} --topk 5 [--plot ./figs/v3\_thresholds\_summary.png]}

\begin{lstlisting}[language=Python, basicstyle=\ttfamily\small]
# tools/audit_thresholds_min.py
# Minimal, reproducible threshold audit consistent with Fig. 2 / Table 4.
...
\end{lstlisting}
"""
import argparse, json, math
import pandas as pd
import yaml
from pathlib import Path

BANDS = [(0.10, "green"), (0.20, "yellow")]  # else red

def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower().replace("-", "_").replace(" ", "_") if ch.isalnum() or ch=="_")

def load_guidelines(path):
    cfg = yaml.safe_load(open(path, "r"))
    # Build a lookup: (dataset_norm, feature_norm) -> dict(pretty, unit, anchor)
    lut = {}
    for dset, feats in cfg.items():
        dkey = _norm(dset)
        for fid, meta in feats.items():
            fkeys = {_norm(fid)}
            for a in meta.get("aliases", []):
                fkeys.add(_norm(a))
            for k in fkeys:
                lut[(dkey, k)] = {
                    "pretty": meta.get("pretty", fid),
                    "unit": meta["unit"],
                    "anchor": float(meta["anchor"]),
                }
    return lut

def classify_band(rel):
    if math.isnan(rel) or math.isinf(rel):
        return "NA"
    for thr, label in BANDS:
        if rel <= thr + 1e-12:
            return label
    return "red"

def main(args):
    lut = load_guidelines(args.guidelines)
    df = pd.read_csv(args.thresholds)
    # expected columns; try to map flexible names
    colmap = {
        "dataset": [c for c in df.columns if _norm(c) in {"dataset","dset"}][0],
        "feature": [c for c in df.columns if _norm(c) in {"feature","feat","feature_id"}][0],
        "median":  [c for c in df.columns if "median" in _norm(c)][0],
        "q1":      [c for c in df.columns if _norm(c) in {"q1","p25","lower_quartile"}][0],
        "q3":      [c for c in df.columns if _norm(c) in {"q3","p75","upper_quartile"}][0],
        # optional pretty/unit if provided
    }
    recs = []
    for _, row in df.iterrows():
        dkey = _norm(str(row[colmap["dataset"]]))
        fkey = _norm(str(row[colmap["feature"]]))
        if (dkey, fkey) not in lut:
            # skip features without anchors
            continue
        meta = lut[(dkey, fkey)]
        med = float(row[colmap["median"]])
        q1  = float(row[colmap["q1"]])
        q3  = float(row[colmap["q3"]])
        anchor = meta["anchor"]
        rel = abs(med - anchor) / (abs(anchor) + 1e-12)
        recs.append({
            "dataset": row[colmap["dataset"]],
            "feature": meta["pretty"],
            "unit": meta["unit"],
            "median": med, "q1": q1, "q3": q3,
            "anchor": anchor,
            "rel_error": rel,
            "band": classify_band(rel),
        })
    out = pd.DataFrame(recs)
    # keep top-k anchored features per dataset (lowest rel_error) for main-text visualization
    if args.topk > 0:
        out = (out.sort_values(["dataset","rel_error"])
                  .groupby("dataset", as_index=False)
                  .head(args.topk))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # Optional: simple heatmap-like panel (traffic-light colors) for Fig. 2
    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np
        # one panel per dataset
        for dset, sub in out.groupby("dataset"):
            sub = sub.sort_values("rel_error")
            fig, ax = plt.subplots(figsize=(6, 1 + 0.5*len(sub)))
            y = np.arange(len(sub))
            colors = sub["band"].map({"green":"#3CB371","yellow":"#FFD700","red":"#CD5C5C"}).fillna("#D3D3D3")
            ax.barh(y, np.ones_like(y), color=colors)
            labels = [f"{f} ({u})\n{m:.2f} vs {a:.2f} ({100*re:.2f}%)"
                      for f,u,m,a,re in zip(sub["feature"], sub["unit"], sub["median"], sub["anchor"], sub["rel_error"])]
            ax.set_yticks(y); ax.set_yticklabels(labels)
            ax.set_xlim(0,1); ax.set_xticks([])
            ax.set_title(f"{dset}: threshold–anchor agreement")
            fig.tight_layout()
            # save per-panel and/or a combined PNG via caller's path convention
            fig.savefig(Path(args.plot).with_name(f"{Path(args.plot).stem}_{_norm(dset)}.png"), dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--thresholds", required=True, help="CSV with per-feature median [Q1,Q3] thresholds in natural units")
    p.add_argument("--guidelines", required=True, help="YAML with anchors/units")
    p.add_argument("--out", required=True, help="Output CSV with rel_error and traffic-light bands")
    p.add_argument("--topk", type=int, default=5, help="Top-k anchored features per dataset for main figure")
    p.add_argument("--plot", default=None, help="Optional PNG path prefix for simple panels")
    main(p.parse_args())
