import json
from pathlib import Path
import pandas as pd
def save_scaler(outdir, feature_means, feature_stds, zscore=True):
    out = {"feature_means": feature_means, "feature_stds": feature_stds, "zscore": bool(zscore)}
    p = Path(outdir)/"config"; p.mkdir(parents=True, exist_ok=True)
    (p/"scaler.json").write_text(json.dumps(out, indent=2))
def save_units(outdir, unit_map):
    p = Path(outdir)/"config"; p.mkdir(parents=True, exist_ok=True)
    lines = ["# feature units"] + [f"{k}: {unit_map.get(k,'')}" for k in unit_map]
    (p/"units.yaml").write_text("\n".join(lines))
def save_splits(outdir, schema, group_key, folds):
    p = Path(outdir)/"config"; p.mkdir(parents=True, exist_ok=True)
    obj = {"schema": schema, "group_key": group_key, "folds": folds}
    (p/"splits.json").write_text(json.dumps(obj, indent=2))
def save_hparams(outdir, method, seed, split, hparams):
    run_dir = Path(outdir)/"runs"/f"seed{seed}_split{split}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"hparams.json").write_text(json.dumps(hparams, indent=2))
def append_runtime_profile(outdir, method, seed, split, profile_rows):
    p = Path(outdir)/"aggregated"; p.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(profile_rows)
    df["method"] = method; df["seed"] = seed; df["split"] = split
    path = p/"runtime_profile.csv"
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)
