#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, pandas as pd
from pathlib import Path
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--dataset", default="CUSTOM")
    args = p.parse_args()
    d = Path(args.dataset_dir); ag = d/"aggregated"; ag.mkdir(parents=True, exist_ok=True)
    rows=[]
    runs = (d/"runs")
    if runs.exists():
        for hp in runs.glob("**/hparams.json"):
            data = json.loads(hp.read_text())
            rows.append(data)
    pd.DataFrame(rows).to_csv(ag/"hyperparams.csv", index=False)
    print("[OK] Wrote hyperparams.csv")
if __name__ == "__main__":
    main()
