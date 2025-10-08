#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, pandas as pd
from pathlib import Path
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    args = p.parse_args()
    d = Path(args.dataset_dir); ag = d/"aggregated"; ag.mkdir(parents=True, exist_ok=True)
    rp = ag/"runtime_profile.csv"
    if rp.exists():
        print("[OK] runtime_profile.csv exists.")
    else:
        # create placeholder from candidates timestamps if any
        pd.DataFrame([{"phase":"fit","duration_s":0,"method":"","seed":0,"split":0}]).to_csv(rp, index=False)
        print("[WARN] Created placeholder runtime_profile.csv")
if __name__ == "__main__":
    main()
