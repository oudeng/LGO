#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended version of 09_plot_clinical_baselines_icu_eicu.py

- ICU: LGO (500k_full) + AutoScore (500k_full) + EBM (30k)
- eICU: LGO (30k) + EBM (30k) [+ AutoScore if eICU AutoScore results are available]

Differences vs. the original script
-----------------------------------
1. Still reads the aggregated `multiseed_summary_*.csv` files to draw AUROC/Brier
   bar plots with mean ± std for each method.
2. Additionally reads the corresponding `multiseed_results_*.csv` files and overlays
   one scatter point per seed on top of each bar (AUROC on the left axis, Brier on
   the right axis).
3. Optionally includes AutoScore on eICU if an eICU AutoScore summary file is found
   under:
       LGO_AutoScore_v3_8/eICU_results_500k_full/
       or
       LGO_AutoScore_v3_8/eICU_results_30k_full/

Usage (from project root):
    python utility_plots/09_plot_clinical_baselines_icu_eicu.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_summary(path: Path) -> pd.DataFrame:
    """Load multiseed_summary_*.csv and return method / AUROC_mean / AUROC_std / Brier_mean / Brier_std."""
    df = pd.read_csv(path)

    # Keep only rows with actual method names (drop header rows)
    df = df[df["Unnamed: 0"].notna() & (df["Unnamed: 0"] != "method")].copy()
    df = df.rename(columns={"Unnamed: 0": "method"})

    # Ensure numeric columns
    for col in ["AUROC", "AUROC.1", "Brier", "Brier.1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["method", "AUROC", "AUROC.1", "Brier", "Brier.1"]]


def summary_to_results_path(summary_path: Path) -> Path:
    """Infer multiseed_results_*.csv path from a multiseed_summary_*.csv path."""
    name = summary_path.name
    if "summary" not in name:
        raise ValueError(f"Expected 'summary' in filename, got: {name}")
    results_name = name.replace("summary", "results")
    return summary_path.with_name(results_name)


def load_results(summary_path: Path) -> pd.DataFrame:
    """Load the corresponding multiseed_results_*.csv for a given summary file."""
    results_path = summary_to_results_path(summary_path)
    if not results_path.is_file():
        raise FileNotFoundError(
            f"multiseed_results CSV not found for {summary_path} "
            f"(expected: {results_path})"
        )
    df = pd.read_csv(results_path)
    # Ensure numeric columns
    for col in ["AUROC", "Brier"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_latest_summary(dir_path: Path) -> Path | None:
    """Pick the latest multiseed_summary_*.csv inside dir_path (or return None)."""
    if not dir_path.is_dir():
        return None
    candidates = sorted(dir_path.glob("multiseed_summary_*.csv"))
    return candidates[-1] if candidates else None


def main() -> None:
    # --- 1) Locate summary CSVs (update here if your paths differ) ---
    base = Path(".")

    # ICU: AutoScore + LGO (500k_full)
    icu_auto_dir_500k = base / "LGO_AutoScore_v3_8" / "ICU_results_500k_full"
    icu_autoscore_csv = pick_latest_summary(icu_auto_dir_500k)

    # ICU: LGO + EBM (30k)
    icu_ebm_dir_30k = base / "LGO_Interpret_v1_4" / "ICU_results_30k"
    icu_ebm_csv = pick_latest_summary(icu_ebm_dir_30k)

    # eICU: LGO + EBM (30k)
    eicu_ebm_dir_30k = base / "LGO_Interpret_v1_4" / "eICU_results_30k"
    eicu_ebm_csv = pick_latest_summary(eicu_ebm_dir_30k)

    # Optional: eICU AutoScore (try 500k_full then 30k_full)
    eicu_auto_csv = None
    for sub in ["eICU_results_500k_full", "eICU_results_30k_full"]:
        cand = pick_latest_summary(base / "LGO_AutoScore_v3_8" / sub)
        if cand is not None:
            eicu_auto_csv = cand
            break

    # Basic sanity checks for the files we *do* require
    assert icu_autoscore_csv is not None, (
        "ICU AutoScore summary not found under "
        "LGO_AutoScore_v3_8/ICU_results_500k_full/"
    )
    assert icu_ebm_csv is not None, (
        "ICU EBM summary not found under "
        "LGO_Interpretv_v1_4/ICU_results_30k/"
    )
    assert eicu_ebm_csv is not None, (
        "eICU EBM summary not found under "
        "LGO_Interpretv_v1_4/eICU_results_30k/"
    )

    print(f"[INFO] ICU AutoScore summary: {icu_autoscore_csv}")
    print(f"[INFO] ICU EBM summary:      {icu_ebm_csv}")
    print(f"[INFO] eICU EBM summary:     {eicu_ebm_csv}")
    if eicu_auto_csv is not None:
        print(f"[INFO] eICU AutoScore summary: {eicu_auto_csv}")
    else:
        print("[WARN] No eICU AutoScore summary found; eICU panel will show LGO + EBM only.")

    # --- 2) Read summaries ---
    icu_auto_df = load_summary(icu_autoscore_csv)   # AutoScore + LGO (ICU, 500k_full)
    icu_ebm_df = load_summary(icu_ebm_csv)          # EBM + LGO (ICU, 30k)
    eicu_ebm_df = load_summary(eicu_ebm_csv)        # EBM + LGO (eICU, 30k)

    if eicu_auto_csv is not None:
        eicu_auto_df = load_summary(eicu_auto_csv)  # AutoScore (and possibly LGO) on eICU
    else:
        eicu_auto_df = None

    # --- 3) Build aggregated stats dict ---
    # data[dataset][method] = {AUROC_mean, AUROC_std, Brier_mean, Brier_std}
    data: dict[str, dict[str, dict[str, float]]] = {"ICU": {}, "eICU": {}}

    # ICU: AutoScore & LGO from 500k_full
    for _, row in icu_auto_df.iterrows():
        method = row["method"]  # "AutoScore" or "LGO"
        data["ICU"][method] = {
            "AUROC_mean": row["AUROC"],
            "AUROC_std": row["AUROC.1"],
            "Brier_mean": row["Brier"],
            "Brier_std": row["Brier.1"],
        }

    # ICU: EBM from ICU_results_30k
    for _, row in icu_ebm_df.iterrows():
        method = row["method"]
        if method != "EBM":
            continue
        data["ICU"]["EBM"] = {
            "AUROC_mean": row["AUROC"],
            "AUROC_std": row["AUROC.1"],
            "Brier_mean": row["Brier"],
            "Brier_std": row["Brier.1"],
        }

    # eICU: LGO + EBM from eICU_results_30k
    for _, row in eicu_ebm_df.iterrows():
        method = row["method"]  # "LGO" or "EBM"
        data["eICU"][method] = {
            "AUROC_mean": row["AUROC"],
            "AUROC_std": row["AUROC.1"],
            "Brier_mean": row["Brier"],
            "Brier_std": row["Brier.1"],
        }

    # eICU: AutoScore if available
    if eicu_auto_df is not None:
        for _, row in eicu_auto_df.iterrows():
            method = row["method"]
            if method != "AutoScore":
                continue
            data["eICU"]["AutoScore"] = {
                "AUROC_mean": row["AUROC"],
                "AUROC_std": row["AUROC.1"],
                "Brier_mean": row["Brier"],
                "Brier_std": row["Brier.1"],
            }

    # --- 4) Read per-seed results for scatter plots ---
    # results[dataset][method] = DataFrame with columns AUROC, Brier, ...
    results: dict[str, dict[str, pd.DataFrame]] = {"ICU": {}, "eICU": {}}

    # ICU: per-seed AutoScore + LGO from 500k_full
    icu_auto_res = load_results(icu_autoscore_csv)
    for method in ["AutoScore", "LGO"]:
        sub = icu_auto_res[icu_auto_res["method"] == method].copy()
        if not sub.empty:
            results["ICU"][method] = sub

    # ICU: per-seed EBM from ICU_results_30k
    icu_ebm_res = load_results(icu_ebm_csv)
    sub_ebm = icu_ebm_res[icu_ebm_res["method"] == "EBM"].copy()
    if not sub_ebm.empty:
        results["ICU"]["EBM"] = sub_ebm

    # eICU: per-seed LGO + EBM from eICU_results_30k
    eicu_ebm_res = load_results(eicu_ebm_csv)
    for method in ["LGO", "EBM"]:
        sub = eicu_ebm_res[eicu_ebm_res["method"] == method].copy()
        if not sub.empty:
            results["eICU"][method] = sub

    # eICU: per-seed AutoScore if available
    if eicu_auto_csv is not None:
        try:
            eicu_auto_res = load_results(eicu_auto_csv)
            sub_auto = eicu_auto_res[eicu_auto_res["method"] == "AutoScore"].copy()
            if not sub_auto.empty:
                results["eICU"]["AutoScore"] = sub_auto
        except FileNotFoundError:
            print("[WARN] eICU AutoScore multiseed_results_* not found; no per-seed scatter for eICU AutoScore.")

    # --- 5) Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    datasets = ["ICU", "eICU"]
    # Desired method order; methods missing in `data[ds]` will be skipped
    method_order = {
        "ICU":  ["AutoScore", "LGO", "EBM"],
        "eICU": ["AutoScore", "LGO", "EBM"],
    }

    for ax, ds in zip(axes, datasets):
        methods = [m for m in method_order[ds] if m in data[ds]]
        x = np.arange(len(methods))
        width = 0.6

        auroc_means = [data[ds][m]["AUROC_mean"] for m in methods]
        auroc_stds  = [data[ds][m]["AUROC_std"]  for m in methods]
        brier_means = [data[ds][m]["Brier_mean"] for m in methods]
        brier_stds  = [data[ds][m]["Brier_std"]  for m in methods]

        # AUROC bars (left axis)
        bars_auroc = ax.bar(
            x - width / 4,
            auroc_means,
            width / 2,
            yerr=auroc_stds,
            capsize=3,
            label="AUROC",
        )
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("AUROC")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(ds)

        # Per-seed AUROC scatter
        for i, m in enumerate(methods):
            if m not in results[ds]:
                continue
            df_res = results[ds][m]
            xs = np.full(len(df_res), x[i] - width / 4)
            ax.scatter(xs, df_res["AUROC"], s=12, alpha=0.6, color="black", zorder=3)

        # Brier bars (right axis, inverted so "lower is better" looks higher)
        ax2 = ax.twinx()
        bars_brier = ax2.bar(
            x + width / 4,
            brier_means,
            width / 2,
            yerr=brier_stds,
            capsize=3,
            alpha=0.5,
            label="Brier",
        )
        ax2.invert_yaxis()
        ax2.set_ylabel("Brier (lower is better)")

        # Per-seed Brier scatter
        for i, m in enumerate(methods):
            if m not in results[ds]:
                continue
            df_res = results[ds][m]
            xs = np.full(len(df_res), x[i] + width / 4)
            ax2.scatter(xs, df_res["Brier"], s=12, alpha=0.6, color="black", zorder=3)

        # Legend per panel
        handles = [bars_auroc[0], bars_brier[0]]
        labels = ["AUROC", "Brier"]
        ax.legend(handles, labels, loc="upper left")

    fig.suptitle("Clinical baselines on ICU and eICU (LGO vs AutoScore vs EBM)",
                 fontweight='bold', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 1.00])  # Reduced title-subplot spacing (was 0.93)

    out_dir = Path("utility_plots") / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "clinical_baselines_icu_eicu.png"
    pdf_path = out_dir / "clinical_baselines_icu_eicu.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    print(f"[OK] Saved figure to: {png_path}")
    print(f"[OK] Saved figure to: {pdf_path}")


if __name__ == "__main__":
    main()