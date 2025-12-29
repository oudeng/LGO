#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
09_feature_threshold_sanity.py

对单个特征做阈值 sanity check：
- 画特征分布（按标签 0/1 分组），标出 guideline & LGO 阈值
- 画该特征的 ROC 曲线，并标出 guideline & LGO 阈值对应的 (FPR, TPR)
- 打印 guideline vs LGO 阈值在当前数据集上的分类指标（TPR/TNR/PPV/BAcc）

用法示例（1）：ICU, 乳酸（lactate）

假设已经在 ICU 的 CSV 里有一个二值标签列 high_risk（0/1）；
LGO 学到的阈值是 4.72 mmol/L，guideline 是 2.0 mmol/L：

python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --label_col high_risk \
  --feature_col lactate_mmol_l \
  --guideline 2.0 \
  --lgo_threshold 4.72 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/ICU_lactate

说明：
- 如果 label_col 已经是 0/1，可以不加 --label_threshold；
- 如果 label_col 是连续风险分数，可以用 --label_threshold 把它二值化，比如 0.5；

如果现在的标签其实是连续的 composite_risk_score，可以例如用 0.5 把它二值化：

python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 0.5 \
  --feature_col lactate_mmol_l \
  --guideline 2.0 \
  --lgo_threshold 4.72 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/ICU_lactate

用法示例（2）：eICU，GCS（低值更危险）

python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --label_col outcome_bin \
  --label_threshold 0.5 \
  --label_positive_below \
  --feature_col gcs_min \
  --guideline 8 \
  --lgo_threshold 13.0 \
  --outdir utility_analysis/sanity_plots/eICU_gcs
  # 注意：这里不加 --higher_is_riskier，默认"低值更危险"

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="原始数据 CSV 路径")
    p.add_argument("--label_col", required=True, help="标签列名（可以是0/1或连续）")
    p.add_argument("--feature_col", required=True, help="特征列名")
    p.add_argument("--guideline", type=float, required=True, help="guideline 阈值（物理单位）")
    p.add_argument("--lgo_threshold", type=float, required=True, help="LGO 学到的阈值（物理单位）")

    # 标签二值化相关
    p.add_argument(
        "--label_threshold",
        type=float,
        default=None,
        help="如果标签是连续的，用该阈值将其二值化：y = (label >= label_threshold)",
    )
    p.add_argument(
        "--label_positive_below",
        action="store_true",
        help="如果设定，则 y = (label <= label_threshold)，默认 False（>=）",
    )

    # 特征风险方向：默认“越大越危险”
    p.add_argument(
        "--higher_is_riskier",
        action="store_true",
        help="若指定，则预测规则是 feature >= threshold 为阳性；否则 feature <= threshold 为阳性",
    )

    p.add_argument("--outdir", default="sanity_plots", help="输出图表目录")
    return p.parse_args()


def binarize_label(y_raw, label_threshold, positive_below=False):
    """把标签变成 0/1."""
    y_raw = np.asarray(y_raw)

    # 指定了阈值：按阈值二值化
    if label_threshold is not None:
        if positive_below:
            y = (y_raw <= label_threshold).astype(int)
        else:
            y = (y_raw >= label_threshold).astype(int)
        return y

    # 未指定阈值：如果本来就是二值
    uniq = np.unique(y_raw[~pd.isna(y_raw)])
    if len(uniq) == 2:
        # 映射成 {0,1}
        # 兼容 {0,1}, {1,2}, {"no","yes"} 等
        # 把“较大/较晚的”值映射为 1
        val0, val1 = sorted(uniq)
        y = (y_raw == val1).astype(int)
        return y

    raise ValueError(
        "label_threshold 未指定，而标签又不是二值的。"
        "请使用 --label_threshold 将连续标签二值化。"
    )


def metrics_for_threshold(x, y, thr, higher_is_riskier=True):
    """计算给定阈值下的一些常用指标."""
    if higher_is_riskier:
        y_pred = (x >= thr).astype(int)
    else:
        y_pred = (x <= thr).astype(int)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # 基本指标
    eps = 1e-9
    tpr = tp / (tp + fn + eps)  # recall / sensitivity
    tnr = tn / (tn + fp + eps)  # specificity
    ppv = precision_score(y, y_pred, zero_division=0)
    # npv 不一定需要
    bacc = 0.5 * (tpr + tnr)

    return {
        "thr": thr,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "TPR": tpr,
        "TNR": tnr,
        "PPV": ppv,
        "BAcc": bacc,
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        raise ValueError(f"label_col '{args.label_col}' 不在数据中")
    if args.feature_col not in df.columns:
        raise ValueError(f"feature_col '{args.feature_col}' 不在数据中")

    y_raw = df[args.label_col].values
    x_raw = df[args.feature_col].values

    # 去掉缺失
    mask = ~pd.isna(y_raw) & ~pd.isna(x_raw)
    y_raw = y_raw[mask]
    x_raw = x_raw[mask]

    # 二值化标签
    y = binarize_label(
        y_raw,
        label_threshold=args.label_threshold,
        positive_below=args.label_positive_below,
    )

    # 特征得分：决定 ROC 的方向
    if args.higher_is_riskier:
        x_score = x_raw  # 值越大，风险越高
        inequality_str = ">="
    else:
        x_score = -x_raw  # 把“越小越危险”翻转
        inequality_str = "<="

    # 整体 AUC（单特征作为 score）
    auc = roc_auc_score(y, x_score)
    fpr, tpr, thr = roc_curve(y, x_score)

    print(f"\n[INFO] 数据点数: {len(y)} (pos={y.sum()}, neg={len(y)-y.sum()})")
    print(f"[INFO] 单特征 '{args.feature_col}' 的 AUC: {auc:.3f}")

    # guideline vs LGO 阈值的性能
    m_guideline = metrics_for_threshold(
        x_raw, y, args.guideline, higher_is_riskier=args.higher_is_riskier
    )
    m_lgo = metrics_for_threshold(
        x_raw, y, args.lgo_threshold, higher_is_riskier=args.higher_is_riskier
    )

    print("\n[INFO] 阈值对比（规则: pred = 1 if feature {} thr else 0）".format(inequality_str))
    rows = []
    for name, m in [("guideline", m_guideline), ("LGO", m_lgo)]:
        rows.append(
            {
                "name": name,
                "thr": m["thr"],
                "TP": m["TP"],
                "FP": m["FP"],
                "TN": m["TN"],
                "FN": m["FN"],
                "TPR": m["TPR"],
                "TNR": m["TNR"],
                "PPV": m["PPV"],
                "BAcc": m["BAcc"],
            }
        )
    df_metrics = pd.DataFrame(rows)
    pd.set_option("display.precision", 3)
    print(df_metrics.to_string(index=False))

    # ---------- 图1: 特征分布 + 阈值线 ----------
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    x_pos = x_raw[y == 1]
    x_neg = x_raw[y == 0]

    bins = 30
    ax.hist(x_neg, bins=bins, alpha=0.5, label="y=0", density=True)
    ax.hist(x_pos, bins=bins, alpha=0.5, label="y=1", density=True)

    ax.axvline(args.guideline, color="C2", linestyle="--", label=f"Guideline = {args.guideline:.3g}")
    ax.axvline(args.lgo_threshold, color="C3", linestyle="-.", label=f"LGO = {args.lgo_threshold:.3g}")

    ax.set_xlabel(args.feature_col)
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"Distribution of {args.feature_col} by label\n(guideline vs LGO threshold)")

    fig.tight_layout()
    fig_path1 = outdir / f"{args.feature_col}_dist.png"
    fig.savefig(fig_path1, dpi=200)
    plt.close(fig)
    print(f"[OK] 保存分布图: {fig_path1}")

    # ---------- 图2: ROC 曲线 + 标出两点 ----------
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")

    # guideline / LGO 点
    def get_tpr_fpr_for_thr(thr_value):
        # 按原始 x_raw 计算预测
        if args.higher_is_riskier:
            y_pred = (x_raw >= thr_value).astype(int)
        else:
            y_pred = (x_raw <= thr_value).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        tpr_v = tp / (tp + fn + 1e-9)
        fpr_v = fp / (fp + tn + 1e-9)
        return fpr_v, tpr_v

    fpr_g, tpr_g = get_tpr_fpr_for_thr(args.guideline)
    fpr_l, tpr_l = get_tpr_fpr_for_thr(args.lgo_threshold)

    ax.scatter([fpr_g], [tpr_g], color="C2", marker="o", label=f"Guideline ({fpr_g:.2f}, {tpr_g:.2f})")
    ax.scatter([fpr_l], [tpr_l], color="C3", marker="s", label=f"LGO ({fpr_l:.2f}, {tpr_l:.2f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR (1 - specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_title(f"ROC of {args.feature_col} (as score)\npoints: guideline vs LGO")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig_path2 = outdir / f"{args.feature_col}_roc.png"
    fig.savefig(fig_path2, dpi=200)
    plt.close(fig)
    print(f"[OK] 保存 ROC 图: {fig_path2}")


if __name__ == "__main__":
    main()