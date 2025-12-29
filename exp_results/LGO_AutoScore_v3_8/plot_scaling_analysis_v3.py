#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_scaling_analysis.py - LGO vs AutoScore 资源缩放效应可视化
================================================================
Version: 3.0.0
Date: Dec 5, 2025

改进:
- 统一使用半透明标签显示p值（移除星号标注）
- 子图a)和d)采用一致的标注风格
- 95% CI误差棒 + p值标签清晰区分

Usage:
------
# ICU
python plot_scaling_analysis_v3.py \
  --results_dirs ICU_results_30k_full,ICU_results_100k_full,ICU_results_200k_full,ICU_results_300k_full,ICU_results_500k_full \
  --output ICU_scaling_analysis_full.png

python plot_scaling_analysis_v3.py \
  --results_dirs ICU_results_30k_raw,ICU_results_100k_raw,ICU_results_200k_raw,ICU_results_300k_raw,ICU_results_500k_raw \
  --output ICU_scaling_analysis_raw.png

# eICU
python plot_scaling_analysis_v3.py \
  --results_dirs eICU_results_30k_full,eICU_results_100k_full,eICU_results_200k_full,eICU_results_300k_full \
  --output eICU_scaling_analysis_full.png

# NHANES
python plot_scaling_analysis_v3.py \
  --results_dirs NHANES_results_30k_full,NHANES_results_100k_full,NHANES_results_200k_full,NHANES_results_300k_full \
  --output NHANES_scaling_analysis_full.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2


def load_data(results_dirs):
    """从CSV文件加载数据"""
    data = {
        'configs': [],
        'lgo_auroc_mean': [], 'lgo_auroc_std': [],
        'autoscore_auroc_mean': [], 'autoscore_auroc_std': [],
        'lgo_auprc_mean': [], 'autoscore_auprc_mean': [],
        'wilcoxon_p': [], 'win_rate': [],
        'delta_auroc': [], 'ci_lower': [], 'ci_upper': []
    }
    
    for rdir in results_dirs:
        config = Path(rdir).name.replace('results_', '')
        csv_files = list(Path(rdir).glob("multiseed_results_*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV found in {rdir}, skipping...")
            continue
        
        df = pd.read_csv(sorted(csv_files)[-1])
        lgo = df[df['method'] == 'LGO']
        auto = df[df['method'] == 'AutoScore']
        merged = lgo.merge(auto, on='seed', suffixes=('_lgo', '_as'))
        
        data['configs'].append(config)
        data['lgo_auroc_mean'].append(merged['AUROC_lgo'].mean())
        data['lgo_auroc_std'].append(merged['AUROC_lgo'].std())
        data['autoscore_auroc_mean'].append(merged['AUROC_as'].mean())
        data['autoscore_auroc_std'].append(merged['AUROC_as'].std())
        data['lgo_auprc_mean'].append(merged['AUPRC_lgo'].mean())
        data['autoscore_auprc_mean'].append(merged['AUPRC_as'].mean())
        
        # Wilcoxon检验
        try:
            _, p = wilcoxon(merged['AUROC_lgo'], merged['AUROC_as'])
        except:
            p = 1.0
        data['wilcoxon_p'].append(float(p))
        
        # 胜率
        diff = merged['AUROC_lgo'] - merged['AUROC_as']
        data['win_rate'].append(int((diff > 0).sum()))
        data['delta_auroc'].append(float(diff.mean()))
        
        # Bootstrap CI
        np.random.seed(42)
        n = len(merged)
        boot_diffs = [merged['AUROC_lgo'].sample(n, replace=True).mean() - 
                      merged['AUROC_as'].sample(n, replace=True).mean() 
                      for _ in range(10000)]
        data['ci_lower'].append(float(np.percentile(boot_diffs, 2.5)))
        data['ci_upper'].append(float(np.percentile(boot_diffs, 97.5)))
    
    return data


def format_pvalue(p):
    """格式化p值显示"""
    if p < 0.001:
        return 'p<0.001'
    elif p < 0.01:
        return 'p<0.01'
    elif p < 0.05:
        return 'p<0.05'
    else:
        return f'p={p:.2f}'


def plot_scaling_analysis(data, output_path='scaling_analysis.png', dpi=150):
    """生成四子图可视化"""
    
    configs = data['configs']
    x = np.arange(len(configs))
    
    # 颜色
    C_LGO, C_AUTO = '#2ecc71', '#3498db'
    C_SIG, C_NSIG, C_MARG = '#2ecc71', '#95a5a6', '#f39c12'
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('LGO vs AutoScore: Performance Scaling Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # === Panel A: AUROC ===
    ax1 = axes[0, 0]
    ax1.errorbar(x, data['lgo_auroc_mean'], yerr=data['lgo_auroc_std'], 
                 fmt='o-', color=C_LGO, lw=2.5, ms=10, capsize=6, capthick=2, label='LGO')
    ax1.fill_between(x, 
                     np.array(data['lgo_auroc_mean']) - np.array(data['lgo_auroc_std']),
                     np.array(data['lgo_auroc_mean']) + np.array(data['lgo_auroc_std']),
                     alpha=0.15, color=C_LGO)
    ax1.errorbar(x, data['autoscore_auroc_mean'], yerr=data['autoscore_auroc_std'],
                 fmt='s--', color=C_AUTO, lw=2.5, ms=10, capsize=6, capthick=2, label='AutoScore')
    
    # 设置y轴范围
    ax1.set_ylim(0.80, 1.02)
    
    # 添加p值标签（半透明背景，与子图d统一风格）
    for i, p in enumerate(data['wilcoxon_p']):
        if p < 0.05:  # 仅显示显著结果
            y_pos = data['lgo_auroc_mean'][i] + data['lgo_auroc_std'][i] + 0.012
            y_pos = min(y_pos, 1.005)
            label = format_pvalue(p)
            ax1.text(x[i], y_pos, label, ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#c0392b',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             edgecolor='#c0392b', alpha=0.8, linewidth=1))
    
    ax1.set_xticks(x); ax1.set_xticklabels(configs)
    ax1.set_xlabel('Computational Budget', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax1.set_title('a) AUROC vs Computational Resources', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right'); ax1.grid(alpha=0.3)
    
    # === Panel B: AUPRC ===
    ax2 = axes[0, 1]
    ax2.plot(x, data['lgo_auprc_mean'], 'o-', color=C_LGO, lw=2.5, ms=10, label='LGO')
    ax2.plot(x, data['autoscore_auprc_mean'], 's--', color=C_AUTO, lw=2.5, ms=10, label='AutoScore')
    ax2.set_xticks(x); ax2.set_xticklabels(configs)
    ax2.set_xlabel('Computational Budget', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
    ax2.set_title('b) AUPRC vs Computational Resources', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right'); ax2.set_ylim(0.70, 1.0); ax2.grid(alpha=0.3)
    
    # === Panel C: Win Rate ===
    ax3 = axes[1, 0]
    colors = [C_SIG if w >= 8 else C_MARG if w >= 6 else C_NSIG for w in data['win_rate']]
    bars = ax3.bar(x, data['win_rate'], color=colors, edgecolor='black', lw=1.5, width=0.6)
    ax3.axhline(y=5, color='gray', ls='--', lw=1.5, label='50% threshold')
    for i, (bar, w) in enumerate(zip(bars, data['win_rate'])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.8,
                 f'{w}/10', ha='center', va='top', fontsize=12, fontweight='bold',
                 color='white' if w >= 6 else 'black')
    ax3.set_xticks(x); ax3.set_xticklabels(configs)
    ax3.set_xlabel('Computational Budget', fontsize=12, fontweight='bold')
    ax3.set_ylabel('LGO Win Count (out of 10)', fontsize=12, fontweight='bold')
    ax3.set_title('c) LGO Win Rate Across Seeds', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right'); ax3.set_ylim(0, 11); ax3.grid(alpha=0.3, axis='y')
    
    # === Panel D: Delta with CI ===
    ax4 = axes[1, 1]
    yerr_lo = [d - l for d, l in zip(data['delta_auroc'], data['ci_lower'])]
    yerr_hi = [u - d for d, u in zip(data['delta_auroc'], data['ci_upper'])]
    bar_colors = [C_NSIG if l < 0 else C_SIG for l in data['ci_lower']]
    bars = ax4.bar(x, data['delta_auroc'], yerr=[yerr_lo, yerr_hi], 
                   color=bar_colors, edgecolor='black', lw=1.5, capsize=6, width=0.6)
    ax4.axhline(y=0, color='#e74c3c', lw=2)
    
    # 动态计算y轴上限
    max_ci_upper = max(data['ci_upper'])
    y_upper = max(0.18, max_ci_upper + 0.04)
    ax4.set_ylim(-0.02, y_upper)
    
    # 添加p值标签（半透明背景，与子图a统一风格）
    for i, (bar, p) in enumerate(zip(bars, data['wilcoxon_p'])):
        label = format_pvalue(p)
        label_y = min(data['ci_upper'][i] + 0.012, y_upper - 0.015)
        ax4.text(bar.get_x() + bar.get_width()/2, label_y, label,
                ha='center', va='bottom', fontsize=9,
                fontweight='bold' if p < 0.05 else 'normal',
                color='#c0392b' if p < 0.05 else '#7f8c8d',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='#c0392b' if p < 0.05 else '#bdc3c7', 
                         alpha=0.8, linewidth=1))
    
    ax4.set_xticks(x); ax4.set_xticklabels(configs)
    ax4.set_xlabel('Computational Budget', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Δ AUROC (LGO − AutoScore)', fontsize=12, fontweight='bold')
    # 修改标题：更准确地描述内容
    ax4.set_title('d) AUROC Difference (95% CI) with Wilcoxon Test', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"[SAVE] {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate scaling analysis plot')
    parser.add_argument('--results_dirs', type=str, required=True,
                        help='Comma-separated list of result directories')
    parser.add_argument('--output', type=str, default='scaling_analysis.png')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()
    
    print("Loading data from CSV files...")
    result_dirs = [d.strip() for d in args.results_dirs.split(',')]
    data = load_data(result_dirs)
    
    if not data['configs']:
        print("[ERROR] No data loaded!")
        return
    
    print(f"Configs: {data['configs']}")
    print(f"LGO AUROC: {[f'{m:.3f}±{s:.3f}' for m,s in zip(data['lgo_auroc_mean'], data['lgo_auroc_std'])]}")
    print(f"Win rates: {data['win_rate']}")
    
    plot_scaling_analysis(data, args.output, args.dpi)
    print("Done!")


if __name__ == '__main__':
    main()