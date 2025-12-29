"""make_plateau_plot.py

Add loss_threshold_plot_final.png to \paragraph{Why coordinate descent?}
生成一幅示意图，展示阈值损失 L(b) 的平原-悬崖结构和网格搜索过程。


python utility_plots/make_plateau_plot.py

output: loss_threshold_plot_final.png。

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_plateau_plot(n=100, true_threshold=0.4, num_candidates=9,
                          seed=0, outpath="utility_plots/loss_threshold_plot_final.png"):
    # 数据模拟
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = (x >= true_threshold).astype(int)

    # 计算硬门控模型的误分类率 L(b)
    bs = np.linspace(x.min(), x.max(), 500)
    losses = [((x >= b).astype(int) != y).mean() for b in bs]

    # 网格搜索候选阈值：取 x 的若干分位点
    quantiles = np.linspace(0.1, 0.9, num_candidates)
    candidate_bs = np.quantile(x, quantiles)
    candidate_losses = [((x >= cb).astype(int) != y).mean() for cb in candidate_bs]
    best_idx = int(np.argmin(candidate_losses))

    # 作图 - 设置全局字体大小为 12pt
    plt.rcParams.update({'font.size': 12})
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bs, losses, label=r"Misclassification error $L(b)$", color="tab:blue")
    ax.scatter(candidate_bs, candidate_losses, color="black", label="Candidate thresholds", zorder=5)
    ax.scatter(candidate_bs[best_idx], candidate_losses[best_idx], color="red",
               marker="*", s=150, label="Best candidate", zorder=6)  # 星号也稍微放大

    ax.set_xlabel(r"Threshold $b$", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(r"Plateau structure and grid search for $L(b)$", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)  # 图例稍小一点避免过于拥挤
    ax.tick_params(axis='both', labelsize=12)

    # 标注：平原、跳跃和候选点说明
    # 标注1: Plateaus where gradient=0 (保持原位置)
    plateau_b = 0.2
    plateau_loss = losses[int(np.searchsorted(bs, plateau_b))]
    ax.annotate("Plateaus where\ngradient=0", xy=(plateau_b, plateau_loss),
                xytext=(0.05, 0.50), arrowprops=dict(arrowstyle="->", lw=1.2),
                fontsize=12)
    
    # 标注2: Jump at data point - 往左上方移动，避免与曲线重叠
    jump_loss = losses[int(np.searchsorted(bs, true_threshold))]
    ax.annotate("Jump at data point", xy=(true_threshold + 0.02, jump_loss + 0.02),
                xytext=(0.48, 0.32),  # 往左移动（原来是0.53）
                arrowprops=dict(arrowstyle="->", lw=1.2),
                fontsize=12)
    
    # 标注3: Candidate search points - 往左移动，箭头指向更靠左的候选点
    cand_idx = num_candidates // 2 + 1  # 选择稍靠右的候选点作为箭头目标
    ax.annotate("Candidate search points\n(quantiles)",
                xy=(candidate_bs[cand_idx], candidate_losses[cand_idx]),
                xytext=(0.62, 0.52),  # 往左移动（原来是0.75, 0.55）
                arrowprops=dict(arrowstyle="->", lw=1.2), 
                fontsize=12)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {outpath}")

if __name__ == "__main__":
    generate_plateau_plot()