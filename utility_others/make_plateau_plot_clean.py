
"""make_plateau_plot_clean.py

生成一幅简洁的示意图，展示阈值损失 L(b) 的平原-悬崖结构和网格搜索过程。
不含标注和箭头，方便后续手动添加。

python utility_plots/make_plateau_plot_clean.py

output: utility_plots/loss_threshold_plot_clean.png

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_plateau_plot(n=100, true_threshold=0.4, num_candidates=9,
                          seed=0, outpath="utility_plots/loss_threshold_plot_clean.png"):
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
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 绘制 Loss 曲线
    ax.plot(bs, losses, label=r"Misclassification error $L(b)$", color="tab:blue")
    
    # 绘制候选阈值点
    ax.scatter(candidate_bs, candidate_losses, color="black", 
               label="Candidate thresholds", zorder=5)
    
    # 绘制最优候选点
    ax.scatter(candidate_bs[best_idx], candidate_losses[best_idx], color="red",
               marker="*", s=150, label="Best candidate", zorder=6)

    # 坐标轴设置
    ax.set_xlabel(r"Threshold $b$", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(r"Plateau structure and grid search for $L(b)$", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.tick_params(axis='both', labelsize=12)

    # 保存
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {outpath}")
    
    # 输出候选点坐标，方便后续添加标注时参考
    print("\n=== Candidate thresholds coordinates (for annotation reference) ===")
    for i, (b_val, loss_val) in enumerate(zip(candidate_bs, candidate_losses)):
        marker = " <-- Best" if i == best_idx else ""
        print(f"  [{i}] b = {b_val:.4f}, loss = {loss_val:.4f}{marker}")
    print(f"\nTrue threshold: {true_threshold}")

if __name__ == "__main__":
    generate_plateau_plot()