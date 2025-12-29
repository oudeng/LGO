"""
Logistic函数与阈值阶跃示意图

该脚本生成一个图像，展示Logistic（Sigmoid）函数曲线与对应的阈值阶跃虚线。
Logistic曲线使用橙色实线表示，阈值阶跃使用黑色虚线表示。

Usage:
    python utility_plots/plot_logistic_threshold.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ============== 可调参数 ==============
# 画面尺寸与纵横比
FIG_WIDTH = 8          # 宽度（英寸）
FIG_HEIGHT = 6         # 高度（英寸）

# Logistic曲线参数
LOGISTIC_COLOR = '#E67300'   # 橙色
LOGISTIC_WIDTH = 12           # 线宽

# 阈值虚线参数
THRESHOLD_COLOR = 'black'    # 黑色
THRESHOLD_WIDTH = 12          # 线宽
THRESHOLD_STYLE = '--'       # 虚线样式: '--', ':', '-.'

# X轴范围
X_MIN, X_MAX = -6, 6
# =====================================


def logistic(x):
    """Logistic (Sigmoid) 函数"""
    return 1 / (1 + np.exp(-x))


def plot_logistic_threshold():
    # 创建图形
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # 生成数据
    x = np.linspace(X_MIN, X_MAX, 500)
    y = logistic(x)
    
    # 绘制logistic曲线
    ax.plot(x, y, color=LOGISTIC_COLOR, linewidth=LOGISTIC_WIDTH, 
            label='Logistic function', zorder=2)
    
    # 绘制阈值阶跃虚线（过中点 x=0, y=0.5）
    # 水平线: y = 0.5 (x < 0) 和 y = 0.5 → 1 的跳跃后 y = 1 (x >= 0)
    ax.hlines(y=0, xmin=X_MIN, xmax=0, colors=THRESHOLD_COLOR, 
              linewidth=THRESHOLD_WIDTH, linestyles=THRESHOLD_STYLE, zorder=1)
    ax.vlines(x=0, ymin=0, ymax=1, colors=THRESHOLD_COLOR, 
              linewidth=THRESHOLD_WIDTH, linestyles=THRESHOLD_STYLE, zorder=1)
    ax.hlines(y=1, xmin=0, xmax=X_MAX, colors=THRESHOLD_COLOR, 
              linewidth=THRESHOLD_WIDTH, linestyles=THRESHOLD_STYLE, zorder=1)
    
    # 移除边框和刻度（保持简洁示意风格）
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('utility_plots/logistic_threshold.png', dpi=600, bbox_inches='tight', 
                transparent=True, edgecolor='none')
    plt.show()
    print("图像已保存为 utility_plots/logistic_threshold.png")


if __name__ == '__main__':
    plot_logistic_threshold()