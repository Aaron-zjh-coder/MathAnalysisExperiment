import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 模拟数据
omega = np.linspace(0.25, 1.75, 100)
iterations = np.where(omega < 1.0, 20/(omega+0.5), np.where(omega < 1.2, 4 + 10*(omega-1.0)**2, 5 + 50*(omega-1.2)**2))
spectral_radius = np.where(omega < 1.0, 1 - 0.5*omega, np.where(omega < 1.5, omega - 0.8, 1.5 - 0.3*(omega-1.5)))

# 绘制迭代步数曲线
ax.plot(omega, iterations, 'b-', linewidth=2, label='迭代步数')
ax.scatter([1.0], [4], color='r', s=100, zorder=5)  # 标记最佳点

# 绘制谱半径曲线
ax2 = ax.twinx()
ax2.plot(omega, spectral_radius, 'g--', linewidth=2, label='谱半径')
ax2.set_ylim(0, 1.1)

# 添加标注
ax.annotate('最佳ω=1.00 步数=4', xy=(1.0, 4), xytext=(1.1, 10),
            arrowprops=dict(facecolor='black', shrink=0.05),
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
ax.text(1.6, 30, '谱半径=1\n(收敛边界)', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

# 添加水平虚线表示收敛边界
ax.axhline(y=4, color='gray', linestyle=':', alpha=0.5)
ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

# 设置坐标轴标签
ax.set_xlabel('松弛因子 ω', fontsize=12)
ax.set_ylabel('达到误差t=0所需迭代步数', color='b', fontsize=12)
ax2.set_ylabel('迭代矩阵谱半径', color='g', fontsize=12)

# 设置刻度
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax2.yaxis.set_major_locator(MultipleLocator(0.25))

# 设置标题和图例
ax.set_title('SOR迭代法松弛因子对收敛速度的影响', fontsize=14, pad=20)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# 网格线
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()