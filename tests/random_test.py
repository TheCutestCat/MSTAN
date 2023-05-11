import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

if __name__ == '__main__':
    # 假设 data 是你的数据
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    # 计算 KDE
    kde = gaussian_kde(data)

    # 生成用于评估的数据点
    x = np.linspace(min(data), max(data), 1000)
    y = kde.evaluate(x)

    x = np.linspace(min(x), max(x), 10000)
    # 计算并绘制 KDE
    # 计算 HPDI
    sorted_indices = np.argsort(y)[::-1]
    sorted_x = x[sorted_indices]
    cumulative_y = np.cumsum(y[sorted_indices])
    cumulative_y /= cumulative_y[-1]

    idx1 = np.searchsorted(cumulative_y, 0.25)
    idx2 = np.searchsorted(cumulative_y, 0.75)
    hdi_min = sorted_x[idx1]
    hdi_max = sorted_x[idx2]

    print(f"50% HPDI: {hdi_min} to {hdi_max}")