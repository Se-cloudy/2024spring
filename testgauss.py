# 可以得到符合题设的高斯函数与积分！

import matplotlib.pyplot as plt
import numpy as np
import math


def calGauss(mu: float, sigma: float, para: float, cur_x: np.ndarray) -> np.ndarray:
    cur_y = para * np.exp(-(cur_x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    return cur_y


if __name__ == "__main__":
    f0 = 5.0  # 均值μ
    B = 10.0  # 标准差δ
    p = 1.0
    num = 50
    x = np.linspace(f0 - 0.5 * B, f0 + 0.5 * B, num)  # 有效域
    u = 5
    sig = 2
    y = calGauss(u, sig, p, x)

    temp = np.sum(y[0:49]) * (x[1] - x[0])  # 粒度需要统一, (x[1] - x[0]) 就是微元下标的delta长度
    print(temp)

    if True:
        plt.plot(x, y, "g", linewidth=2)  # 加载曲线
        plt.grid(True)  # 网格线
        plt.show()  # 显示
