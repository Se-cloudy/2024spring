<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt


def calGauss(mu, sigma, para, cur_x):
    cur_y = para * np.exp(-(cur_x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return cur_y


# 定义两个高斯信号
u1, sig1, p1 = 23, 1, 1
u2, sig2, p2 = 20, 1, 2

# x = np.linspace(min(u1 - 3 * sig1, u2 - 3 * sig2), max(u1 + 3 * sig1, u2 + 3 * sig2), 100)  # 有效域
x = np.linspace(17, 28, 100)  # 统一有效域
y1 = calGauss(u1, sig1, p1, x)
y2 = calGauss(u2, sig2, p2, x)

# 找到相互重叠的区域
overlap_region = np.minimum(y1, y2)

# 计算相互重叠区域的面积
# 也可以quad定积分或者np.trapz梯形数值积分
overlap_area = np.sum(overlap_region) * (x[1] - x[0])
print(f"相互重叠的面积：{overlap_area}")

# 计算 y1 的面积；功率1
area_y1 = np.sum(y1) * (x[1] - x[0])
print(f"y1 的面积：{area_y1}")

# 计算 y2 的面积
area_y2 = np.sum(y2) * (x[1] - x[0])
print(f"y2 的面积：{area_y2}")

# 可视化两个高斯信号及其相互重叠的区域
plt.plot(x, y1, label='Signal 1')
plt.plot(x, y2, label='Signal 2')
plt.fill_between(x, overlap_region, color='gray', alpha=0.5, label='Overlap Region')
plt.legend()
plt.grid(True)
plt.show()
=======
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
>>>>>>> c11fe143e18899b5eb4f502dd0969ebfa23e8979
