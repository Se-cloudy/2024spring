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
