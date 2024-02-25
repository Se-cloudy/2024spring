import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# PSD方程计算式子
def gaussian_psd(frequency, center, bandwidth, power):
    """
    计算高斯功率谱密度。
    标准差由带宽决定：带宽是高斯分布标准差的2.355倍（FWHM）。
    """
    std = bandwidth / 2.355  # 功率谱密度函数的标准差，半宽最大幅频
    return power * np.exp(-0.5 * ((frequency - center) / std) ** 2)


def calculate_snr(signal_center, signal_bandwidth, signal_power,
                  noise_center, noise_bandwidth, noise_power,
                  fake_center, fake_bandwidth, fake_power,
                  frequency_range):
    """
    计算基于重叠范围的信干噪比（SNR）。

    参数:
    signal_center (float): 信号的中心频率
    signal_bandwidth (float): 信号的带宽
    signal_power (float): 信号的总功率
    noise_center (float): 噪声的中心频率
    noise_bandwidth (float): 噪声的带宽
    noise_power (float): 噪声的总功率
    frequency_range (np.array): 频率范围数组

    返回:
    float: 计算得到的信干噪比（dB）
    """
    # 计算信号和噪声的功率谱
    signal_spectrum = gaussian_psd(frequency_range, signal_center, signal_bandwidth, signal_power)
    noise_spectrum = gaussian_psd(frequency_range, noise_center, noise_bandwidth, noise_power)
    fake_spectrum = gaussian_psd(frequency_range, fake_center, fake_bandwidth, fake_power)
    signal_indices = np.where(signal_spectrum > 0.01)[0]  # 找出索引值
    noise_indices = np.where(noise_spectrum > 0.01)[0]

    mid_num = (frequency_range[-1] - frequency_range[0]) / len(frequency_range)
    # 计算重叠区域
    overlap_start = max((signal_indices[0] * mid_num) + frequency_range[0],
                        (noise_indices[0] * mid_num) + frequency_range[0])  # 重叠区域起点
    overlap_end = min((signal_indices[-1] * mid_num) + frequency_range[0],
                      (noise_indices[-1] * mid_num) + frequency_range[0])  # 重叠区域终点
    overlap_range = frequency_range[(frequency_range >= overlap_start) & (frequency_range <= overlap_end)]

    # 如果没有重叠区域，则SNR为一定值
    if len(overlap_range) == 0:
        # 返回
        return float('50')

    # 计算重叠区域的辅助函数，
    # overlap_ss = [(frequency_range >= overlap_start) & (frequency_range <= overlap_end)]
    # ss = signal_spectrum[(frequency_range >= overlap_start) & (frequency_range <= overlap_end)]
    # a = ss.sum()

    # 在重叠区域内计算信号和噪声的功率
    overlap_signal_power = integrate.simps(
        signal_spectrum[(frequency_range >= overlap_start) & (frequency_range <= overlap_end)], overlap_range)
    signal_power = integrate.simps(
        signal_spectrum[(frequency_range >= frequency_range[0]) & (frequency_range <= frequency_range[-1])],
        frequency_range)
    overlap_noise_power = integrate.simps(
        noise_spectrum[(frequency_range >= overlap_start) & (frequency_range <= overlap_end)], overlap_range)

    # 计算信干噪比
    # snr = 10 * np.log(overlap_signal_power / overlap_noise_power)
    snr = 10 * np.log(signal_power / overlap_noise_power)
    # snr = overlap_signal_power / overlap_noise_power

    """
    # 绘图函数【辅助用】
    plt.plot(frequency_range, signal_spectrum, label='Signal Power Spectrum', color='blue')
    plt.plot(frequency_range, noise_spectrum, label='Jamming Power Spectrum', color='orange')
    plt.plot(frequency_range, fake_spectrum, label='fake Power Spectrum', color='green')
    plt.title(f"SINR: {round(np.clip(snr, -50, 50), 3)}")
    plt.legend()
    plt.xlim(140, 260)
    plt.savefig("信道占用模拟_3.png")
    plt.show()
    """

    # 针ui需要特别指定输出->[信干噪比, 隐真信号在可用频段上的功率谱计算值, 干扰信号在可用频段上的功率谱计算值, 示假在可用频段上的功率谱计算值]
    out = np.array([np.clip(snr, -50, 50), signal_spectrum, noise_spectrum, fake_spectrum]).reshape(4, -1)

    return out

# 示例计算
# frequency_range = np.linspace(140, 260, 12000)
# snr_value = calculate_snr(signal_center=180, signal_bandwidth=20, signal_power=20,
#                           noise_center=180, noise_bandwidth=6, noise_power=20,
#                           fake_center=150, fake_bandwidth=4, fake_power=18,
#                           frequency_range=frequency_range)
#
# print("SNR (dB):", snr_value[0][0])
