import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import psd_function as psd


# v2版本：将我方动作空间由原来的20×2降维成1×2->[真链路, 假链路]，解决了奖励函数直接向下走的问题，使得奖励函数均值稳定在0左右
# 这和不用强化学习的方法正确概率差不多。【√】
# v3版本：将我方的状态也描述为[真链路, 假链路, 敌方干扰链路]=========>【弃用！！！！！！！！！！！！！！！！！！】
# v4版本：将我方的状态描述为[真链路频点, 假链路频点, 敌方干扰链路频点, 真链带宽, 假链带宽, 敌方干扰带宽, 真链功率, 假链功率, 干扰功率]

# 计算信干噪比的函数
def sinr_reward(action, state, frequency_range):
    # 下一时刻动作
    real_next_fre, fake_next_fre = action[:2]
    real_next_band, fake_next_band = action[2:4]
    real_next_power, fake_next_power = action[4:]
    # 现有的状态
    real_fre, fake_fre, jam_fre = state[0][:3]
    real_band, fake_band, jam_band = state[0][3:6]
    real_power, fake_power, jam_power = state[0][6:]
    # 根据隐真链路计算信干噪比[还没用上噪声，用上了干扰信号]
    sinr_real_jam = psd.calculate_snr(real_fre, real_band, real_power, jam_fre, jam_band, jam_power, frequency_range)
    sinr_fake_jam = psd.calculate_snr(fake_fre, fake_band, fake_power, jam_fre, jam_band, jam_power, frequency_range)
    return np.array([sinr_real_jam, sinr_fake_jam])
    pass


# 自定义环境模拟我方信道选择策略，敌方策略还是对我方历史选择的最大干扰
class CommunicationEnv(gym.Env):

    def __init__(self):
        super(CommunicationEnv, self).__init__()

        self.channels = 20  # 可用信道数量
        self.enemy_accuracy = 0.6  # 敌方判断准确率，还是负值，越小敌方越广泛搜索，

        # 定义动作空间和观测空间：真、假、干扰 * 3[频点、带宽、功率]
        actino_space_high = np.array([[250, 250, 1.6, 1.6, 50, 50]])
        action_space_low = np.array([[150, 150, 0, 0, 0, 0]])
        observation_space_high = np.array([[250, 250, 260, 1.6, 1.6, 0.8, 50, 50, 60]])
        observation_space_low = np.array([[150, 150, 140, 0.2, 0.2, 0, 1, 1, 0]])
        # 动作空间,[真链路，假链路]，每个链路的频点、带宽、功率
        self.action_space = spaces.Box(low=action_space_low, high=actino_space_high, shape=(1, 6), dtype=np.float32)
        # 观测空间，[真链路，假链路，敌方干扰链路]，每个链路的频点、带宽、功率谱密度
        self.observation_space = spaces.Box(low=observation_space_low, high=observation_space_high, shape=(1, 9),
                                            dtype=np.float32)

        # 初始化状态
        self.state = None
        self.enemy_guess_record = None
        self.control_num = 100
        self.time_step = 0
        self.enemy = None
        self.frequency_range = np.linspace(140, 260, 12000)

    # 重置环境
    def reset(self):
        # 随机初始化我方和敌方的状态
        # self.state = np.zeros((1, 6))  # 初始化我方的状态
        # 随机初始化状态，随机采样频点、带宽、功率
        real_fre, fake_fre = round(np.random.uniform(190, 210), 2), round(np.random.uniform(190, 210), 2)
        real_band, fake_band = round(np.random.uniform(0.02, 0.06), 2), round(np.random.uniform(0.02, 0.06), 2)
        real_power, fake_power = 10, 5
        # self.state = np.array([[real_fre, fake_fre, real_band, fake_band, real_power, fake_power]])  # 随机选择作为我方的状态

        # 随机初始化敌方状态
        jam_fre_mid = round(np.random.uniform(-0.5, 0.5), 1)  # 随机设定频点增加的数值+-
        jam_band_mid = round(np.random.uniform(-0.02, 0.02), 1)  # 随机设定带宽增加的数值+-
        jam_power_mid = round(np.random.uniform(-2, 2), 1)  # 随机设定功率增加的数值+-
        self.enemy = np.array([real_fre + jam_fre_mid, real_band + jam_band_mid, real_power + jam_power_mid])
        # self.enemy_guess_record = np.zeros(self.channels)  # 初始化敌方对我方的记录数组
        self.time_step = 0
        self.state = np.array([[real_fre, fake_fre, self.enemy[0], real_band, fake_band, self.enemy[1], real_power,
                                fake_power, self.enemy[2]]])  # 随机选择作为我方的状态
        return self.state

    def step(self, action):
        """
        执行动作，返回新状态、奖励、完成标志和调试信息
        """
        done = False
        reward = 0
        states = self.state  # 获取当前的状态，可以认为是t时刻的状态

        # 限制动作范围[隐真频点，示假频点，隐真带宽，示假带宽，隐真功率，示假功率]
        action = np.clip(action, [150, 150, 0.2, 0.2, 1, 1], [250, 250, 1.6, 1.6, 50, 50])

        # 敌方干扰逻辑
        self.enemy = self.enemy_action()

        # 更新状态，将上述的action和状态拼接：[隐真频点，示假频点，干扰频点，隐真带宽，示假带宽，干扰带宽，隐真功率，示假功率，干扰功率]
        self.state = np.array([[action[0], action[1], self.enemy[0], action[2], action[3], self.enemy[1], action[4],
                                action[5], self.enemy[2]]])

        # 计算奖励
        # 计算隐真和示假信号各自的信干噪比
        sinr_real_jam, sinr_fake_jam = sinr_reward(action, self.state, self.frequency_range)
        # 我方隐真频点的信干噪比越大越好，我方示假频点的信干噪比越小越好(被噪声功率谱密度函数包围，减负值等于加正直)
        if sinr_real_jam >= 10:
            reward = 1
        else:
            reward = -1

        if sinr_fake_jam < 10:
            reward = reward + 1
        else:
            reward = reward - sinr_fake_jam / 50
        # reward = sinr_real_jam / 50 - sinr_fake_jam / 50

        # 更新时间，检查是否结束
        self.time_step += 1
        if self.time_step >= self.control_num:
            done = True

        return self.state, reward, done, {}

    def enemy_action(self):
        enemy_action = self.enemy
        if self.time_step % 2 == 0:
            enemy_action[0] += 5
            if enemy_action[0] >= 250:
                enemy_action[0] = 150
            return enemy_action
        return enemy_action
        pass

    # 待做，没写好，主程序中省略了【===========================================================】
    # ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇暂时没用！！！！！！！！！！！！！！！！！
    def render(self, mode='human'):
        """
        可视化当前环境状态
        """
        if mode == 'human':
            # 设置颜色
            real_channel_color = (0, 1, 0)  # 绿色，真实信道
            fake_channel_color = (0, 0, 1)  # 蓝色，假信道
            unused_channel_color = (0.7, 0.7, 0.7)  # 灰色，未使用的信道

            # 创建图形
            plt.figure(figsize=(10, 6))
            plt.title(f"Time Step: {self.time_step}")
            plt.xlabel("Channel")
            plt.ylabel("Status")

            # 绘制信道状态
            for i in range(self.channels):
                if self.state[i, 0] == 1:
                    # 真实信道
                    plt.bar(i, 1, color=real_channel_color)
                elif self.state[i, 1] == 1:
                    # 假信道
                    plt.bar(i, 1, color=fake_channel_color)
                else:
                    # 未使用的信道
                    plt.bar(i, 1, color=unused_channel_color)

            # 显示图形
            plt.show()
        else:
            super(CommunicationEnv, self).render(mode=mode)

    def close(self):
        """
        清理环境资源
        """
        pass
