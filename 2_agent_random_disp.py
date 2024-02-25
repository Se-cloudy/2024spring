# 1.4 agent的第一轮训练 敌方是随机干扰 波形验证演示
# PPO_u_0103是用随机干扰训练出来的。

import gym
import numpy as np
import torch as th
from gym.spaces.box import Box
from matplotlib import pyplot as plt
from numpy import linspace
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os


def calGauss(mu, sigma, para, cur_x):
    cur_y = para * np.exp(-(cur_x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return cur_y


class MyEnv(gym.Env):
    """
    description
    注意绘图函数可以在时隙内或者时隙外。
    todo 问题：
    1、step写随机干扰，agent波形完全不动。。。。不知道的还以为这是jamming agent，，
        jamming是随机的，为什么只随机到25-30？
        agent不动，策略就是躲在最右边。。初始值就是最右边?model输出的策略action = [1,-1,1]，直接让mu_u = 最大值。模型训的问题。
        范围限制太小。恐怕要修改训练的范围。要让agent的可行范围增加。
    """

    def __init__(self):
        # 用户，干扰源与位置参数;
        # todo 不随时间变化的位置可能导致对功率的优化不明显
        self.ue_num = 1
        self.ue_location = [50, 0, 0]
        self.uav_location = [0, 0, 50]
        self.jam_location = [-20, 0, 0]

        # 信道参数
        self.h0 = 1  # 信道系数，与各方相对位置有关；目前不考虑移动
        self.B = 10  # todo 根据项目调整 B f0 L
        self.f0 = 200  # 项目 200MHz

        self.t_num = 60  # 时隙个数; 每一轮内循环中游戏失败(over) or 到达第N=60(s)个时隙，然后reset，进行外循环；
        self.delta = int(0.1 * self.B)

        self.mu_u0 = self.f0  # 初始值
        self.sigma_u0 = 1  # 10
        self.p_u0 = 1  # 与轨迹有关；
        self.p_um = 5  # 功率最大值约束

        # 干扰功率谱密度参数 完全已知；扫频；
        self.p_jm = 5

        self.mu_j0 = self.f0 - 2 * self.B
        self.sigma_j0 = 1  # 10
        self.p_j0 = 1

        # 训练参数 我方和敌方的状态
        self.state = np.zeros(6, dtype=np.float32)
        # 动作对齐：我方采取的行动 [mu_u, sig_u, p_u]
        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # 归一化到了[-1,1]
        self.ActionBound_u = [self.B, 0.5 * self.B - 1, self.p_um]
        self.ActionBound_j = [self.B, 0.5 * self.B - 1, self.p_jm]
        # full observation 归一化了
        self.observation_space = Box(low=np.zeros(6),
                                     high=np.array([1, 1, 1, 1, 1, 1]),
                                     dtype=np.float32, shape=(6,))

        self.ep_length = 100  # 最大交互次数 可以自行设置值
        self.current_step = 0
        self.reward = 0  # 外层
        self.cur_reward = 0  # 内层
        self.done = False

    def step(self, action):
        # action = action * self.ActionBound  # 没有归一化，不需要映射
        # action = [mu_u, sig_u, p_u]
        # 解归一化
        action[0] = 0.5 * (action[0] + 1) * self.ActionBound_u[0] + self.f0 - 0.5 * self.B  # mu = f0 +- 0.5B
        action[1] = 0.5 * (action[1] + 1) * self.ActionBound_u[1] + 1  # sig>=1
        action[2] = 0.5 * (action[2] + 1) * self.ActionBound_u[2]

        self.state[0] = action[0]
        self.state[1] = action[1]
        self.state[2] = action[2]
        self.state[3] = np.random.uniform(self.f0 - 0.5 * self.B, self.f0 + 0.5 * self.B)  # random.f/
        self.state[4] = np.random.uniform(1, 0.5 * self.B)
        self.state[5] = np.random.uniform(0, self.p_jm)

        # 双方功率
        [mu_u, sig_u, p_u, mu_j, sig_j, p_j] = self.state  # obs

        fh = int(max(mu_u + 3 * sig_u, mu_j + 3 * sig_j))
        fl = int(min(mu_u - 3 * sig_u, mu_j - 3 * sig_j))
        cur_num = int((fh - fl) / self.delta)  # 频域离散点数
        cur_x = np.linspace(fl, fh, cur_num)  # 统一有效域

        Ut_u = calGauss(mu_u, sig_u, p_u, cur_x)  # 我方功率谱密度函数
        Ut_j = calGauss(mu_j, sig_j, p_j, cur_x)  # 敌方功率谱密度函数

        Pt_u = self.h0 * np.sum(Ut_u) * (cur_x[1] - cur_x[0])  # 我方有效域内功率
        Pt_j = self.h0 * np.sum(Ut_j) * (cur_x[1] - cur_x[0])  # 敌方有效域内功率

        # 干扰功率：重叠面积
        Ut_overlap = np.minimum(Ut_u, Ut_j)
        Pt_overlap = np.sum(Ut_overlap) * self.delta

        # AWGN，强度调小
        Ut_n = 0.01 * self.p_u0 * np.ones(len(cur_x))
        Pt_n = np.sum(Ut_n) * (cur_x[1] - cur_x[0])

        # SINR 仅考虑通信链路
        sinr = Pt_u / (Pt_overlap + Pt_n)
        rate = (sig_u * 3) * (np.log2(1 + sinr)) # self.B * (np.log2(1 + sinr))  # todo B还是sig_u
        f1 = rate.sum() / self.ue_num  # 默认就是最大化 不用取负

        # 训练
        self.current_step += 1
        self.reward = f1
        # print("reward=-", self.reward)

        # 归一化便于训练
        self.state[0] = (self.state[0] - (self.f0 - 0.5 * self.B)) / self.B
        self.state[1] = (self.state[1] - 1) / (0.5 * self.B - 1)
        self.state[2] = self.state[2] / self.p_um
        self.state[3] = (self.state[3] - (self.f0 - 0.5 * self.B)) / self.B
        self.state[4] = (self.state[4] - 1) / (0.5 * self.B - 1)
        self.state[5] = self.state[5] / self.p_jm

        if self.current_step >= self.ep_length:
            self.done = True
        return self.state, self.reward, self.done, {}

    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0
        # 统统归一化 利于训练 改问题参数也只需要改bound了
        self.state[0] = (np.random.uniform(self.f0 - 0.5 * self.B, self.f0 + 0.5 * self.B) - (
                    self.f0 - 0.5 * self.B)) / self.B  # random.f/B
        self.state[1] = (np.random.uniform(1, 0.5 * self.B) - 1) / (0.5 * self.B - 1)  # -1有点混乱
        self.state[2] = np.random.uniform(0, self.p_um) / self.p_um
        self.state[3] = (np.random.uniform(self.f0 - 0.5 * self.B, self.f0 + 0.5 * self.B) - (self.f0 - 0.5 * self.B)) / self.B
        self.state[4] = (np.random.uniform(1, 0.5 * self.B) - 1) / (0.5 * self.B - 1)
        self.state[5] = np.random.uniform(0, self.p_jm) / self.p_jm
        return self.state

# 函数用于实时更新并显示高斯曲线及其重叠图像
    def update_plot(self, ax, obs, sinr_value):
        u1, sig1, p1, u2, sig2, p2 = obs
        # 清空当前轴
        ax.clear()

        # 计算高斯信号
        x = np.linspace(self.f0 - 3 * self.B, self.f0 + 3 * self.B, 100)
        y1 = calGauss(u1, sig1, p1, x)
        y2 = calGauss(u2, sig2, p2, x)

        # 找到相互重叠的区域
        overlap_region = np.minimum(y1, y2)

        # 可视化两个高斯信号及其相互重叠的区域
        ax.plot(x, y1, label='signal')
        ax.plot(x, y2, label='jamming')
        ax.fill_between(x, overlap_region, color='gray', alpha=0.5, label='Overlap Region')
        ax.legend()
        ax.grid(True)

        # 固定横纵轴范围
        # ax.set_xlim([self.f0 - 0.5 * self.B, self.f0 + 0.5 * self.B])
        # ax.set_ylim([0, max(max(y1), max(y2))])  # max(self.p_um, self.p_jm)])
        yh = max(self.p_um, self.p_jm)/min(self.p_um, self.p_jm)
        ax.set_xlim([self.f0 - 3 * self.B, self.f0 + 3 * self.B])
        ax.set_ylim([0, yh])  # max(max(y1), max(y2))

        sinr_value = round(sinr_value, 3)
        ax.text(0.5, 1.1, f'SINR = {sinr_value}', transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.pause(0.5)  # 添加短暂的延迟，使得图形有足够时间更新

    def func_obs(self, ls):
        # 将归一化的obs映射为真实数据
        ls[0] = 0.5 * (ls[0] + 1) * self.ActionBound_u[0] + self.f0 - 0.5 * self.B  # mu = f0 +- 0.5B
        ls[1] = 0.5 * (ls[1] + 1) * self.ActionBound_u[1] + 1  # sig>=1
        ls[2] = 0.5 * (ls[2] + 1) * self.ActionBound_u[2]
        ls[3] = 0.5 * (ls[3] + 1) * self.ActionBound_j[0] + self.f0 - 0.5 * self.B
        ls[4] = 0.5 * (ls[4] + 1) * self.ActionBound_j[1] + 1
        ls[5] = 0.5 * (ls[5] + 1) * self.ActionBound_j[2]


if __name__ == "__main__":
    env = MyEnv()
    t_num = env.t_num
    model = PPO.load("./PPO_u_0112", env=env)  # PPO_u_0103
    dispNum = 1000
    s1 = np.zeros(dispNum)  # 存储reward可视化

    # 创建初始图形
    fig, ax = plt.subplots()
    for i in range(dispNum):
        obs = env.reset()
        ep_reward = 0
        for j in range(t_num):
            action, state = model.predict(obs, deterministic=True)  # action = mu_u, sig_u, p_u（归一化之后） = [1,-1,1]
            obs, reward, done, info = env.step(action)  # obs=[mu_u, sig_u, p_u, mu_j, sig_j, p_j] =  [1,0,1]
            ep_reward += reward

            # env.func_obs(obs)
            # print(obs)
            # ep_reward = ep_reward/(j+1)
            # env.update_plot(ax, obs, ep_reward)  # 更新并显示高斯曲线及其重叠图像
        env.func_obs(obs)
        print(obs)
        ep_reward = ep_reward / t_num
        env.update_plot(ax, obs, ep_reward)  # 更新并显示高斯曲线及其重叠图像
        s1[i] = ep_reward

    x = range(1, dispNum + 1)
    plt.figure()
    plt.plot(x, s1)
    plt.title('reward')
    plt.show()

