# 1.1 agent的第一轮训练
# 敌方干扰是初始固定扫频，非智能；

# 一个地面用户，一个无人机基站，一个干扰源。不考虑同频干扰，用户移动。

import gym
import numpy as np
import math
import random
from gym import spaces
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from matplotlib import pyplot as plt
from numpy import linspace
from stable_baselines3 import PPO


class MyEnv(gym.Env):
    """
    description
    """

    def __init__(self):
        # 用户，干扰源与位置参数; 目前不随时间变化
        self.ue_num = 1
        self.ue_location = [50, 0, 0]
        self.uav_location = [0, 0, 50]
        self.jam_location = [-20, 0, 0]

        # 信道参数
        self.h0 = 1  # 信道系数，与各方相对位置有关；目前不考虑移动
        self.B = 10
        self.f0 = 5

        # 信号功率谱密度参数
        self.t_num = 10  # 时隙个数; 每一轮 over or 到达第n=60个时隙，reset; reward=sum(); stb默认平均；
        self.L = 10  # 频段离散个数
        self.mu_u0 = self.f0  # 初始值
        self.sigma_u0 = 1
        self.p_u0 = 1  # 与轨迹有关；
        self.p_max = 5  # 好像不需要这个了

        # 干扰功率谱密度参数 完全已知；扫频；
        self.mu_j = linspace(1, self.f0 + 0.5 * self.B, self.t_num)  # 要注意从1开始 注意划分数目以得到整数
        self.sigma_j = 1  # 1/tnum 总带宽
        self.p_j = np.ones(self.t_num) * 2

        # 训练参数
        self.state0 = (self.mu_u0, self.sigma_u0)  # , self.p_u0)

        # self.action_space = spaces.Tuple((
        #     spaces.Discrete(self.B),
        #     spaces.Discrete(0.5 * self.B)
        #     spaces.Discrete(self.p_max)
        # ))
        self.action_space = Box(low=np.array([0, 0]), high=np.array([self.B, 0.5 * self.B]), dtype=np.float32, shape=(2,))
        # full observation
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([self.B, 0.5 * self.B]), dtype=np.float32, shape=(2,))

        self.ep_length = 100  # 最大交互次数 可以自行设置值
        self.current_step = 0
        self.reward = 0
        self.done = False

    def step(self, action):
        new_state = action  # todo stb3归一化到[-1,1]，需要映射

        # 功率
        fl = self.f0 - 0.5 * self.B
        fh = self.f0 + 0.5 * self.B
        delta = self.B / self.L

        Ut_u = np.random.normal(loc=new_state[0], scale=new_state[1], size=(1, self.B))
        Pt_u = self.h0 * sum(Ut_u[fl:fh:delta])  # 间隔取离散值

        # 敌方干扰
        Ut_j = np.random.normal(loc=self.mu_j[0], scale=self.sigma_j[0], size=(1, self.B))

        # todo 范围要改
        Pt_j = self.h0 * sum(Ut_u[fl:fh:delta] - Ut_j[fl:fh:delta])

        Un = np.ones(1, self.B)  # 高斯白噪声 功率谱密度
        Pt_n = sum(Un[fl:fh:delta])

        # SINR 仅考虑真实通信链路
        sinr1 = Pt_u / (Pt_j + Pt_n)
        rate1 = self.B * (np.log2(1 + sinr1))
        f1 = rate1.sum() / self.ue_num

        # 训练
        self.reward = f1

        self.current_step += 1
        if self.current_step >= self.ep_length:
            self.done = True

        return new_state, self.reward, self.done, {}

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.state0

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        pass


if __name__ == "__main__":
    env = MyEnv()

    model = PPO(policy="MlpPolicy", env=env)
    model.learn(total_timesteps=int(1e2))  # 训练

    # model.save("./model_u")
    obs = env.reset()
    num = int(1e2)  # 演示
    s1 = np.zeros((num,))
    for ii in range(num):
        action, state = model.predict(observation=obs)
        obs, reward, done, info = env.step(action)
        # print(reward, obs)
        s1[ii] = reward

    x = range(1, num + 1)
    plt.figure()
    plt.plot(x, s1)
    plt.title('reward')
    plt.show()
