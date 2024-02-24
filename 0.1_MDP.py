# !/usr/bin/env python
# -*-coding:utf-8 -*-

import gym
import numpy as np
import math
import random
from gym import spaces
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from matplotlib import pyplot as plt
from stable_baselines3 import PPO


class MyEnv(gym.Env):
    """
    description
    """

    def __init__(self):
        # TODO 默认信道参数...
        self.p0 = 1
        self.B = 1e6
        self.f0 = 5e5

        # TODO 默认用户参数...
        self.ue_num = 5
        self.timeslot_num = 10
        self.L = 10  # 离散

        self.jammer_xyz = [(0, 0, 100)] * 10  # [(),(),...]
        self.jammer_xyz = np.array(self.jammer_xyz)
        self.ue_xyz = [[None] * self.timeslot_num for _ in range(self.ue_num)]  # 创建一个5行10列的二维数组
        for i in range(self.ue_num):  # ue_num
            for j in range(self.timeslot_num):  # timeslot_num
                self.ue_xyz[i][j] = [random.randint(0, 100), random.randint(0, 100), 10]  # xyz
        self.d_mat = np.sqrt(np.square(self.ue_xyz - self.jammer_xyz).sum(axis=-1))

        # TODO 干扰分布...
        self.Uj = math.sin  # 应该改成一个序列 分布函数
        self.U0 = np.array([1, 1])  # m0,var0 示假 是参数 不是分布函数
        self.U1 = np.array([1, 1])  # m1,var1 隐真

        # 训练参数
        self.state0 = (self.U0[0], self.U0[1], self.U1[0], self.U1[1], self.Uj)  # 需要和action对齐

        # A = update{m0,var0,m1,var1} 离散化的频点和方差 因为是更新 所以完全对齐observation？
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.B),  # m0，均值
            spaces.Discrete(0.5 * self.B),  # var0，方差
            spaces.Discrete(self.B),  # m1，均值
            spaces.Discrete(0.5 * self.B),  # var1，方差
            self.Uj  # TODO
        ))
        # Box(low=0, high=self.L, shape=(1,))

        # TODO 观察到的环境信息 当前的state 指导action
        self.observation_space = Box(low=0, high=self.B, shape=(5,))
        self.observation_space[-1] = self.Uj

        self.ep_length = 100  # 最大交互次数 可以自行设置值
        self.current_step = 0

        # 评估参数
        self.reward = 0
        self.done = False

    def step(self, action):
        state = action  # 直接更新 不需要存储state到self。
        # TODO 两条链路都直接更新吗。

        # 真实功率
        Ut0 = np.random.normal(loc=action[0], scale=action[1], size=(1, self.B))
        Ut1 = np.random.normal(loc=action[2], scale=action[3], size=(1, self.B))

        fl = self.f0 - 0.5 * self.B
        fh = self.f0 + 0.5 * self.B
        delta = self.B / self.L

        para_t = 1  # TODO 地面指挥系统   ？
        Pt1 = para_t * sum(Ut1[fl:fh:delta])  # 间隔取离散值
        Pt0 = para_t * sum(Ut0[fl:fh:delta])

        # TODO 同频干扰
        para_i = 1
        delta_i = 0  # 中心频点偏置
        Pi = para_i * sum(Ut1[fl:fh:delta] - Ut1[fl + delta_i:fh + delta_i:delta])

        # 敌方干扰
        para_j = self.p0 / (self.d_mat ** 2)  # ue-j的距离
        Uj = action[4]
        Pj = para_j * sum(Ut1[fl:fh:delta] - Uj[fl:fh:delta])

        Un = np.ones(1, self.B)  # TODO 高斯白噪声 功率谱密度
        Pn = sum(Un[fl:fh:delta])

        # SINR
        sinr1 = Pt1 / (Pi + Pj + Pn)
        rate1 = self.B * (np.log2(1 + sinr1))
        f1 = rate1.sum() / self.ue_num

        sinr0 = Pt0 / (Pi + Pj + Pn)
        rate0 = self.B * (np.log2(1 + sinr0))
        f0 = rate0.sum() / self.ue_num

        # 训练
        self.reward = f1 - f0  # TODO 正负的reward处理?

        self.current_step += 1
        if self.current_step >= self.ep_length:
            self.done = True

        return state, self.reward, self.done, {}

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
    model.learn(total_timesteps=int(1e2))

    # model.save("./mdp_model")

    obs = env.reset()

    num = int(1e2)
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
