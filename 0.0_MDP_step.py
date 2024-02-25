# !/usr/bin/env python
# -*-coding:utf-8 -*-

import gym
import numpy as np
import math
from gym import spaces
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
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
        self.ue_num = 10
        self.L = 10
        self.jammer_xy = []
        self.ue_xy = [[]]

        # TODO 干扰分布...
        self.Uj = math.sin

        self.s0 = (1, 1)  # f,c

        # 训练参数
        # A # TODO BOX?
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.L),  # f取值范围为 0-L 的整数
            spaces.Discrete(2)  # C取值范围为 0 或 1
        ))

        # S
        # TODO Box(low=0, high=self.L, shape=(1,))
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.L),  # f取值范围为 0-L 的整数
            spaces.Discrete(2)  # C取值范围为 0 或 1
        ))
        self.state = self.s0  # 对齐action

        self.ep_length = 100  # 最大交互次数 可以自行设置值
        self.current_step = 0

        # 评估参数
        self.reward = 0
        self.done = False

    def discrete(self):
        return np.ones(self.L)

    def step(self, action):
        self.state = action[0]
        self.state = self.state + action
        # TODO 位置与距离
        ue_xyz = self.ue_xy
        jammer_xyz = self.jammer_xy
        d_mat = np.sqrt(
            np.square(ue_xyz[:, np.newaxis, :] - jammer_xyz[np.newaxis, :, :]).sum(axis=-1))  # mat都是N*1的 N个用户1个小区

        # 功率
        ft = self.discrete(self.Ut)  # TODO 离散化函数
        Pt = sum(ft)

        # 干扰
        para_i = 1
        fi = self.discrete(Ui)
        Pi = para_i * sum(fi)

        para_j = 1
        fj = self.discrete(Uj)
        Pj = para_j * sum(fj)

        fn = np.ones(1, self.L)  # TODO 高斯白噪声 功率谱密度
        Pn = sum(fn)

        # SINR
        sinr = Pt / (Pi + Pj + Pn)
        rate = self.B * (np.log2(1 + sinr))
        f = rate.sum() / self.ue_num

        # 训练
        self.reward = f
        self.current_step += 1
        if self.current_step >= self.ep_length:
            self.done = True

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.current_step = 0
        self.state = self.s0
        self.done = False
        return self.state

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        pass


if __name__ == "__main__":
    env = MyEnv()

    model = PPO(policy="MlpPolicy", env=env)
    model.learn(total_timesteps=int(1e2))

    # model.save("./singleCell1.2")

    obs = env.reset()

    num = int(1e2)
    s1 = np.zeros((num,))

    for ii in range(num):
        action, state = model.predict(observation=obs)
        obs, reward, done, info = env.step(action)
        # print(reward, obs)
        s1[ii] = reward
