# 1.2 jam的训练
# 我方动作基于训练好的模型输入；action, state = model.predict(observation=obs)

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

def calGauss(mu, sigma, para, cur_x):
    cur_y = para * np.exp(-(cur_x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return cur_y


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
        self.f0 = 25

        self.t_num = 60  # 时隙个数
        self.delta = int(0.1 * self.B)

        # 信号功率谱密度参数
        self.mu_u0 = self.f0  # 初始值
        self.sigma_u0 = 0.1 * self.B
        self.p_u0 = 1  # 与轨迹有关；
        self.p_um = 5  # 功率最大值约束

        # 干扰功率谱密度参数；
        self.mu_j0 = self.f0
        self.sigma_j0 = 1
        self.p_j0 = 1
        self.p_jm = 5

        # 训练参数
        self.state0 = np.array([self.mu_u0, self.sigma_u0, self.p_u0, self.mu_j0, self.sigma_j0, self.p_j0])
        self.state = self.state0
        # 敌方动作空间 [mu_j, sig_j, p_j]
        self.action_space = Box(low=np.array([self.f0 - 0.5 * self.B, 1, 1]),
                                high=np.array([self.f0 + 0.5 * self.B, 0.5 * self.B, self.p_jm]),
                                dtype=np.float32, shape=(3,))
        self.ActionBound = [self.B, 0.5 * self.B, self.p_jm]
        # full observation
        self.observation_space = Box(low=np.zeros(6),
                                     high=np.array([self.B, 0.5 * self.B, self.p_um, self.B, 0.5 * self.B, self.p_jm]),
                                     dtype=np.float32, shape=(6,))

        self.ep_length = 100  # 最大交互次数 可以自行设置值
        self.current_step = 0
        self.reward = 0  # 外层
        self.cur_reward = 0  # 内层
        self.done = False

        self.model_u = PPO.load("model_utest")

    def step(self, action):
        # 输入的是action = [mu_j.sig_j. p_j]，和动作对齐
        # model_u.predict(observations=obs)  # TODO
        action_uu, state_uu = self.model_u.predict(self.state, deterministic=True)
        self.state = np.hstack([action_uu, action])  # [state_u, state_j]

        [mu_u, sig_u, p_u, mu_j, sig_j, p_j] = self.state  # obs
        # 功率
        fh = int(max(mu_u + 3 * sig_u, mu_j + 3 * sig_j))
        fl = int(min(mu_u - 3 * sig_u, mu_j - 3 * sig_j))
        cur_num = 100  # int((fh - fl) / self.delta)  # 频域离散点数
        cur_x = np.linspace(fl, fh, cur_num)  # 统一有效域

        Ut_u = calGauss(mu_u, sig_u, p_u, cur_x)  # 我方功率谱密度函数
        Ut_j = calGauss(mu_j, sig_j, p_j, cur_x)  # 敌方功率谱密度函数

        Pt_u = self.h0 * np.sum(Ut_u) * (cur_x[1] - cur_x[0])  # 我方有效域内功率
        Pt_j = self.h0 * np.sum(Ut_j) * (cur_x[1] - cur_x[0])  # 我方有效域内功率

        # 干扰功率：重叠面积
        Ut_overlap = np.minimum(Ut_u, Ut_j)
        Pt_overlap = np.sum(Ut_overlap) * self.delta

        # AWGN
        Ut_n = 0.01 * self.p_u0 * np.ones(len(cur_x))
        Pt_n = np.sum(Ut_n) * (cur_x[1] - cur_x[0])

        # SINR 仅考虑通信链路
        sinr = Pt_u / (Pt_overlap + Pt_n)
        rate = self.B * (np.log2(1 + sinr))  # todo B还是sig_u
        f1 = rate.sum() / self.ue_num  # 最小化
        # 训练
        self.current_step += 1
        self.reward = f1
        print("reward=", self.reward)

        if self.current_step >= self.ep_length:
            self.done = True
        return self.state, self.reward, self.done, {}

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.state0

    # def render(self, mode="human"):
    #    pass

    # def seed(self, seed=None):
    #    pass


if __name__ == "__main__":
    train_mode = input("Train mode = 0/1: ")
    train_mode = int(train_mode)
    env = MyEnv()
    if train_mode:
        model_j = PPO(policy="MlpPolicy", env=env)
        model_j.learn(total_timesteps=int(1e2))  # 训练
        model_j.save("./model_jtest")
    else:
        model_u = PPO.load("model_utest", env=env)
        model_j = PPO.load("model_jtest", env=env)
        obs = env.reset()
        num = int(1e5)  # 演示
        s1 = np.zeros((num,))
        for ii in range(num):
            action_j, state_j = model_j.predict(observation=obs)
            # action_u, state_u = model_u.predict(observation=obs)
            # action = np.hstack([action_u, action_j])
            obs, reward, done, info = env.step(action_j)
            # print(reward, obs)
            s1[ii] = reward

        x = range(1, num + 1)
        plt.figure()
        plt.plot(x, s1)
        plt.title('reward')
        plt.show()
