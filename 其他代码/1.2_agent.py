# 1.2 agent的第一轮训练
# 干扰计算；双层循环输出（时隙，迭代次数）；归一映射
# 敌方干扰是初始固定扫频，非智能；

# 一个地面用户，一个无人机基站，一个干扰源。不考虑同频干扰，用户移动。

import gym
import numpy as np
from gym.spaces.box import Box
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
        # 用户，干扰源与位置参数;
        # todo 不随时间变化的位置可能导致对功率的优化不明显
        self.ue_num = 1
        self.ue_location = [50, 0, 0]
        self.uav_location = [0, 0, 50]
        self.jam_location = [-20, 0, 0]

        # 信道参数
        self.h0 = 1  # todo 信道系数，与各方相对位置有关；目前不考虑移动
        self.B = 10  # todo 根据项目调整 B f0 L
        self.f0 = 25  # 项目 200MHz

        self.t_num = 60  # 时隙个数; 每一轮内循环中游戏失败(over) or 到达第N=60(s)个时隙，然后reset，进行外循环; reward=sum()即可; stb存在默认平均；
        self.delta = int(0.1 * self.B)

        self.mu_u0 = self.f0  # 初始值
        self.sigma_u0 = 0.1 * self.B
        self.p_u0 = 1  # 与轨迹有关；
        self.p_um = 5  # 功率最大值约束

        # 干扰功率谱密度参数 完全已知；扫频；
        self.mu_j = linspace(self.f0 - 0.5 * self.B, self.f0 + 0.5 * self.B, self.t_num)  # 要注意从1开始 注意划分数目以得到整数
        self.sigma_j = 0.1 * self.B  # 0.1 * 总带宽
        self.p_j = np.ones(self.t_num) * 2
        self.p_jm = 5

        self.mu_j0 = self.mu_j[0]
        self.sigma_j0 = self.sigma_j
        self.p_j0 = self.p_j[0]

        # 训练参数 我方和敌方的状态
        self.state0 = np.array([self.mu_u0, self.sigma_u0, self.p_u0, self.mu_j0, self.sigma_j0, self.p_j0])
        self.state = self.state0
        # 动作对齐：我方采取的行动 [mu_u, sig_u, p_u]
        self.action_space = Box(low=np.array([self.f0 - 0.5 * self.B, 1, 1]),
                                high=np.array([self.f0 + 0.5 * self.B, 0.5 * self.B, self.p_um]),
                                dtype=np.float32, shape=(3,))
        self.ActionBound = [self.B, 0.5 * self.B, self.p_um]
        # full observation
        self.observation_space = Box(low=np.zeros(6),
                                     high=np.array([self.B, 0.5 * self.B, self.p_um, self.B, 0.5 * self.B, self.p_jm]),
                                     dtype=np.float32, shape=(6,))

        self.ep_length = 100  # 最大交互次数 可以自行设置值
        self.current_step = 0
        self.tt = 0  # 时隙计数器
        self.reward = 0  # 外层
        self.cur_reward = 0  # 内层
        self.done = False

    def step(self, action):
        # action = action * self.ActionBound  # 没有归一化，不需要映射
        # action = [mu_u, sig_u, p_u]
        self.state[:3] = action
        self.state[3:] = [self.mu_j[self.tt], self.sigma_j, self.p_j[self.tt]]  # 小循环参数，由当前时隙决定

        # 双方功率
        [mu_u, sig_u, p_u, mu_j, sig_j, p_j] = self.state  # obs

        fh = int(max(mu_u + 3 * sig_u, mu_j + 3 * sig_j))
        fl = int(min(mu_u - 3 * sig_u, mu_j - 3 * sig_j))
        cur_num = 100  # int((fh - fl) / self.delta)  # 频域离散点数
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
        # print(Pt_u,Pt_overlap,Pt_n)
        sinr = Pt_u / (Pt_overlap + Pt_n)
        rate = self.B * (np.log2(1 + sinr))  # todo B还是sig_u
        f1 = - rate.sum() / self.ue_num  # 取负

        self.current_step += 1
        self.tt += 1  # 控制时隙循环；取出当前时隙；
        # 训练
        self.reward = f1  # todo 不处理reward与时隙的关系
        print("reward=-", self.reward)

        if self.tt == 59:
            self.tt = 0
        if self.current_step >= self.ep_length:
            self.done = True
        return self.state, self.reward, self.done, {}

    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.tt = 0
        return self.state0

    # def render(self, mode="human"):
    #    pass

    # def seed(self, seed=None):
    #    pass


if __name__ == "__main__":
    train_mode = input("Train mode = 0/1: ")
    train_mode = int(train_mode)
    t_num = 60
    if train_mode:
        env = MyEnv()
        # model = PPO(policy="MlpPolicy", env=env, learning_rate=0.002)
        model = PPO(policy="MlpPolicy", env=env)
        model.learn(total_timesteps=int(1e2))  # , callback= eval_callback)  # 训练
        model.save("./model_utest")
    else:
        env = MyEnv()
        model = PPO.load("./model_utest", env=env)
        dispNum = 100
        s1 = np.zeros(dispNum)  # 存储reward可视化
        for i in range(dispNum):
            obs = env.reset()
            ep_reward = 0
            for j in range(t_num):
                action, state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
            s1[i] = ep_reward

        x = range(1, dispNum + 1)
        plt.figure()
        plt.plot(x, s1)
        plt.title('reward')
        plt.show()
