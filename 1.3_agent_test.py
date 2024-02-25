# 1.3 agent的第一轮训练
# 干扰计算；归一映射
# 敌方干扰是随机干扰；删去了tt；

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
    1 增加了训练可视化
    2 改成了并行
    3 决策量归一化
    4 敌方干扰变成了随机干扰（1.2是固定扫频）
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

        self.t_num = 60  # 时隙个数; 每一轮内循环中游戏失败(over) or 到达第N=60(s)个时隙，然后reset，进行外循环；
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
        # self.state0 = np.array([self.mu_u0, self.sigma_u0, self.p_u0, self.mu_j0, self.sigma_j0, self.p_j0])
        # self.state = self.state0
        self.state = np.zeros(6, dtype=np.float32)
        # 动作对齐：我方采取的行动 [mu_u, sig_u, p_u]
        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # 归一化到了[-1,1]
        self.ActionBound = [self.B, 0.5 * self.B - 1, self.p_um]  # todo -1+1是为什么
        # full observation 归一化了
        self.observation_space = Box(low=np.zeros(6),
                                     high=np.array([1, 1, 1, 1, 1, 1]),
                                     dtype=np.float32, shape=(6,))

        self.ep_length = 100  # 最大交互次数 可以自行设置值
        self.current_step = 0
        # self.tt = 0  # 时隙计数器
        self.reward = 0  # 外层
        self.cur_reward = 0  # 内层
        self.done = False

    def step(self, action):
        # action = action * self.ActionBound  # 没有归一化，不需要映射
        # action = [mu_u, sig_u, p_u]

        action[0] = 0.5 * (action[0] + 1) * self.ActionBound[0] + self.f0 - 0.5 * self.B  # mu = f0 +- 0.5B
        action[1] = 0.5 * (action[1] + 1) * self.ActionBound[1] + 1  # sig>=1
        action[2] = 0.5 * (action[2] + 1) * self.ActionBound[2]

        self.state[0] = action[0]
        self.state[1] = action[1]
        self.state[2] = action[2]
        self.state[3] = np.random.uniform(self.f0 - 0.5 * self.B, self.f0 + 0.5 * self.B)  # random.f/
        self.state[4] = np.random.uniform(1, 0.5 * self.B)
        self.state[5] = np.random.uniform(0, self.p_jm)
        # self.state[3:] = [self.mu_j[self.tt], self.sigma_j, self.p_j[self.tt]]  # 小循环参数，由当前时隙决定

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

        self.state[3] = (np.random.uniform(self.f0 - 0.5 * self.B, self.f0 + 0.5 * self.B) - (
                    self.f0 - 0.5 * self.B)) / self.B
        self.state[4] = (np.random.uniform(1, 0.5 * self.B) - 1) / (0.5 * self.B - 1)
        self.state[5] = np.random.uniform(0, self.p_jm) / self.p_jm

        return self.state

    # def render(self, mode="human"):
    #    pass

    # def seed(self, seed=None):
    #    pass


log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    if len(y) > 3000:
                        self.model.save("PPO3")
        return True


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = MyEnv()
        env.seed(seed + rank)
        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        env = Monitor(env, log_file)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    train_mode = input("Train mode = 0/1: ")
    train_mode = int(train_mode)
    t_num = 60
    if train_mode:
        # 并行处理
        num_cpu = 8
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        env_test = MyEnv()
        env_test = Monitor(env_test)
        eval_callback = EvalCallback(env_test, best_model_save_path='./best/',
                                     log_path='./result/', eval_freq=5000,
                                     deterministic=True, render=False)
        policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[128, 128], vf=[128, 128])])

        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, gamma=0.995,
                    tensorboard_log="./PPO_tensorboard/",
                    n_steps=4096, learning_rate=0.00022941853017034746, clip_range=0.4, n_epochs=1)

        # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

        # Train the agent
        model.learn(total_timesteps=int(3e6), callback=eval_callback)  # 5e6
        model.save("PPO_u_0103")
    else:
        env = MyEnv()
        model = PPO.load("./PPO_u_0103", env=env)
        dispNum = 1000
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
