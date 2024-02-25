# 1.3 第三轮训练 用智能敌方训练我方

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
    description:
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
        self.state = np.zeros(6, dtype=np.float32)
        # 敌方采取的行动 [mu_u, sig_u, p_u]
        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # 归一化到了[-1,1]
        self.ActionBound_j = [self.B, 0.5 * self.B - 1, self.p_jm]
        self.ActionBound_u = [self.B, 0.5 * self.B - 1, self.p_um]
        # full observation 归一化了
        self.observation_space = Box(low=np.zeros(6),
                                     high=np.array([1, 1, 1, 1, 1, 1]),
                                     dtype=np.float32, shape=(6,))

        self.ep_length = 100  # 最大交互次数 可以自行设置值
        self.current_step = 0
        self.reward = 0  # 外层
        self.cur_reward = 0  # 内层
        self.done = False

        self.model_j = PPO.load("PPO_j_0103.zip")

    def step(self, action):
        # action = [mu_u, sig_u, p_u]
        action[0] = 0.5 * (action[0] + 1) * self.ActionBound_u[0] + self.f0 - 0.5 * self.B  # mu = f0 +- 0.5B
        action[1] = 0.5 * (action[1] + 1) * self.ActionBound_u[1] + 1  # sig>=1
        action[2] = 0.5 * (action[2] + 1) * self.ActionBound_u[2]

        # 载入敌方智能体动作
        action_jj, state_jj = self.model_j.predict(self.state, deterministic=True)  # mu_j, sig_j, p_j
        action_jj[0] = 0.5 * (action_jj[0] + 1) * self.ActionBound_j[0] + self.f0 - 0.5 * self.B
        action_jj[1] = 0.5 * (action_jj[1] + 1) * self.ActionBound_j[1] + 1
        action_jj[2] = 0.5 * (action_jj[2] + 1) * self.ActionBound_j[2]

        self.state[0] = action[0]
        self.state[1] = action[1]
        self.state[2] = action[2]
        self.state[3] = action_jj[0]
        self.state[4] = action_jj[1]
        self.state[5] = action_jj[2]

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
        rate = self.B * (np.log2(1 + sinr))
        f1 = rate.sum() / self.ue_num
        # 训练
        self.current_step += 1
        self.reward = f1
        # print("reward=", self.reward)

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
        eval_callback = EvalCallback(env_test, best_model_save_path='./best_final/',
                                     log_path='./result_final/', eval_freq=5000,
                                     deterministic=True, render=False)
        policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[128, 128], vf=[128, 128])])

        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, gamma=0.995,
                    tensorboard_log="./PPO_tensorboard_final/",
                    n_steps=4096, learning_rate=0.00022941853017034746, clip_range=0.4, n_epochs=1)

        # Train the agent
        model.learn(total_timesteps=int(1e6), callback=eval_callback)  # 5e6
        model.save("PPO_0104")
    else:
        env = MyEnv()
        model = PPO.load("./PPO_0104", env=env)
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

        # model_u = PPO.load("model_utest", env=env)
        # model_j = PPO.load("model_jtest", env=env)
