import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
import time


from stable_baselines3.her import HER
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from UAVMEC.top_layer_backupV3.Top_Env_DQN import TOP_ENV_DQN
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import DDPG, SAC, PPO, DQN

position = np.zeros((4,2), dtype=np.float32)

best_mean_reward, n_steps = -np.inf, 0
best_mean_reward, n_steps = -np.inf, 0
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

env = TOP_ENV_DQN()
env = Monitor(env, log_dir)
th.manual_seed(0)

env.seed(0)
log_file = log_dir + '/data/'
os.makedirs(log_file, exist_ok=True)
env = Monitor(env, log_file)

env = TOP_ENV_DQN()
model = DQN.load("./Top_best/best_model", env=env)
Average_Com_Energy = 0
Average_Fly_Energy = 0

# position[1,1] = 1
# position[2,0] = 1
# position[2,1] = 1
# position[3,0] = 1

for i in range(1):
    obs = env.reset()
    ep_reward = 0
    Total_Com_Energy = 0
    Total_Fly_Energy = 0
    Total_Distance = 0
    for j in range(10):#此处先写死
        a = 0
        for k in range(4):#4个无人机
            a=0
            obs[0] = position[k, 0]
            obs[1] = position[k, 1]
            action, _states = model.predict(obs, deterministic=True)
            env.give_value(obs)
            obs, rewards, dones, info = env.step(action)
            position[k, 0] = obs[0]
            position[k, 1] = obs[1]
            ep_reward += rewards
            Total_Com_Energy += env.Total_Com_Energy
            Total_Fly_Energy += env.Total_Fly_Energy
            Total_Distance += env.Total_Distance
            if dones:
                #env.render()
                print('| Service_num: %.2f' % j,
                      '| ep_reward: %.2f' % ep_reward,
                      '| Total_Com_Energy: %.2f' % Total_Com_Energy,
                      '| Total_Fly_Energy: %.2f' % Total_Fly_Energy,
                      '| Total_Distance: %.2f' % Total_Distance,
                      )
                break