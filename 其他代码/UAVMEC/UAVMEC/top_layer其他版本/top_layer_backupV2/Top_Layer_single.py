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
from UAVMEC.top_layer.Top_Env_DQN import TOP_ENV_DQN
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import DDPG, SAC, PPO, HER

best_mean_reward, n_steps = -np.inf, 0

If_Train = True

best_mean_reward, n_steps = -np.inf, 0
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

env = TOP_ENV_DQN()
env = Monitor(env, log_dir)
th.manual_seed(0)



if __name__ == '__main__':
    if If_Train:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
        eval_callback = EvalCallback(env, best_model_save_path='./Top_best/',
                                     log_path='./Top_result/', eval_freq=5000,
                                     deterministic=True, render=False)
        policy_kwargs = dict(activation_fn=th.nn.Sigmoid, net_arch=[dict(pi=[64, 64], vf=[64, 64])])
        #policy_kwargs = dict(activation_fn=th.nn.Sigmoid, net_arch=dict(pi=[64, 64], qf=[64, 64]))
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=0.999)
        model.learn(total_timesteps=int(1e6), callback=eval_callback)
        model.save("Top_PPO_M")
    else:
        env = TOP_ENV_DQN()
        model = PPO.load("./Top_best/best_model", env=env)
        for i in range(1000):
            obs = env.reset()
            ep_reward = 0
            ep_flying_energy = 0
            ep_com = 0
            for j in range(env.Service_num):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                ep_reward += rewards
                if dones:
                    print('| ep_reward: %.2f' % ep_reward,
                          )
                    break
            env.render()



