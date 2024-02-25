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
from UAVMEC.lower_layer.UAV_Env_V2 import UAV_env_V2
from UAVMEC.lower_layer.UAV_Env_Multi import UAV_env_Multi
from UAVMEC.lower_layer.UAV_Env import UAV_env
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import DDPG, SAC, PPO, HER

best_mean_reward, n_steps = -np.inf, 0

If_Train = False


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
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

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
        env = UAV_env_Multi()
        env.seed(seed + rank)
        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        env = Monitor(env, log_file)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    if If_Train:
        num_cpu = 8
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        env_test = UAV_env_Multi()
        eval_callback = EvalCallback(env_test, best_model_save_path='./best/',
                                     log_path='./result/', eval_freq=5000,
                                     deterministic=True, render=False)
        policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[dict(pi=[128, 128], vf=[128, 128])])

        #model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, n_steps=1024, gamma=0.9, n_epochs=20, learning_rate=3e-4, verbose=1, tensorboard_log="./PPO_tensorboard/")
        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, gamma=0.99, tensorboard_log="./PPO_tensorboard/")

        #callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

        # Train the agent
        model.learn(total_timesteps=int(1e8), callback=eval_callback)
        model.save("PPO_M")
    else:
        env = UAV_env_Multi()
        #model = PPO.load("PPO_M", env=env)
        model = PPO.load("./best/best_model", env=env)
        Average_Energy = 0
        for i in range(50):
            obs = env.reset()
            ep_reward = 0
            ep_flying_energy = 0
            ep_com = 0
            for j in range(201):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                ep_reward += rewards

                if dones:
                    Average_Energy += env.E_com
                    print('| ep_reward: %.2f' % ep_reward,
                          '| ep_flying_energy: %.2f' % env.E_fly,
                          '| ep_Communication_energy: %.2f' % env.E_com,
                          '| max_flying_energy: %.2f' % env.energy_budget,
                          )
                    break
            #env.render()
        Average_Energy = Average_Energy/(i+1)
        print('| Average_Energy: %.2f' % Average_Energy)



