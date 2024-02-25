import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import torch as th
import optuna

from stable_baselines3.her import HER
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.test0330.newenv import UAV_env
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import DDPG, SAC, PPO, HER

best_mean_reward, n_steps = -np.inf, 0

log_dir = "/tmp/DDPGW/"
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
        env = UAV_env(ep_length=256)
        env.seed(seed + rank)
        env = Monitor(env, log_dir, allow_early_resets=True)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    num_cpu = 4
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # env = Monitor(env, log_dir, allow_early_resets=True)
    # env = DummyVecEnv([lambda: env])
    # # Automatically normalize the input features and reward
    # env = VecNormalize(env, norm_obs=True, norm_reward=True,
    #                    clip_obs=10.)

    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                         net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    # policy_kwargs = dict(activation_fn=th.nn.ReLU,
    #                      net_arch=dict(pi=[128, 128], qf=[128, 128]))
    # model = DDPG(MlpPolicy, env, learning_rate=1e-4, action_noise=action_noise, policy_kwargs=dict(net_arch=[64, 128, 64]), verbose=1, tensorboard_log="./DDPG_tensorboard/")
    # model = DDPG(MlpPolicy, env, buffer_size=100000, learning_rate=1e-3, policy_kwargs=policy_kwargs, verbose=1, action_noise=action_noise, tensorboard_log="./DDPG_tensorboard/")
    # model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, batch_size=64, n_steps=1024, n_epochs=5, gamma=0.9, learning_rate=5e-4, verbose=1, tensorboard_log="./PPO_tensorboard/")
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, n_steps=1024, gamma=0.9, n_epochs=20, learning_rate=3e-4, verbose=1, tensorboard_log="./PPO_tensorboard/")

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Train the agent
    model.learn(total_timesteps=int(1e6))
    # stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    # env.save(stats_path)

    model.save("PPO4")
    #
    del model, env
    model = PPO.load("PPO3")
    env = UAV_env(ep_length=256)
    # env = Monitor(env, log_dir, allow_early_resets=True)
    # env = DummyVecEnv([lambda: env])
    # # Automatically normalize the input features and reward
    # env = VecNormalize.load(stats_path, env)
    # env.training = False
    # env.norm_reward = False
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    #
    # print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    obs = env.reset()
    for i in range(10000):
        # env.render()
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # print(obs, action)
        if done:
            env.render()
            obs = env.reset()


