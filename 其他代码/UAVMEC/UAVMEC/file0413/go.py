import os
import numpy as np
import torch


from stable_baselines3.common.env_checker import check_env
from UAVMEC.file0413.UAV_Env_new import UAV_env_new
from UAVMEC.file0413.UAV_Env_Multi import UAV_env_multi
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback

If_Load = True

best_mean_reward, n_steps = -np.inf, 0
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

env = UAV_env_new()
env = Monitor(env, log_dir)
torch.manual_seed(0)

if If_Load:
    model = PPO.load("PPO_M", env=env)
    #model = PPO.load("./best/best_model", env=env)
    for i in range(1000):
        obs = env.reset()
        for j in range(50):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if dones:
                break
        env.render()
else:
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
    eval_callback = EvalCallback(env, best_model_save_path='./best/',
                                 log_path='./result/', eval_freq=5000,
                                 deterministic=True, render=False)
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1,
                 tensorboard_log="./PPO_tensorboard/")
    model.learn(total_timesteps=int(1e6), callback=eval_callback)
    model.save("PPO_M")