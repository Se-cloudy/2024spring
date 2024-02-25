import numpy as np
import os

from UAVMEC.Method2_V2.UAV_Env_V2 import UAV_env_V2
from stable_baselines3 import DDPG, SAC, PPO, HER

best_mean_reward, n_steps = -np.inf, 0

log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

def bubble_sort(list):
    length = len(list)
    # 第一级遍历
    for index in range(length):
        # 第二级遍历
        for j in range(1, length - index):
            if list[j - 1][0] > list[j][0]:
                # 交换两者数据，这里没用temp是因为python 特性元组。
                list[j - 1][0], list[j][0] = list[j][0], list[j - 1][0]
                list[j - 1][1], list[j][1] = list[j][1], list[j - 1][1]
    return list

if __name__ == '__main__':
        env = UAV_env_V2()
        #model = PPO.load("PPO_M", env=env)
        model = PPO.load("./best/best_model", env=env)
        obs = np.zeros(6, dtype=np.float32)
        distance = np.zeros((env.User_num, 2), np.float32)
        UAV_initial_position = np.zeros(2, np.float32)
        flag = np.zeros(env.User_num, np.float32)
        step_num = 0
        for i in range(20):
            obs_raw = env.reset()
            obs[0] = obs_raw[0]
            obs[1] = obs_raw[1]
            for j in range(env.User_num):
                flag[j] = 0

            while sum(flag) < env.User_num:
                UAV_initial_position[0] = env.combine_tra[env.tra_num - 1][0]
                UAV_initial_position[1] = env.combine_tra[env.tra_num - 1][1]

                # 对远近进行排序
                for num in range(env.User_num):
                    distance[num][0] = np.sqrt(
                        np.power((obs_raw[2 + num * 2] - UAV_initial_position[0] / env.radius), 2) + np.power(
                            (obs_raw[2 + num * 2 + 1] - UAV_initial_position[1] / env.radius), 2))
                    distance[num][1] = num
                bubble_sort(distance)

                while flag[int(distance[step_num][1])] == 1:
                    step_num = step_num + 1
                    if step_num > env.User_num:
                        step_num = 0

                obs[2] = obs_raw[2 + int(distance[step_num][1]) * 2] - env.combine_tra[env.tra_num - 1][
                    0] / env.radius
                obs[3] = obs_raw[2 + int(distance[step_num][1]) * 2 + 1] - env.combine_tra[env.tra_num - 1][
                    1] / env.radius
                obs[4] = obs_raw[2 * (1 + env.User_num) + env.User_num - 1]
                obs[5] = obs_raw[2 * (1 + env.User_num) + env.User_num]
                obs[0] = 0
                obs[1] = 0
                obs = np.array(obs)
                env.reset_single(obs, UAV_initial_position)
                for time in range(50):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, dones, info = env.step(action)
                    if dones:
                        env.combine(int(distance[step_num][1]))
                        flag[int(distance[step_num][1])] = 1
                        step_num = 0
                        break

            env.render()




