import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding
from functools import reduce

from UAVMEC.top_layer.UAV_Env_V2 import UAV_env_V2
from stable_baselines3 import PPO

UAV_num = 2
User_num = 10
X_max = 50
Y_max = 50
Area_Bound = 100
Radius = 30
L_max = 200*1024.
E_max = 5000
N = 50
Service_num = int(User_num/UAV_num)

STATE_DIM = 2*(UAV_num+User_num) + User_num
ACTION_DIM = UAV_num
#ACTION_BOUND = np.array([np.pi/2, UAV_max_velocity, Lc_max], dtype=np.float32)
ACTION_BOUND = User_num-1

class TOP_ENV(gym.Env):
    """
      Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10):
        super(TOP_ENV, self).__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(UAV_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array(np.ones(STATE_DIM, dtype=np.float32)*-100000), high=np.array(np.ones(STATE_DIM, dtype=np.float32)*100000), dtype=np.float32)
        self.seed()
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.Service_num = Service_num
        self.low_model = PPO.load("./low_best/low_best_model")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.Flag1 = 0
        self.Flag2 = 0

        self.current_step = 0

        self.state[0] = 0./Area_Bound
        self.state[1] = 0./Area_Bound
        self.state[2] = 100./Area_Bound
        self.state[3] = 100./Area_Bound


        # for j in range(User_num):
        #     self.state[2*UAV_num + j*2] = np.random.uniform(0, Area_Bound)/Area_Bound
        #     self.state[2*UAV_num + j*2 + 1] = np.random.uniform(0, Area_Bound)/Area_Bound

        self.state[4] = 40./Area_Bound
        self.state[5] = 15./Area_Bound
        self.state[6] = 65./Area_Bound
        self.state[7] = 80./Area_Bound
        self.state[8] = 25. / Area_Bound
        self.state[9] = 30. / Area_Bound
        self.state[10] = 45. / Area_Bound
        self.state[11] = 16. / Area_Bound
        self.state[12] = 37. / Area_Bound
        self.state[13] = 30. / Area_Bound
        self.state[14] = 92. / Area_Bound
        self.state[15] = 50. / Area_Bound
        self.state[16] = 25. / Area_Bound
        self.state[17] = 5. / Area_Bound
        self.state[18] = 90. / Area_Bound
        self.state[19] = 76. / Area_Bound
        self.state[20] = 74. / Area_Bound
        self.state[21] = 56. / Area_Bound
        self.state[22] = 77. / Area_Bound
        self.state[23] = 99. / Area_Bound

        for i in range(User_num):
            self.state[2*(User_num+UAV_num) + i] = 0

        self.state = np.array(self.state)

        return self.state

    def prod(self, x, y):
        return x * y

    def step(self, action):
        self.current_step += 1
        state_next = np.zeros(STATE_DIM, dtype=np.float32)
        obs = np.zeros(6, dtype=np.float32)

        action = np.round(action * ACTION_BOUND)

        #检查是否重复服务
        for i in range(UAV_num):
            self.Flag1 = 1
            while(self.Flag1 == 1):
                if self.state[2*(User_num+UAV_num) + int(action[i])] == 1:
                    action[i] += 1
                    if(action[i] == User_num):
                        action[i] = 0
                else:
                    self.state[2 * (User_num + UAV_num) + int(action[i])] = 1
                    self.Flag1 = 0
                    break

        # action[0] = 2
        # action[1] = 5

        Total_Energy = np.zeros(UAV_num, dtype=np.float32)

        low_env = UAV_env_V2()
        #更新无人机位置
        for i in range(UAV_num):
            obs[0] = 0.
            obs[1] = 0.
            obs[2] = (self.state[2*UAV_num + int(action[i])*2] - self.state[i*2])*Area_Bound/Radius
            obs[3] = (self.state[2*UAV_num + int(action[i])*2 + 1] - self.state[i*2 + 1])*Area_Bound/Radius
            obs[4] = 1.
            obs[5] = 1.
            obs = np.array(obs)
            for time in range(N):
                low_action, _states = self.low_model.predict(obs, deterministic=True)
                obs, rewards, dones, info = low_env.step(obs, low_action)
                Total_Energy[i] += low_env.Energy
                if dones:
                    state_next[2*i] = obs[0]*Radius/Area_Bound + self.state[i*2]
                    state_next[2*i + 1] = obs[1]*Radius/Area_Bound + self.state[i*2 + 1]
                    if info == False:
                        Total_Energy[i] += 50
                    break
        #low_env.close()

        #更新用户位置（保持不变）
        for j in range(User_num):
            state_next[2*UAV_num + j*2] = self.state[2*UAV_num + j*2]
            state_next[2*UAV_num + j*2 + 1] = self.state[2*UAV_num + j*2 + 1]

        #更新标志位
        for j in range(User_num):
            state_next[2 * (User_num + UAV_num) + j] = self.state[2 * (User_num + UAV_num) + j]
        # for i in range(UAV_num):
        #     state_next[2*(User_num+UAV_num) + int(action[i])] = 1 + state_next[2*(User_num+UAV_num) + int(action[i])]

        #确定奖励函数
        reward = -sum(Total_Energy)

        if reduce(self.prod, state_next[2*(User_num+UAV_num):2*(User_num+UAV_num) + User_num]) == 1:
            done = True
        else:
            done = False
        # if self.Flag1 == 1:
        #     done = True
        #     reward -= (self.Service_num - self.current_step + 1)/self.Service_num

        self.state = state_next
        return state_next, reward, done, {}

    def render(self,mode='human'):
        fig1 = plt.figure(1)
        for j in range(User_num):
            plt.plot(self.state_raw[2 + j * 2]*Radius, self.state_raw[2 + j * 2 + 1]*Radius, 'om')
        plt.plot(self.combine_tra[0:self.tra_num, 0], self.combine_tra[0:self.tra_num, 1])
        plt.plot(self.combine_tra[0, 0], self.combine_tra[0, 1], '.r')
        plt.draw()
        axes = plt.gca()
        axes.set_xlim([-X_max, X_max])
        axes.set_ylim([-Y_max, Y_max])

        fig2 = plt.figure(2)
        for j in range(User_num):
            plt.plot(self.Combine_Data_size[j, 0:self.tra_num, 0], self.Combine_Data_size[j, 0:self.tra_num, 1])
        plt.draw()

        # plt.axis('equal')
        plt.pause(10)  # 间隔的秒数： 4s
        plt.close(fig1)
        plt.close(fig2)
        # plt.close(fig3)

    def close(self):
        pass
