import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding
from functools import reduce

from UAVMEC.top_layer.UAV_Env_V2 import UAV_env_V2
from UAVMEC.lower_layer.UAV_Env_Multi import UAV_env_Multi
from stable_baselines3 import PPO

UAV_num = 1
User_num = 4
Area_Bound = 30
E_max = 700
N = 50
T_max = 3.1
############################################
T = 3
E = 700
Service_num = int(User_num/UAV_num)

STATE_DIM = 2*(UAV_num+User_num) + User_num
ACTION_DIM = UAV_num
#ACTION_BOUND = np.array([np.pi/2, UAV_max_velocity, Lc_max], dtype=np.float32)
ACTION_BOUND = User_num-1

class TOP_ENV_DQN(gym.Env):
    """
      Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10):
        super(TOP_ENV_DQN, self).__init__()
        self.action_space = spaces.Discrete(User_num)
        #self.action_space = spaces.Box(low=0, high=1, shape=(UAV_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array(np.ones(STATE_DIM, dtype=np.float32)*-100000), high=np.array(np.ones(STATE_DIM, dtype=np.float32)*100000), dtype=np.float32)
        self.seed()
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.trajectory = np.zeros((User_num+10,2), dtype=np.float32)
        self.uav_tra = np.zeros((N,2), dtype=np.float32)
        self.Service_num = Service_num
        self.low_model = PPO.load("./low_best/low_best_model")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.Flag1 = 0
        self.Flag2 = 0

        self.current_step = 0

        # self.state[0] = np.random.uniform(0, Area_Bound)/Area_Bound
        # self.state[1] = np.random.uniform(0, Area_Bound)/Area_Bound
        self.state[0] = 0./Area_Bound
        self.state[1] = 0./Area_Bound


        for j in range(User_num):
            self.state[2*UAV_num + j*2] = np.random.uniform(0, Area_Bound)/Area_Bound
            self.state[2*UAV_num + j*2 + 1] = np.random.uniform(0, Area_Bound)/Area_Bound

        # self.state[2] = 40./Area_Bound*0.3
        # self.state[3] = 15./Area_Bound*0.3
        # self.state[4] = 65./Area_Bound*0.3
        # self.state[5] = 80./Area_Bound*0.3
        # self.state[6] = 25. / Area_Bound*0.3
        # self.state[7] = 30. / Area_Bound*0.3
        # self.state[8] = 45. / Area_Bound*0.3
        # self.state[9] = 16. / Area_Bound*0.3
        # self.state[10] = 37. / Area_Bound*0.3
        # self.state[11] = 30. / Area_Bound*0.3
        # self.state[12] = 92. / Area_Bound*0.3
        # self.state[13] = 50. / Area_Bound*0.3
        # self.state[14] = 25. / Area_Bound*0.3
        # self.state[15] = 5. / Area_Bound*0.3
        # self.state[16] = 90. / Area_Bound*0.3
        # self.state[17] = 76. / Area_Bound*0.3
        # self.state[18] = 74. / Area_Bound*0.3
        # self.state[19] = 56. / Area_Bound*0.3
        # self.state[20] = 77. / Area_Bound*0.3
        # self.state[21] = 99. / Area_Bound*0.3

        for i in range(User_num):
            self.state[2*(User_num+UAV_num) + i] = 0

        self.trajectory = np.array(self.trajectory)
        self.uav_tra = np.array(self.uav_tra)
        self.state = np.array(self.state)

        return self.state

    def prod(self, x, y):
        return x * y

    def step(self, action):
        self.uav_tra_No = 0
        self.trajectory[self.current_step][0] = self.state[0]*Area_Bound
        self.trajectory[self.current_step][1] = self.state[1]*Area_Bound

        self.current_step += 1
        state_next = np.zeros(STATE_DIM, dtype=np.float32)
        obs = np.zeros(7, dtype=np.float32)

        Total_Com_Energy = 0
        Total_Fly_Energy = 0
        Total_Distance = 0
        low_env = UAV_env_Multi()
        _ = low_env.reset()
        #更新无人机位置
        obs[0] = 0.
        obs[1] = 0.
        obs[2] = self.state[2 + int(action)*2] - self.state[0]
        obs[3] = self.state[2 + int(action)*2 + 1] - self.state[1]
        obs[4] = 1.
        obs[5] = E/E_max
        obs[6] = (T_max - T) / T_max
        obs = np.array(obs)
        for time in range(N+1):
            low_action, _states = self.low_model.predict(obs, deterministic=True)
            obs, rewards, dones, info = low_env.step(obs, low_action)
            Total_Com_Energy += low_env.E_com
            Total_Fly_Energy += low_env.E_fly
            Total_Distance += low_env.Distance
            if dones:
                distance = pow((pow(obs[2],2) + pow(obs[3],2)), 0.5)
                Total_reward = (1 - distance)  # 这个是R2部分的得分
                state_next[0] = obs[0] + self.state[0]
                state_next[1] = obs[1] + self.state[1]
                if info == False:
                    Total_reward = -1                   # 这个是R3部分的得分
                    Total_Com_Energy += 100
                # for t in range(low_env.Tra_No):
                #     self.uav_tra[t][0] = low_env.trajectory[t][0] + self.state[0]*Area_Bound
                #     self.uav_tra[t][1] = low_env.trajectory[t][1] + self.state[1]*Area_Bound
                #     self.uav_tra_No += 1
                break
        self.Total_Com_Energy = Total_Com_Energy
        self.Total_Fly_Energy = Total_Fly_Energy
        self.Total_Distance = Total_Distance

        #更新用户位置（保持不变）
        for j in range(User_num):
            state_next[2*UAV_num + j*2] = self.state[2*UAV_num + j*2]
            state_next[2*UAV_num + j*2 + 1] = self.state[2*UAV_num + j*2 + 1]

        #更新标志位
        for j in range(User_num):
            state_next[2 * (User_num + UAV_num) + j] = self.state[2 * (User_num + UAV_num) + j]
        for i in range(UAV_num):
            state_next[2*(User_num+UAV_num) + int(action)] = 1 + state_next[2*(User_num+UAV_num) + int(action)]
            if state_next[2*(User_num+UAV_num) + int(action)] > 1:
                self.Flag1 = 1

        #确定奖励函数
        reward = Total_reward

        if reduce(self.prod, state_next[2*(User_num+UAV_num):2*(User_num+UAV_num) + User_num]) == 1: #成功的
            done = True
            self.trajectory[self.current_step][0] = state_next[0] * Area_Bound
            self.trajectory[self.current_step][1] = state_next[1] * Area_Bound
        else:
            done = False
        if self.Flag1 == 1:
            done = True
            self.trajectory[self.current_step][0] = state_next[0] * Area_Bound
            self.trajectory[self.current_step][1] = state_next[1] * Area_Bound
            reward = -1  # 重复服务扣分
            self.Total_Com_Energy += 2000000
        # else:
        #     reward += 1/User_num      #这个是R1部分的得分

        self.state = state_next

        return state_next, reward, done, {}

    def render(self,mode='human'):
        fig1 = plt.figure(1)
        for j in range(User_num):
            plt.plot(self.state[2 + j * 2]*Area_Bound, self.state[2 + j * 2 + 1]*Area_Bound, 'om')
        plt.plot(self.trajectory[0:self.current_step+1, 0], self.trajectory[0:self.current_step+1, 1])
        plt.draw()
        axes = plt.gca()
        axes.set_xlim([-Area_Bound, Area_Bound])
        axes.set_ylim([-Area_Bound, Area_Bound])

        # plt.axis('equal')
        plt.pause(5)  # 间隔的秒数： 4s
        plt.close(fig1)

    def close(self):
        pass

    def give_value(self,State):
        for j in range(STATE_DIM):
            self.state[j] = State[j]


