import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding
from functools import reduce

from UAVMEC.top_layer_backup.UAV_Env_V2 import UAV_env_V2
from stable_baselines3 import PPO

UAV_num = 1
User_num = 10
X_max = 50
Y_max = 50
Area_Bound = 100
Radius = 50
L_max = 200*1024.
E_max = 5000
N = 50
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
        self.Service_num = Service_num
        self.low_model = PPO.load("./low_best/low_best_model")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.Flag1 = 0
        self.Flag2 = 0

        self.current_step = 0

        self.state[0] = np.random.uniform(0, Area_Bound)/Area_Bound
        self.state[1] = np.random.uniform(0, Area_Bound)/Area_Bound
        # self.state[0] = 0./Area_Bound
        # self.state[1] = 0./Area_Bound


        # for j in range(User_num):
        #     self.state[2*UAV_num + j*2] = np.random.uniform(0, Area_Bound)/Area_Bound
        #     self.state[2*UAV_num + j*2 + 1] = np.random.uniform(0, Area_Bound)/Area_Bound

        self.state[2] = 40./Area_Bound
        self.state[3] = 15./Area_Bound
        self.state[4] = 65./Area_Bound
        self.state[5] = 80./Area_Bound
        self.state[6] = 25. / Area_Bound
        self.state[7] = 30. / Area_Bound
        self.state[8] = 45. / Area_Bound
        self.state[9] = 16. / Area_Bound
        self.state[10] = 37. / Area_Bound
        self.state[11] = 30. / Area_Bound
        self.state[12] = 92. / Area_Bound
        self.state[13] = 50. / Area_Bound
        self.state[14] = 25. / Area_Bound
        self.state[15] = 5. / Area_Bound
        self.state[16] = 90. / Area_Bound
        self.state[17] = 76. / Area_Bound
        self.state[18] = 74. / Area_Bound
        self.state[19] = 56. / Area_Bound
        self.state[20] = 77. / Area_Bound
        self.state[21] = 99. / Area_Bound

        for i in range(User_num):
            self.state[2*(User_num+UAV_num) + i] = 0

        #self.state[28] = 1

        self.state = np.array(self.state)

        return self.state

    def prod(self, x, y):
        return x * y

    def step(self, action):
        self.current_step += 1
        state_next = np.zeros(STATE_DIM, dtype=np.float32)
        obs = np.zeros(6, dtype=np.float32)

        #action = np.round(action * ACTION_BOUND)
        # action = 9
        # action[1] = 5

        k1=1
        Total_Com_Energy = 0
        Total_Uav_Energy = 0
        Total_Energy = 0
        low_env = UAV_env_V2()
        #更新无人机位置
        obs[0] = 0.
        obs[1] = 0.
        obs[2] = (self.state[2 + int(action)*2] - self.state[0])*Area_Bound/Radius
        obs[3] = (self.state[2 + int(action)*2 + 1] - self.state[1])*Area_Bound/Radius
        obs[4] = 1.
        obs[5] = 1.
        obs = np.array(obs)
        for time in range(N):
            low_action, _states = self.low_model.predict(obs, deterministic=True)
            obs, rewards, dones, info = low_env.step(obs, low_action)
            Total_Com_Energy += low_env.Com_Energy
            Total_Uav_Energy += low_env.Uav_Energy/8000
            Total_Energy = k1*Total_Com_Energy + (1-k1)*Total_Uav_Energy
            if dones:
                state_next[0] = obs[0]*Radius/Area_Bound + self.state[0]
                state_next[1] = obs[1]*Radius/Area_Bound + self.state[1]
                if info == False:
                    Total_Energy += 50
                break
        #low_env.close()

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
        reward = -Total_Energy

        if reduce(self.prod, state_next[2*(User_num+UAV_num):2*(User_num+UAV_num) + User_num]) == 1:
            done = True
        else:
            done = False
        if self.Flag1 == 1:
            done = True
            reward -= 50

        self.state = state_next
        return state_next, reward, done, {}

    def render(self,mode='human'):
        fig1 = plt.figure(1)
        for j in range(User_num):
            plt.plot(self.state[2 + j * 2]*Area_Bound, self.state[2 + j * 2 + 1]*Area_Bound, 'om')
        # plt.plot(self.combine_tra[0:self.tra_num, 0], self.combine_tra[0:self.tra_num, 1])
        # plt.plot(self.combine_tra[0, 0], self.combine_tra[0, 1], '.r')
        plt.draw()
        axes = plt.gca()
        axes.set_xlim([0, Area_Bound])
        axes.set_ylim([0, Area_Bound])
        #plt.axis('equal')

        # fig2 = plt.figure(2)
        # for j in range(User_num):
        #     plt.plot(self.Combine_Data_size[j, 0:self.tra_num, 0], self.Combine_Data_size[j, 0:self.tra_num, 1])
        # plt.draw()

        # plt.axis('equal')
        plt.pause(10)  # 间隔的秒数： 4s
        plt.close(fig1)
       # plt.close(fig2)
        # plt.close(fig3)

    def close(self):
        pass
