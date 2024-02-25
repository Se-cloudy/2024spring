import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding
from pandas import DataFrame
from pandas import read_csv
from functools import reduce

UAV_num = 4
User_num = 40
Area_Bound = 30
L_max = 200*1024.
E_max = 5000
N = 50
Service_num = int(User_num/UAV_num)

STATE_DIM = 2*(UAV_num+User_num) + User_num
ACTION_DIM = UAV_num
#ACTION_BOUND = np.array([np.pi/2, UAV_max_velocity, Lc_max], dtype=np.float32)
ACTION_BOUND = User_num-1

class Reset_Env(gym.Env):
    """
      Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10):
        super(Reset_Env, self).__init__()
        self.action_space = spaces.Discrete(User_num)
        #self.action_space = spaces.Box(low=0, high=1, shape=(UAV_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array(np.ones(STATE_DIM, dtype=np.float32)*-100000), high=np.array(np.ones(STATE_DIM, dtype=np.float32)*100000), dtype=np.float32)
        self.seed()
        self.state = np.zeros(STATE_DIM, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        position = np.zeros((User_num, 2), dtype=np.float32)
        # self.state[0] = np.random.uniform(0, Area_Bound)/Area_Bound
        # self.state[1] = np.random.uniform(0, Area_Bound)/Area_Bound
        self.state[0] = 0./Area_Bound
        self.state[1] = 0./Area_Bound
        self.state[2] = 0./Area_Bound
        self.state[3] = 0./Area_Bound
        self.state[4] = 0./Area_Bound
        self.state[5] = 0./Area_Bound
        self.state[6] = 0./Area_Bound
        self.state[7] = 0./Area_Bound

        df_r = read_csv('User_Position_final.csv')
        for j in range(User_num):
            self.state[2*UAV_num + j*2] = df_r.iloc[j,0]
            self.state[2*UAV_num + j*2 + 1] = df_r.iloc[j,1]

        # for j in range(User_num):
        #     self.state[2*UAV_num + j*2] = np.random.uniform(0, 30)/Area_Bound
        #     self.state[2*UAV_num + j*2 + 1] = np.random.uniform(0, 30)/Area_Bound
        #     position[j,0] = self.state[2*UAV_num + j*2]
        #     position[j,1] = self.state[2*UAV_num + j*2 + 1]
        # df_w = DataFrame({'x':position[0:User_num,0], 'y':position[0:User_num,1]})
        # df_w.to_csv('User_Position.csv',index=False)

        #第1种固定分布
        # self.state[2*UAV_num] = 40./Area_Bound*0.3
        # self.state[2*UAV_num+1] = 15./Area_Bound*0.3
        # self.state[2*UAV_num+2] = 65./Area_Bound*0.3
        # self.state[2*UAV_num+3] = 80./Area_Bound*0.3
        # self.state[2*UAV_num+4] = 25. / Area_Bound*0.3
        # self.state[2*UAV_num+5] = 30. / Area_Bound*0.3
        # self.state[2*UAV_num+6] = 45. / Area_Bound*0.3
        # self.state[2*UAV_num+7] = 16. / Area_Bound*0.3
        # self.state[2*UAV_num+8] = 37. / Area_Bound*0.3
        # self.state[2*UAV_num+9] = 30. / Area_Bound*0.3
        # self.state[2*UAV_num+10] = 92. / Area_Bound*0.3
        # self.state[2*UAV_num+11] = 50. / Area_Bound*0.3
        # self.state[2*UAV_num+12] = 25. / Area_Bound*0.3
        # self.state[2*UAV_num+13] = 5. / Area_Bound*0.3
        # self.state[2*UAV_num+14] = 90. / Area_Bound*0.3
        # self.state[2*UAV_num+15] = 76. / Area_Bound*0.3
        # self.state[2*UAV_num+16] = 74. / Area_Bound*0.3
        # self.state[2*UAV_num+17] = 56. / Area_Bound*0.3
        # self.state[2*UAV_num+18] = 77. / Area_Bound*0.3
        # self.state[2*UAV_num+19] = 99. / Area_Bound*0.3

        #第2种固定分布
        # self.state[2*UAV_num] = 56./Area_Bound*0.3
        # self.state[2*UAV_num+1] = 53./Area_Bound*0.3
        # self.state[2*UAV_num+2] = 74./Area_Bound*0.3
        # self.state[2*UAV_num+3] = 45./Area_Bound*0.3
        # self.state[2*UAV_num+4] = 82. / Area_Bound*0.3
        # self.state[2*UAV_num+5] = 55. / Area_Bound*0.3
        # self.state[2*UAV_num+6] = 63. / Area_Bound*0.3
        # self.state[2*UAV_num+7] = 4. / Area_Bound*0.3
        # self.state[2*UAV_num+8] = 29. / Area_Bound*0.3
        # self.state[2*UAV_num+9] = 74. / Area_Bound*0.3
        # self.state[2*UAV_num+10] = 37. / Area_Bound*0.3
        # self.state[2*UAV_num+11] = 27. / Area_Bound*0.3
        # self.state[2*UAV_num+12] = 83. / Area_Bound*0.3
        # self.state[2*UAV_num+13] = 69. / Area_Bound*0.3
        # self.state[2*UAV_num+14] = 2. / Area_Bound*0.3
        # self.state[2*UAV_num+15] = 20. / Area_Bound*0.3
        # self.state[2*UAV_num+16] = 49. / Area_Bound*0.3
        # self.state[2*UAV_num+17] = 39. / Area_Bound*0.3
        # self.state[2*UAV_num+18] = 57. / Area_Bound*0.3
        # self.state[2*UAV_num+19] = 69. / Area_Bound*0.3

        for i in range(User_num):
            self.state[2*(User_num+UAV_num) + i] = 0

        self.state = np.array(self.state)

        return self.state
