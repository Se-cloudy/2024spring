import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding

N = 50
P_tr = 0.1
sigma = 1e-3
X_max = 20.
Y_max = 20.
L_max = 100*1024.#kB
UAV_num = 1
User_num = 1
UAV_max_velocity = 40.
H = 20
N0 = 3.9811E-21  #-174dbm
B = 10000000.  #10MHz
k = 0.2171  #质量系数
rc = 10E-28 #effective switched capacitance of UAV processor
g0 = 1e-5
O_k = 0.5
C_K = 1550.7
E_max = 500000
d_safe = 1.
I = [2.,4.,6.]#kB
T = 1.
delta = T/N

STATE_DIM = 2*(UAV_num+User_num)+User_num
ACTION_DIM = 2
ACTION_BOUND = np.array([np.pi/2, UAV_max_velocity], dtype=np.float32)

class UAV_env(gym.Env):
    """
      Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10):
        super(UAV_env, self).__init__()

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0,0,0,0,0]), high=np.array([X_max,Y_max,X_max,Y_max,L_max]), dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.trajectory = np.zeros((N, 2), np.float32)
        for i in range(UAV_num):
            #self.state[i * 2] = np.random.uniform(0., X_max)
            #self.state[i * 2 + 1] = np.random.uniform(0., Y_max)
            self.state[i * 2] = 0.
            self.state[i * 2 + 1] = 0.

        for j in range(User_num):
            #self.state[UAV_num * 2 + j * 2] = np.random.uniform(0., X_max)
            #self.state[UAV_num * 2 + j * 2 + 1] = np.random.uniform(0., Y_max)
            self.state[UAV_num * 2 + j * 2] = 20.
            self.state[UAV_num * 2 + j * 2 + 1] = 20.

        for j in range(User_num):
            self.state[(UAV_num + User_num) * 2 + j] = 40 * 1024.

        self.state = np.array(self.state)

        self.trajectory[self.current_step][0] = self.state[0]
        self.trajectory[self.current_step][1] = self.state[1]

        return self.state

    def calculate_tr_s(self, state):
        rs = np.zeros((UAV_num, User_num), np.float32)
        for i in range(UAV_num):
            for j in range(User_num):
                G = g0 / (np.power((state[i * 2] - state[UAV_num * 2 + j * 2]), 2) + np.power(
                    (state[i * 2 + 1] - state[UAV_num * 2 + j * 2 + 1]), 2) + np.power(H, 2))
                rs[i][j] = B * np.log2(1 + P_tr * G / sigma ** 2)

        return rs

    def calculate_energy(self, state):
        E_com = 0
        for i in range(UAV_num):
            for j in range(User_num):
                if state[(UAV_num + User_num) * 2 + j] <= 0:
                    E_com += 0
                else:
                    E_com += P_tr * delta
        return E_com

    def step(self, action):
        self.current_step += 1

        action = action * ACTION_BOUND
        state_next = np.zeros(STATE_DIM, dtype=np.float32)
        state_next[0] = self.state[0] + action[1] * delta * np.cos(action[0])
        state_next[1] = self.state[1] + action[1] * delta * np.sin(action[0])

        if self.current_step < N:
            self.trajectory[self.current_step][0] = state_next[0]
            self.trajectory[self.current_step][1] = state_next[1]

        for j in range(User_num):
            state_next[2 + j * 2] = self.state[2 + j * 2]
            state_next[2 + j * 2 + 1] = self.state[2 + j * 2 + 1]

        rs = self.calculate_tr_s(self.state)

        for i in range(UAV_num):
            for j in range(User_num):
                state_next[(UAV_num + User_num) * 2 + j] = self.state[(UAV_num + User_num) * 2 + j] - delta * rs[i][j]
                if state_next[(UAV_num + User_num) * 2 + j] <= 0:
                    state_next[(UAV_num + User_num) * 2 + j] = 0.

        state_next = np.array(state_next)
        rs_next = self.calculate_tr_s(state_next)
        rs_max = B * np.log2(1 + P_tr * g0 / np.power(H, 2) / sigma ** 2)
        gradient = (sum(sum(rs_next)) - sum(sum(rs)))/rs_max*30

        reward = -state_next[(UAV_num + User_num) * 2 + j]/sum(sum(rs_next))*P_tr + gradient
        if sum(state_next[(UAV_num + User_num) * 2:(UAV_num + User_num) * 2 + User_num]) == 0:
            done = True
            # reward += 1
        else:
            done = False

        if self.current_step >= N:
            done = True

        self.state = state_next
        return state_next, reward, done, {}

    def render(self,mode='human'):
        fig1 = plt.figure(1)
        for j in range(User_num):
            plt.plot(self.state[2 + j * 2], self.state[2 + j * 2 + 1], 'om')
        plt.plot(self.trajectory[0:self.current_step, 0], self.trajectory[0:self.current_step, 1])
        plt.plot(self.trajectory[0, 0], self.trajectory[0, 1], '.r')
        plt.draw()

        axes = plt.gca()
        axes.set_xlim([0, 30])
        axes.set_ylim([0, 30])
        # plt.axis('equal')
        plt.pause(2)  # 间隔的秒数： 4s
        plt.close(fig1)

    def close(self):
        pass
