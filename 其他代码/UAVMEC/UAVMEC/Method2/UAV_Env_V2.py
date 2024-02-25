import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding

N = 50
P_tr = 0.1
sigma2 = 1e-14#-110dbm
X_max = 30.
Y_max = 30.
Radius = 30.
L_max = 200*1024.#kB
Lc_max = 20*1024 #kB
UAV_num = 1
User_num = 2
UAV_max_velocity = 30.
H = 20
N0 = 3.9811E-21  #-174dbm
B = 1000000.  #1MHz
k = 0.2171  #质量系数
rc = 10E-28 #effective switched capacitance of UAV processor
g0 = 1e-5
O_k = 0.5
C_K = 1550.7
E_max = 5000
d_safe = 1.
I = [2.,4.,6.]#kB
T = 1.
delta = T/N
Mass = 10

STATE_DIM = 2*(UAV_num+User_num) + User_num + 1
ACTION_DIM = 3
ACTION_BOUND = np.array([np.pi/2, UAV_max_velocity, Lc_max], dtype=np.float32)

class UAV_env_V2(gym.Env):
    """
      Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10):
        super(UAV_env_V2, self).__init__()

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0,0,0,0,0,0]), high=np.array([X_max,Y_max,X_max,Y_max,L_max,E_max]), dtype=np.float32)
        self.seed()
        self.current_step = 0
        self.rs_max = 0
        self.FlyingEnergy_max = 0
        self.state = np.zeros(6, dtype=np.float32)
        self.state_raw = np.zeros(STATE_DIM, dtype=np.float32)
        self.trajectory = np.zeros((N, 2), np.float32)
        self.combine_tra = np.zeros((N*User_num, 2), np.float32)
        self.Data_size = np.zeros((N, 2), np.float32)
        self.User_num = User_num
        self.STATE_DIM = STATE_DIM
        self.UAV_initial_position = np.zeros(2, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0

        for i in range(UAV_num):
            self.state_raw[i * 2] = 0./X_max
            self.state_raw[i * 2 + 1] = 0./Y_max

        for j in range(User_num):
            theta = np.random.uniform(0, 2*np.pi)
            Distance = np.random.uniform(0, Radius)
            self.state_raw[UAV_num * 2 + j * 2] = np.cos(theta)*Distance/Radius
            self.state_raw[UAV_num * 2 + j * 2 + 1] = np.sin(theta)*Distance/Radius

        # self.state_raw[UAV_num * 2 + 0 * 2] = 10./Radius
        # self.state_raw[UAV_num * 2 + 0 * 2 + 1] = 10./Radius
        # self.state_raw[UAV_num * 2 + 1 * 2] = 20./Radius
        # self.state_raw[UAV_num * 2 + 1 * 2 + 1] = -20./Radius

        for j in range(User_num):
            self.state_raw[(UAV_num + User_num) * 2 + j] = 1.

        #self.energy_budget = np.random.uniform(E_max, E_max)
        self.state_raw[2*(UAV_num+User_num) + User_num] = 1.
        self.energy_budget = E_max

        self.state_raw = np.array(self.state_raw)

        self.tra_num = 0

        return self.state_raw

    def reset_single(self,obs, initial_position):
        self.current_step = 0

        self.state[0] = obs[0]
        self.state[1] = obs[1]
        self.state[2] = obs[2]
        self.state[3] = obs[3]
        self.state[4] = obs[4]
        self.state[5] = obs[5]
        self.state = np.array(self.state)

        self.trajectory[self.current_step][0] = self.state[0]*Radius
        self.trajectory[self.current_step][1] = self.state[1]*Radius


        self.Data_size[self.current_step][0] = 0
        self.Data_size[self.current_step][1] = 0.

        self.UAV_initial_position[0] = initial_position[0]*Radius
        self.UAV_initial_position[1] = initial_position[1]*Radius

        return self.state

    def calculate_tr_s(self, state):
        rs = np.zeros((UAV_num, User_num), np.float32)
        for i in range(UAV_num):
            for j in range(User_num):
                G = g0 / (np.power((state[i * 2]*Radius - state[UAV_num * 2 + j * 2]*Radius), 2) + np.power(
                    (state[i * 2 + 1]*Radius - state[UAV_num * 2 + j * 2 + 1]*Radius), 2) + np.power(H, 2))
                rs[i][j] = B * np.log2(1 + P_tr * G / (sigma2*B))

        return rs

    def calculate_energy(self,L,state):
        E_com = 0.

        for i in range(UAV_num):
            for j in range(User_num):
                A = np.power(2, L[j] / B / delta * User_num) - 1
                G = g0 / (np.power((state[i * 2] - state[UAV_num * 2 + j * 2])*Radius, 2) + np.power(
                    (state[i * 2 + 1] - state[UAV_num * 2 + j * 2 + 1])*Radius, 2) + np.power(H, 2))
                E_com += sigma2 * B * delta / User_num / G * A

        return E_com

    def step(self, action):
        self.current_step += 1

        #更新无人机及用户位置
        action[2] = (action[2]+1)/2.
        action = action * ACTION_BOUND
        state_next = np.zeros(6, dtype=np.float32)
        state_next[0] = (self.state[0]*Radius + action[1] * delta * np.cos(action[0]))/X_max
        state_next[1] = (self.state[1]*Radius + action[1] * delta * np.sin(action[0]))/Y_max
        state_next[2] = self.state[2]
        state_next[3] = self.state[3]

        #记录无人机航迹和数传过程
        if self.current_step < N:
            self.trajectory[self.current_step][0] = state_next[0]*Radius
            self.trajectory[self.current_step][1] = state_next[1]*Radius
            self.Data_size[self.current_step][0] = self.current_step
            self.Data_size[self.current_step][1] = action[2]


        #对剩余数据量进行更新
        state_next[4] = (self.state[4]*L_max - action[2])/L_max
        if state_next[4] <= 0:
            state_next[4] = 0.

        #对剩余能量进行更新
        #E_com = self.calculate_energy(action[2], state_next)
        delataV = np.sqrt(np.power(self.state[0]*Radius - state_next[0]*Radius, 2) + np.power(self.state[1]*Radius - state_next[1]*Radius, 2))/delta
        FlyingEnergy = 0.5*Mass*delataV*delataV*delta
        self.FlyingEnergy_max = 0.5 * Mass * UAV_max_velocity ** 2 * delta
        state_next[5] = (self.state[5]*E_max - FlyingEnergy)/E_max
        if state_next[5] <= 0.:
            state_next[5] = 0.

        #形成奖励
        g = g0 / (np.power((state_next[0] - state_next[2])*Radius, 2) + np.power((state_next[1] - state_next[3])*Radius, 2) + np.power(H, 2))
        g_max = g0 / np.power(H, 2)
        reward = 2*(action[2]/Lc_max)*(g/g_max - 0.9)

        #形成下一时刻状态量
        state_next = np.array(state_next)

        #检查是否完成
        if state_next[4] == 0:
            done = True
            #reward += (N - self.current_step)*2*(g/g_max - 0.9)
        else:
            done = False
        if self.current_step >= N:
            done = True
            reward -= 50
        if state_next[5] == 0:
            done = True
            reward -= 50

        #更新State并返回相关数据
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

        # fig2 = plt.figure(2)
        # plt.plot(self.Data_size[0, 0:self.current_step, 0], self.Data_size[0, 0:self.current_step, 1])
        # plt.draw()
        # axes = plt.gca()
        # axes.set_xlim([0, N])
        # axes.set_ylim([0, Lc_max])
        #
        # fig3 = plt.figure(3)
        # plt.plot(self.Data_size[1, 0:self.current_step, 0], self.Data_size[1, 0:self.current_step, 1])
        # plt.draw()
        # axes = plt.gca()
        # axes.set_xlim([0, N])
        # axes.set_ylim([0, Lc_max])


        # plt.axis('equal')
        plt.pause(2)  # 间隔的秒数： 4s
        plt.close(fig1)
        # plt.close(fig2)
        # plt.close(fig3)

    def combine(self):
        for i in range(self.current_step):
            self.combine_tra[self.tra_num][0] = self.trajectory[i][0] + self.UAV_initial_position[0]
            self.combine_tra[self.tra_num][1] = self.trajectory[i][1] + self.UAV_initial_position[1]
            self.tra_num += 1

    def close(self):
        pass
