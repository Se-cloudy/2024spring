import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding
import math

N = 50
P_tr = 0.1
sigma2 = 1e-14#-110dbm
X_max = 30.
Y_max = 30.
Radius = 30.
L_max = 200*1024.#kB
UAV_num = 1
Other_UAV_num = 1
User_num = 1
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
obstacle_sigma = 0.469

STATE_DIM = 2*(UAV_num+User_num) + User_num + 1 + Other_UAV_num*2
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
        self.observation_space = spaces.Box(low=np.array([0,0,0,0,0,0,0,0]), high=np.array([X_max,Y_max,X_max,Y_max,L_max,E_max,3,3]), dtype=np.float32)
        self.seed()
        self.current_step = 0
        self.rs_max = 0
        self.FlyingEnergy_max = 0
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.trajectory = np.zeros((N, 2), np.float32)
        self.obstacle = np.zeros((Other_UAV_num, 2), np.float32)
        self.prob = np.zeros(N, np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        self.prob = 0

        for i in range(UAV_num):
            self.state[i * 2] = 0./X_max
            self.state[i * 2 + 1] = 0./Y_max

        for j in range(User_num):
            theta = np.random.uniform(0, 2*np.pi)
            Distance = np.random.uniform(29, Radius)
            self.state[UAV_num * 2 + j * 2] = np.cos(theta)*Distance/Radius
            self.state[UAV_num * 2 + j * 2 + 1] = np.sin(theta)*Distance/Radius
            #self.state[UAV_num * 2 + j * 2] = 20./Radius
            #self.state[UAV_num * 2 + j * 2 + 1] = 20./Radius

        for j in range(User_num):
            self.state[(UAV_num + User_num) * 2 + j] = 1.
            #self.state[(UAV_num + User_num) * 2 + j] = np.random.uniform(150*1024, L_max)/L_max

        self.energy_budget = np.random.uniform(4900, E_max)
        self.state[2*(UAV_num+User_num) + User_num] = self.energy_budget/E_max

        for j in range(Other_UAV_num):
            self.state[2 * (UAV_num + User_num) + User_num + 1 + j*2] = ((self.state[0] + self.state[2])*Radius/2 + np.random.uniform(-d_safe, d_safe))/Radius
            self.state[2 * (UAV_num + User_num) + User_num + 1 + j*2+1] = ((self.state[1] + self.state[3])*Radius/2 + np.random.uniform(-d_safe, d_safe))/Radius

        self.state = np.array(self.state)

        self.trajectory[self.current_step][0] = self.state[0]*Radius
        self.trajectory[self.current_step][1] = self.state[1]*Radius

        for j in range(Other_UAV_num):
            self.obstacle[j][0] = self.state[2 * (UAV_num + User_num) + User_num + 1 + j*2]*Radius
            self.obstacle[j][1] = self.state[2 * (UAV_num + User_num) + User_num + 1 + j*2+1]*Radius

        return self.state

    def calculate_tr_s(self, state):
        rs = np.zeros((UAV_num, User_num), np.float32)
        for i in range(UAV_num):
            for j in range(User_num):
                G = g0 / (np.power((state[i * 2]*Radius - state[UAV_num * 2 + j * 2]*Radius), 2) + np.power(
                    (state[i * 2 + 1]*Radius - state[UAV_num * 2 + j * 2 + 1]*Radius), 2) + np.power(H, 2))
                rs[i][j] = B * np.log2(1 + P_tr * G / (sigma2*B))

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

    def calculate_prob(self,state):
        prob = np.zeros(Other_UAV_num, np.float32)
        temp = 2*(UAV_num + User_num) + User_num + 1
        for i in range(Other_UAV_num):
            d = np.sqrt(np.power((state[temp + i*2]-state[0])*Radius,2) + np.power((state[temp + i*2 + 1]-state[1])*Radius,2))
            d_for = np.sqrt(np.power((self.state[temp + i*2]-self.state[0])*Radius,2) + np.power((self.state[temp + i*2 + 1]-self.state[1])*Radius,2))
            prob[i] = 1/np.sqrt(2*np.pi)/obstacle_sigma*math.exp(-d**2/2/obstacle_sigma)

        return prob


    def step(self, action):
        self.current_step += 1

        #更新无人机及用户位置
        action = action * ACTION_BOUND
        state_next = np.zeros(STATE_DIM, dtype=np.float32)
        state_next[0] = (self.state[0]*Radius + action[1] * delta * np.cos(action[0]))/Radius
        state_next[1] = (self.state[1]*Radius + action[1] * delta * np.sin(action[0]))/Radius
        for j in range(User_num):
            state_next[2 + j * 2] = self.state[2 + j * 2]
            state_next[2 + j * 2 + 1] = self.state[2 + j * 2 + 1]

        #更新无人机与障碍间的相对距离
        for j in range(Other_UAV_num):
            state_next[2*(UAV_num + User_num) + User_num + 1 + j*2] = self.state[2*(UAV_num + User_num) + User_num + 1 + j*2]
            state_next[2*(UAV_num + User_num) + User_num + 1 + j*2 + 1] = self.state[2*(UAV_num + User_num) + User_num + 1 + j*2 + 1]

        #记录无人机航迹
        if self.current_step < N:
            self.trajectory[self.current_step][0] = state_next[0]*Radius
            self.trajectory[self.current_step][1] = state_next[1]*Radius

        #对剩余数据量进行更新
        rs = self.calculate_tr_s(self.state)
        rs_next = self.calculate_tr_s(state_next)
        self.rs_max = B * np.log2(1 + P_tr * g0 / np.power(H, 2) / (sigma2 * B))
        for i in range(UAV_num):
            for j in range(User_num):
                state_next[(UAV_num + User_num) * 2 + j] = (self.state[(UAV_num + User_num) * 2 + j]*L_max - delta * rs[i][j])/L_max
                if state_next[(UAV_num + User_num) * 2 + j] <= 0:
                    state_next[(UAV_num + User_num) * 2 + j] = 0.

        #对剩余能量进行更新
        delataV = np.sqrt(np.power(self.state[0]*Radius - state_next[0]*Radius, 2) + np.power(self.state[1]*Radius - state_next[1]*Radius, 2))/delta
        FlyingEnergy = 0.5*Mass*delataV*delataV*delta
        self.FlyingEnergy_max = 0.5 * Mass * UAV_max_velocity ** 2 * delta
        state_next[2*(UAV_num+User_num) + User_num] = (self.state[2*(UAV_num+User_num) + User_num]*E_max - FlyingEnergy)/E_max
        if state_next[2*(UAV_num+User_num) + User_num] <= 0.:
            state_next[2 * (UAV_num + User_num) + User_num] = 0.

        #计算碰撞概率
        prob = self.calculate_prob(state_next)
        if sum(prob)/Other_UAV_num > self.prob:
            self.prob = sum(prob)/Other_UAV_num

        #形成奖励
        reward = sum(sum(rs_next))/self.rs_max - self.prob # - FlyingEnergy/self.FlyingEnergy_max*0.1

        #形成下一时刻状态量
        state_next = np.array(state_next)

        #检查是否完成
        if sum(state_next[(UAV_num + User_num) * 2:(UAV_num + User_num) * 2 + User_num]) == 0:
            done = True
            reward += (N - self.current_step)*sum(sum(rs_next))/self.rs_max
        else:
            done = False
        if self.current_step >= N:
            done = True
        if state_next[2*(UAV_num+User_num) + User_num] == 0:
            done = True
            #reward -= (N - self.current_step)

        #更新State并返回相关数据
        self.state = state_next
        return state_next, reward, done, {}

    def render(self,mode='human'):
        fig1 = plt.figure(1)
        plt.plot(self.trajectory[0:self.current_step, 0], self.trajectory[0:self.current_step, 1])
        for j in range(User_num):
            plt.plot(self.state[2 + j * 2]*Radius, self.state[2 + j * 2 + 1]*Radius, 'om')

        for j in range(Other_UAV_num):
            plt.plot(self.obstacle[j][0], self.obstacle[j][1], 'v')
        plt.plot(self.trajectory[0, 0], self.trajectory[0, 1], '.r')
        plt.draw()

        axes = plt.gca()
        axes.set_xlim([-X_max, X_max])
        axes.set_ylim([-Y_max, Y_max])
        # plt.axis('equal')
        plt.pause(2)  # 间隔的秒数： 4s
        plt.close(fig1)

    def close(self):
        pass
