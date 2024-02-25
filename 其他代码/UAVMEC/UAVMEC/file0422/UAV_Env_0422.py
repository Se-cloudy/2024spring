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

STATE_DIM = 2*UAV_num + Other_UAV_num*2 + 2
ACTION_DIM = 1
ACTION_BOUND = np.array(np.pi/2, dtype=np.float32)

class UAV_env(gym.Env):
    """
      Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10):
        super(UAV_env, self).__init__()

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1]), dtype=np.float32)
        self.seed()
        self.current_step = 0
        self.rs_max = 0
        self.FlyingEnergy_max = 0
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.trajectory = np.zeros((N, 2), np.float32)
        self.obstacle = np.zeros((N, Other_UAV_num, 2), np.float32)
        self.prob = np.zeros(N, np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0

        for i in range(UAV_num):
            self.state[i * 2] = 0./X_max
            self.state[i * 2 + 1] = 0./Y_max

        for j in range(Other_UAV_num):
            self.state[2*UAV_num + j*2] = np.random.uniform(-d_safe, d_safe)/Radius
            self.state[2*UAV_num + j*2 + 1] = np.random.uniform(-d_safe, d_safe)/Radius

        self.state[2*(UAV_num + Other_UAV_num)] = np.random.uniform(-UAV_max_velocity, UAV_max_velocity)/UAV_max_velocity
        self.state[2 * (UAV_num + Other_UAV_num) + 1] = np.random.uniform(-np.pi/2, np.pi/2)/(np.pi/2)

        self.state = np.array(self.state)

        self.trajectory[self.current_step][0] = self.state[0]*Radius
        self.trajectory[self.current_step][1] = self.state[1]*Radius

        return self.state

    def calculate_prob(self,state):
        prob = np.zeros(Other_UAV_num, np.float32)
        for i in range(Other_UAV_num):
            d = np.sqrt(np.power((state[2*UAV_num + i*2] - state[0])*Radius,2) + np.power((state[2*UAV_num + i*2 + 1] - state[1])*Radius,2))
            d_for = np.sqrt(np.power((self.state[2*UAV_num + i*2] - self.state[0])*Radius,2) + np.power((self.state[2*UAV_num + i*2 + 1] - self.state[1])*Radius,2))
            prob[i] = 3.5/np.sqrt(2*np.pi)/obstacle_sigma*math.exp(-d**2/2/obstacle_sigma)

        return prob


    def step(self, action):
        self.current_step += 1

        #更新无人机及用户位置
        action = action * ACTION_BOUND
        state_next = np.zeros(STATE_DIM, dtype=np.float32)
        state_next[0] = (self.state[0]*Radius + self.state[2*(UAV_num + Other_UAV_num)]*UAV_max_velocity*delta*np.cos(action[0] + self.state[2*(UAV_num + Other_UAV_num)+1]*np.pi/2))/Radius
        state_next[1] = (self.state[1]*Radius + self.state[2*(UAV_num + Other_UAV_num)]*UAV_max_velocity*delta*np.sin(action[0] + self.state[2*(UAV_num + Other_UAV_num)+1]*np.pi/2))/Radius

        for j in range(Other_UAV_num):
            state_next[2*UAV_num + j*2] = self.state[2*UAV_num + j*2]
            state_next[2*UAV_num + j*2 + 1] = self.state[2*UAV_num + j*2 + 1]

        state_next[2*(UAV_num + Other_UAV_num)] = self.state[2*(UAV_num + Other_UAV_num)]
        state_next[2 * (UAV_num + Other_UAV_num) + 1] = self.state[2 * (UAV_num + Other_UAV_num) + 1] + action[0]/(np.pi/2)


        #记录无人机航迹
        if self.current_step < N:
            self.trajectory[self.current_step][0] = state_next[0]*Radius
            self.trajectory[self.current_step][1] = state_next[1]*Radius

        #计算碰撞概率
        prob = self.calculate_prob(state_next)
        self.prob = sum(prob)/Other_UAV_num

        #形成奖励
        reward = -self.prob

        #形成下一时刻状态量
        state_next = np.array(state_next)

        #检查是否完成
        if self.prob < 0.2619:
            done = True
        else:
            done = False

        #更新State并返回相关数据
        self.state = state_next
        return state_next, reward, done, {}

    def render(self,mode='human'):
        fig1 = plt.figure(1)
        plt.plot(self.trajectory[0:self.current_step, 0], self.trajectory[0:self.current_step, 1])

        for j in range(Other_UAV_num):
            plt.plot(self.state[2*UAV_num + j*2]*Radius, self.state[2*UAV_num + j*2 + 1]*Radius, 'v')
        plt.plot(self.trajectory[0, 0], self.trajectory[0, 1], '.r')
        plt.draw()

        axes = plt.gca()
        axes.set_xlim([-1, 1])
        axes.set_ylim([-1, 1])
        # plt.axis('equal')
        plt.pause(2)  # 间隔的秒数： 4s
        plt.close(fig1)

    def close(self):
        pass
