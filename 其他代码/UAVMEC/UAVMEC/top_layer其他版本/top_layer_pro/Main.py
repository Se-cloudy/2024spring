import numpy as np
import os
import torch as th

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3 import DQN,PPO
from UAVMEC.top_layer_pro.Top_Env_DQN import TOP_ENV_DQN
from UAVMEC.top_layer_pro.UAV_Env_V2 import UAV_env_V2
from UAVMEC.top_layer_pro.UAV_Env_Multi import UAV_env_Multi
from UAVMEC.top_layer_pro.Reset import Reset_Env
from stable_baselines3.common.monitor import Monitor
from functools import reduce
import matplotlib.pyplot as plt
import copy
from pandas import DataFrame

Uav_num = 4
User_num = 40
Area_Bound = 30
Max_service_No = 10
N = 50


def prod(x, y):
    return x*y

def Divide_Area(x1,x2,y1,y2,state,Number):
    area_position = np.zeros((5,2*User_num), dtype=np.float32)
    area_No = np.zeros(5, dtype=np.int8)
    area_bound = np.zeros((4,4), dtype=np.float32)
    #第1块区域边界
    area_bound[0, 0] = x1
    area_bound[0, 1] = (x1 + x2) / 2
    area_bound[0, 2] = y1
    area_bound[0, 3] = (y1 + y2) / 2
    #第2块区域边界
    area_bound[1, 0] = (x1 + x2) / 2
    area_bound[1, 1] = x2
    area_bound[1, 2] = y1
    area_bound[1, 3] = (y1 + y2) / 2
    #第3块区域边界
    area_bound[2, 0] = x1
    area_bound[2, 1] = (x1 + x2) / 2
    area_bound[2, 2] = (y1 + y2) / 2
    area_bound[2, 3] = y2
    #第4块区域边界
    area_bound[3, 0] = (x1 + x2) / 2
    area_bound[3, 1] = x2
    area_bound[3, 2] = (y1 + y2) / 2
    area_bound[3, 3] = y2

    for j in range(Number):
        if state[2*j] >= x1 and state[2*j] < (x1+x2)/2 and state[2*j + 1] >= y1 and state[2*j + 1] < (y1+y2)/2:
            area_position[0, area_No[0]*2] = state[2*j]
            area_position[0, area_No[0]*2 + 1] = state[2*j + 1]
            area_No[0] += 1
        if state[2*j] >= (x1+x2)/2 and state[2*j] <= x2 and state[2*j + 1] >= y1 and state[2*j + 1] < (y1+y2)/2:
            area_position[1, area_No[1]*2] = state[2*j]
            area_position[1, area_No[1]*2 + 1] = state[2*j + 1]
            area_No[1] += 1
        if state[2*j] >= x1 and state[2*j] < (x1+x2)/2 and state[2*j + 1] >= (y1+y2)/2 and state[2*j + 1] <= y2:
            area_position[2, area_No[2]*2] = state[2*j]
            area_position[2, area_No[2]*2 + 1] = state[2*j + 1]
            area_No[2] += 1
        if state[2*j] >= (x1+x2)/2 and state[2*j] <= x2 and state[2*j + 1] >= (y1+y2)/2 and state[2*j + 1] <= y2:
            area_position[3, area_No[3]*2] = state[2*j]
            area_position[3, area_No[3]*2 + 1] = state[2*j + 1]
            area_No[3] += 1

    return area_position, area_No, area_bound

def calculate_center(area_position, area_No, area_bound):
    center = np.zeros((4,2), dtype=np.float32)

    for i in range(4):
        if area_No[i] == 0:
            center[i, 0] = (area_bound[i, 0] + area_bound[i, 1])/2
            center[i, 1] = (area_bound[i, 2] + area_bound[i, 3])/2
        else:
            for k in range(area_No[i]):
                center[i, 0] += area_position[i, 2 * k]
                center[i, 1] += area_position[i, 2 * k + 1]
            center[i, 0] = center[i, 0]/area_No[i]
            center[i, 1] = center[i, 1]/area_No[i]

    return center

def find_point(k, action_series, area_position, area_No):
    point = np.zeros(2, dtype=np.float32)

    Min_distance = 100
    for i in range(area_No[action_series[k]]):
        for j in range(area_No[action_series[k+1]]):
            distance = pow((pow((area_position[action_series[k],2*i] - area_position[action_series[k+1],2*j]), 2) + pow((area_position[action_series[k],2*i + 1] - area_position[action_series[k+1],2*j + 1]), 2)), 0.5)
            if distance < Min_distance:
                Min_distance = distance
                point[0] = area_position[action_series[k+1],2*j]
                point[1] = area_position[action_series[k+1], 2*j + 1]

    return point

def path_planning(state):
    obs = np.zeros(6, dtype=np.float32)
    position = np.zeros(2, dtype=np.float32)
    obs[0] = 0.
    obs[1] = 0.
    obs[2] = state[2] - state[0]
    obs[3] = state[3] - state[1]
    obs[4] = 1.
    obs[5] = 1.
    obs = np.array(obs)
    for time in range(N):
        low_action, _states = model_low.predict(obs, deterministic=True)
        obs, rewards, dones, info = env_low.step(obs, low_action)
        if dones:
            position[0] = obs[0] + state[0]
            position[1] = obs[1] + state[1]
            break

    return position

def planning(i, action, area_position, area_No, point_position, area_bound):
    series = np.zeros(5, dtype=np.int8)
    output_series = np.zeros(area_No[action]+1, dtype=np.int8)
    ID = 0
    dones = False

    if area_No[action] > 4:
        _ = env_top.reset()
        #划分边界
        area_position_next, area_No_next, area_bound_next = Divide_Area(area_bound[action,0], area_bound[action,1], area_bound[action,2], area_bound[action,3], user_state,User_num)
        # 确定区域中心
        center = calculate_center(area_position_next, area_No_next, area_bound_next)
        # 初始化
        state = np.zeros(2*5 + 4, dtype=np.float32)

        state[0] = Uav_position[i, 0]
        state[1] = Uav_position[i, 1]

        for j in range(4):
            state[2 + 2*j] = center[j, 0]
            state[2 + 2*j + 1] = center[j, 1]

        for j in range(4):
            if area_No_next[j] == 0:
                state[10 + j] = 1
            else:
                state[10 + j] = 0

        # 粗粒度规划
        _ = env_top.reset()
        while not dones:
            env_top.give_value(state)
            action, State_next = model_top.predict(state, deterministic=True)
            State_next, rewards, dones, info = env_top.step(action)
            series[ID] = action
            ID += 1
            state = copy.deepcopy(State_next)
            point_position[0] = 0
            point_position[1] = 0
            _ = planning(i, action, area_position_next, area_No_next, point_position, area_bound_next)

        # # 开始递归
        # for k in range(ID):
        #     if k == ID - 1:
        #         point_position[0] = 0
        #         point_position[1] = 0
        #     else:
        #         point_position = find_point(k, series, area_position_next, area_No_next)
        #     series_next = planning(i, series[k], area_position_next, area_No_next, point_position, area_bound_next)
        #     np.append(output_series, series_next)
    else:
        #状态初始化
        state = np.zeros(2*5 + 4, dtype=np.float32)
        state[0] = Uav_position[i, 0]
        state[1] = Uav_position[i, 1]
        for j in range(area_No[action]):
            state[2 + 2*j] = area_position[action,2*j]
            state[2 + 2*j + 1] = area_position[action, 2*j + 1]
            # for k in range(User_num):
            #     if area_position[action,2*j] == obs[2*Uav_num + 2*k] and area_position[action,2*j + 1] == obs[2*Uav_num + 2*k + 1]:
            #         state[2 + 2*4 + j] = Flag[k]
            #         break
        if area_No[action]<4:
            for j in range(4-area_No[action]):
                state[2 + 2*4 + area_No[action] + j] = 1
        #开始规划
        if reduce(prod, state[2 + 2*4:2 + 2*4 + 3]) == 1:
            pass
        else:
            _ = env_top.reset()
            while not dones:
                env_top.give_value(state)
                action, State_next = model_top.predict(state, deterministic=True)
                State_next, rewards, dones, info = env_top.step(action)
                output_series[ID] = action
                ID += 1
                state = State_next
                Trajectory[i, Trajectory_No[i], 0] = state[0]
                Trajectory[i, Trajectory_No[i], 1] = state[1]
                Trajectory_No[i] += 1
                # for t in range(env_top.uav_tra_No):
                #     Final_Trajectory[i, Final_Trajectory_No[i], 0] = env_top.uav_tra[t][0]
                #     Final_Trajectory[i, Final_Trajectory_No[i], 1] = env_top.uav_tra[t][1]
                #     Final_Trajectory_No[i] += 1
                Energy_Com[i] += env_top.Total_Com_Energy
                Energy_Fly[i] += env_top.Total_Fly_Energy
                Distance[i] += env_top.Total_Distance
            #更新无人机位置
            Uav_position[i, 0] = state[0]
            Uav_position[i, 1] = state[1]

        # 更新切入点状态
        for j in range(User_num):
            if obs[2*Uav_num + 2*j] == point_position[0] and obs[2*Uav_num + 2*j + 1] == point_position[1]:
                Flag[j] = 1
                break
    return output_series

def render(Trajectory, Trajectory_No, obs):
    fig1 = plt.figure(1)
    for j in range(User_num):
        plt.plot(obs[2*Uav_num + j * 2] * Area_Bound, obs[2*Uav_num + j * 2 + 1] * Area_Bound, 'om')

    for i in range(Uav_num):
        plt.plot(Trajectory[i, 0:Trajectory_No[i], 0], Trajectory[i, 0:Trajectory_No[i], 1])
    plt.draw()
    axes = plt.gca()
    axes.set_xlim([-0, Area_Bound])
    axes.set_ylim([-0, Area_Bound])

    # plt.axis('equal')
    plt.pause(5)  # 间隔的秒数： 4s
    plt.close(fig1)



if __name__ == '__main__':
    #顶层规划用的环境
    env_top = TOP_ENV_DQN()
    #底层规划用的环境
    env_low = UAV_env_Multi()
    #生成目标用的环境
    env_reset = Reset_Env()
    #模型载入
    model_top = DQN.load("./Top_best/best_model_top", env=env_top)
    model_low = PPO.load("./low_best/low_best_model", env = env_low)

    user_state = np.zeros(2*User_num, dtype=np.float32)
    State = np.zeros(14, dtype=np.float32)
    Service_No = np.zeros(Uav_num, dtype=np.int8)
    Uav_position = np.zeros((Uav_num,2), dtype=np.float32)
    action_series = np.zeros((Uav_num, 4), dtype=np.int8)
    action_No = np.zeros(Uav_num, dtype=np.int8)
    Trajectory = np.zeros((Uav_num, User_num+1, 2), dtype=np.float32)
    Trajectory_No = np.zeros(Uav_num, dtype=np.int8)
    Final_Trajectory = np.zeros((Uav_num, (User_num+1)*N, 2), dtype=np.float32)
    Final_Trajectory_No = np.zeros(Uav_num, dtype=np.int16)
    Flag = np.zeros(User_num, dtype=np.int8)
    point_position = np.zeros(2, dtype=np.float32)
    Energy_Com = np.zeros(Uav_num, dtype=np.float32)
    Distance = np.zeros(Uav_num, dtype=np.float32)
    Energy_Fly = np.zeros(Uav_num, dtype=np.float32)
    dones = False

    #初始化
    obs = env_reset.reset()
    _ = env_top.reset()

    #区域划分
    for j in range(User_num):
        user_state[2*j] = obs[2*Uav_num + 2*j]
        user_state[2*j + 1] = obs[2*Uav_num + 2*j + 1]
    area_position, area_No, area_bound = Divide_Area(0,1,0,1,user_state,User_num)

    #计算区域中心
    center = calculate_center(area_position, area_No, area_bound)

    #初步规划
    for i in range(Uav_num):
        Uav_position[i,0] = obs[2*i]
        Uav_position[i,1] = obs[2*i + 1]

    for j in range(4):
        State[2 + 2*j] = center[j,0]
        State[2 + 2*j + 1] = center[j, 1]

    for j in range(4):
        if area_No[j] == 0:
            State[10 + j] = 1
        else:
            State[10 + j] = 0

    for i in range(Uav_num):
        Trajectory[i, Trajectory_No[i], 0] = Uav_position[i, 0]
        Trajectory[i, Trajectory_No[i], 1] = Uav_position[i, 1]
        Final_Trajectory[i, Trajectory_No[i], 0] = Uav_position[i, 0]
        Final_Trajectory[i, Trajectory_No[i], 1] = Uav_position[i, 1]
        Trajectory_No[i] += 1
        Final_Trajectory_No[i] += 1

    while not dones:
        for i in range(Uav_num):
            State[0] = Uav_position[i, 0]
            State[1] = Uav_position[i, 1]
            if Service_No[i] < Max_service_No:
                action, State_next = model_top.predict(State, deterministic=True)
                env_top.give_value(State)
                State_next, rewards, dones, info = env_top.step(action)
                Service_No[i] += area_No[action]
                State = copy.deepcopy(State_next)
                # action_series[i,action_No[i]] = action
                # action_No[i] += 1
                point_position[0] = 0
                point_position[1] = 0
                _ = planning(i, action, area_position, area_No, point_position, area_bound)

    # #详细规划
    # for i in range(Uav_num):
    #     for k in range(action_No[i]):
    #         if k == action_No[i]-1:
    #             point_position[0] = 0
    #             point_position[1] = 0
    #         else:
    #             point_position = find_point(k, action_series[i,:], area_position, area_No)
    #         series = planning(i, action_series[i,k], area_position, area_No, point_position, area_bound)
            #最终整合
    Trajectory = Trajectory*Area_Bound
    df_w = DataFrame({'x1': Trajectory[0,0:Trajectory_No[0], 0], 'y1': Trajectory[0,0:Trajectory_No[0], 1]})#, 'x2': Trajectory[1,0:Trajectory_No[1], 1], 'y2': Trajectory[1,0:Trajectory_No[1], 1], 'x3': Trajectory[2,0:Trajectory_No[2], 1], 'y3': Trajectory[2,0:Trajectory_No[2], 1], 'x4': Trajectory[3,0:Trajectory_No[3], 1], 'y4': Trajectory[3,0:Trajectory_No[3], 1]})
    df_w.to_csv('Trajectory1.csv', index=False)
    df_w = DataFrame({'x1': Trajectory[1,0:Trajectory_No[1], 0], 'y1': Trajectory[1,0:Trajectory_No[1], 1]})
    df_w.to_csv('Trajectory2.csv', index=False)
    df_w = DataFrame({'x1': Trajectory[2, 0:Trajectory_No[2], 0], 'y1': Trajectory[2, 0:Trajectory_No[2], 1]})
    df_w.to_csv('Trajectory3.csv', index=False)
    df_w = DataFrame({'x1': Trajectory[3, 0:Trajectory_No[3], 0], 'y1': Trajectory[3, 0:Trajectory_No[3], 1]})
    df_w.to_csv('Trajectory4.csv', index=False)
    render(Trajectory,Trajectory_No,obs)
    Com_Energy = sum(Energy_Com)
    Fly_Energy = sum(Energy_Fly)
    Total_Distance = sum(Distance)
    print('| The energy consumption of comminication is : %.2f' % Com_Energy)
    print('| The energy consumption of Flying is : %.2f' % Fly_Energy)
    print('| The Total_Distance is : %.2f' % Total_Distance)


