<<<<<<< HEAD
"""
test

一个机器人位于一个 m x n 网格的左上角。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。
问总共有多少条不同的路径？

test：
2*2->2
3*2->

C_m^n组合数 C(m,m-1)*C(n,n-1)?
"""
import math
from typing import List
import numpy as np


def func1(m, n) -> int:
    # 组合数 C(m,n)=m...(m-n)/n!=m!/(m-n)!n!
    res = int(math.factorial(m) / (math.factorial(m - n) * math.factorial(n)))
    return res


def func(m, n) -> int:
    dp = [[0] * n for i in range(m)]
    # dp[i][j] 截止到i,j的可行路径数目
    # dp[i+1][j+1] = dp[i][j+1] + dp[i+1][j]

    # init
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1

    # dp
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]  # + dp[i - 1][j - 1]

    return dp[m - 1][n - 1]


if __name__ == "__main__":
    m = 2
    n = 2
    print(func(m, n))  # test
    # print(func1(m, n-1)*func1(n, m-1))
=======
import gym
from gym import spaces

# 定义 action space
action_space = spaces.Tuple((
    spaces.Discrete(11),  # 第一个变量取值范围为 0-10 的整数
    spaces.Discrete(2)    # 第二个变量取值范围为 0 或 1
))

# 检查 action space 的样例
action = action_space.sample()
print(action[0])  # 输出一个随机示例：(7, 1)

# 定义 observation space
observation_space = spaces.Tuple((
    spaces.Discrete(11),  # 第一个变量取值范围为 0-10 的整数
    spaces.Discrete(2)    # 第二个变量取值范围为 0 或 1
))

state = (5, 1)  # 示例 state

# 检查 observation space 中的样例
observation = observation_space.sample()
print(observation)  # 输出一个随机示例：(3, 0)

# 获取 state 中各个变量的值
var1, var2 = state
print(var1)  # 输出第一个变量的值：5
print(var2)  # 输出第二个变量的值：1

>>>>>>> c11fe143e18899b5eb4f502dd0969ebfa23e8979
