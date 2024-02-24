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

