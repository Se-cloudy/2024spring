import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import numpy as np
from environment.environment_v4 import CommunicationEnv
import matplotlib.pyplot as plt


# v2版本忘了保存，v3版本是在v2版本上的改动，v2版本主要是对环境设置进行了改动，将动作空间从20×2维度变成1×2，用于表示我方[示假, 隐真]的两个信道
# v3版本：至少奖励函数是正的了，偶尔有惊喜总奖励50，能达到48，不知道奖励函数最终是否收敛
# v4版本的改进方向：
#           1、实现可视化，并保存高奖励时刻的信道占用和敌方攻击时的时刻图【待做】
#           2、实现均值和全部奖励的可视化视图【待做】
#           4、经验回放池函数写错了，应该是50步走完将整体存入回放池，不是一步一步存！！！！！！！！！！！！！！！！！！！
#           3、奖励函数的归一化【看文献，突然想到了，本周20231211待做一下】
# v4版本的实际改动：
#           1、环境采用environment_v3：将我方的状态也描述为[真链路, 假链路, 敌方干扰链路]
#           2、Actor网络的输出是softmax函数的概率(所有信道个数下的概率分布),选择概率最大的为真信道，选择概率最小的为假信道(迷惑干扰机)

# np.random.seed(32)
# 定义动作网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        # self.fc1 = nn.Linear(state_dim, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 20)
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        # self.fc4 = nn.Linear(512, 20)
        self.fc4 = nn.Linear(512, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.softmax(self.fc4(x), dim=-1)
        #x = torch.tanh(self.fc4(x)) * torch.tensor(self.action_bound)
        x = torch.relu(self.fc4(x)) * torch.tensor(self.action_bound)
        return x


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # self.fc1 = nn.Linear(state_dim + action_dim, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 1)
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


env = CommunicationEnv()

# 设定参数
state_dim = env.observation_space.shape[1]  # 状态空间的维度
action_dim = env.action_space.shape[1]  # 动作空间的维度
action_bound = env.action_space.high[0]
# lr_actor = 3e-4  # Actor网络的学习率
# lr_critic = 3e-3  # Critic网络的学习率
lr_actor = 2e-2  # Actor网络的学习率
lr_critic = 2e-2  # Critic网络的学习率
gamma = 0.99  # 折扣因子
tau = 0.005  # 目标网络软更新参数
buffer_size = 100000  # 经验回放的大小
batch_size = 256  # 训练批次大小
num_episodes = 1000  # 训练次数
max_steps_per_episode = 100
channels = 20  # 信道数目
actor_loss_list = []
critic_loss_list = []

actor = Actor(state_dim, action_dim, action_bound)
critic = Critic(state_dim, action_dim)
actor_target = Actor(state_dim, action_dim, action_bound)
critic_target = Critic(state_dim, action_dim)

# 将目标网络的权重初始化为与主网络相同
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

# 设置优化器
optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

# 初始化记忆回放
memory = ReplayBuffer(buffer_size)

# 动作选择函数
# 此函数定义在actor(state)前，是为了将actor网络输出层的softmax函数输出的len(channel)个数映射到具体的信道选择上
# 选择最大的概率为真信道，最小的概率为假信道
"""
def select_action(state, actor, epsilon=0.1):
    if random.random() < epsilon:
        # 以 epsilon 的概率进行随机探索
        return np.random.choice(20, 2, replace=False)  # 随机选择两个不同的动作
    else:
        # 以 1-epsilon 的概率选择最优动作
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
        action_probs = actor(state).detach().numpy()[0]
        action_mid = np.zeros(2)
        actions = np.argsort(action_probs)[:]
        action_mid[0] = actions[0]  # 最大概率为真信道
        action_mid[1] = actions[-1]  # 最小概率为假信道
        # action_mid[0] = actions[-1]  # 最小概率为真信道
        # action_mid[1] = actions[-0]  # 最大概率为假信道
        return action_mid
"""


def select_action(state, actor):
    state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
    action_probs = actor(state).detach().numpy()[0]
    action_probs += action_probs + 0.01 * np.random.randn(len(action_probs))
    return action_probs


# 此函数用在target_actor前，将概率映射成(采样批次, 一批次数据)合适的(采样批次, 一次信道选择)
def change_target_action(action_probs):
    # 初始化一个空数组用于存储每个样本的两个最高概率动作
    action_shape = action_probs.shape[0]
    top_two_actions = np.zeros((action_shape, 2), dtype=int)
    ac = action_probs.detach().numpy()
    # 遍历每个样本
    for i in range(action_shape):
        # 对每个样本的动作概率进行降序排序并取前两个
        # top_two_actions[i] = np.argsort(ac[i])[-2:]
        actions = np.argsort(ac[i])[:]
        top_two_actions[i, 0] = actions[0]  # 最大概率为真信道
        top_two_actions[i, 1] = actions[-1]  # 最大概率为真信道
        # top_two_actions[i, 0] = actions[-1]  # 最小概率为真信道
        # top_two_actions[i, 1] = actions[0]  # 最大概率为真信道
    return torch.tensor(top_two_actions)


def update_network(batch_size, gamma, tau):
    if len(memory) < batch_size:
        return

    # 提取批数据
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.tensor(np.array(batch.state).reshape(batch_size, -1), dtype=torch.float32)
    actions = torch.tensor(np.array(batch.action).reshape(batch_size, -1), dtype=torch.float32)
    rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32)
    next_states = torch.tensor(np.array(batch.next_state).reshape(batch_size, -1), dtype=torch.float32)
    dones = torch.tensor(np.array(batch.done), dtype=torch.float32)

    # 计算Critic网络的损失
    target_actions = actor_target(next_states)  # 保留此行，重写下一行
    # target_actions = change_target_action(target_actions)  # 从128×2中选择128×40的动作，现在动作维度不对
    target_values = critic_target(next_states, target_actions)
    expected_values = rewards.view(-1, 1) + gamma * target_values * (1 - dones.view(-1, 1))
    # actions = change_action(actions)
    values = critic(states, actions)
    critic_loss = F.mse_loss(values, expected_values)
    critic_loss_list.append(critic_loss)
    # print(critic_loss)
    # print(expected_values)

    # 更新Critic网络

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    # 计算Actor网络的损失
    actions_pred = actor(states)
    # actions_pred = change_target_action(actions_pred)
    actor_loss = -critic(states, actions_pred).mean()
    actor_loss_list.append(actor_loss)
    # print(actor_loss)

    # 更新Actor网络
    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    # 更新目标网络
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


sum_reward = []
# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(max_steps_per_episode):
        action = select_action(state, actor)
        next_state, reward, done, _ = env.step(action)

        memory.push(state, action, next_state, reward, done)
        state = next_state
        episode_reward += reward

        update_network(batch_size, gamma, tau)

        if done:
            break
    sum_reward.append(episode_reward)
    print(f"Episode {episode}, Total Reward: {episode_reward}")

# 指定保存文件的路径和文件名
file_path = "./output_data/my_array_v5.1.txt"

# 使用np.savetxt()将NumPy数组保存为文本文件
np.savetxt(file_path, sum_reward, fmt='%d', delimiter=',')
plt.plot(np.arange(len(sum_reward)), sum_reward)
plt.title("Simulation of channel occupancy and interference at each moment")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig('./output_fig/奖励函数_v5.1.png')
plt.show()
torch.save(actor.state_dict(), './model/actor_v5.bin')
torch.save(critic.state_dict(), './model/critic_v5.bin')
