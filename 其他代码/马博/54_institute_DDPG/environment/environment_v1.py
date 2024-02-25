import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt


# 自定义环境模拟我方信道选择策略，敌方策略还是对我方历史选择的最大干扰
class CommunicationEnv(gym.Env):

    def __init__(self):
        super(CommunicationEnv, self).__init__()

        self.channels = 20  # 可用信道数量
        self.enemy_accuracy = 0.6  # 敌方判断准确率，还是负值，越小敌方越广泛搜索，

        # 定义动作空间和观测空间
        # 动作空间，还是对每个信道进行设置两种状态[1：为真链路, 2：为假链路]×所有信道数目 = 20 × 2，
        self.action_space = spaces.Discrete(self.channels * 2)
        # 观测空间，还是判断20×2个矩阵，每行[0]为1是选为真信道, 每行[1]为1是该信道选为假信道
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.channels, 2), dtype = np.int32)

        # 初始化状态
        self.state = None
        self.enemy_guess_record = None
        self.time_step = 0

    # 重置环境
    def reset(self):
        self.state = np.zeros((self.channels, 2))  # 初始化我方的状态
        self.enemy_guess_record = np.zeros(self.channels)  # 初始化敌方对我方的记录数组
        self.time_step = 0
        return self.state

    def step(self, action):
        """
        执行动作，返回新状态、奖励、完成标志和调试信息
        """
        done = False
        reward = 0

        # 解析动作
        real_channel = action % self.channels
        fake_channel = (action // self.channels) % self.channels

        # 敌方干扰逻辑
        if np.random.rand() < self.enemy_accuracy:
            disturbed_channel = np.argmax(self.enemy_guess_record)
        else:
            disturbed_channel = np.random.choice([real_channel, fake_channel])

        # 更新敌方判断记录
        self.enemy_guess_record[real_channel] += 1

        # 更新状态
        self.state = np.zeros((self.channels, 2))
        self.state[real_channel, 0] = 1  # 真实信道
        self.state[fake_channel, 1] = 1  # 假信道

        # 计算奖励
        reward = 1 if disturbed_channel != real_channel else -1

        # 更新时间步并检查是否结束
        self.time_step += 1
        if self.time_step >= 50:
            done = True

        return self.state, reward, done, {}

    def render(self, mode='human'):
        """
        可视化当前环境状态
        """
        if mode == 'human':
            # 设置颜色
            real_channel_color = (0, 1, 0)  # 绿色，真实信道
            fake_channel_color = (0, 0, 1)  # 蓝色，假信道
            unused_channel_color = (0.7, 0.7, 0.7)  # 灰色，未使用的信道

            # 创建图形
            plt.figure(figsize=(10, 6))
            plt.title(f"Time Step: {self.time_step}")
            plt.xlabel("Channel")
            plt.ylabel("Status")

            # 绘制信道状态
            for i in range(self.channels):
                if self.state[i, 0] == 1:
                    # 真实信道
                    plt.bar(i, 1, color=real_channel_color)
                elif self.state[i, 1] == 1:
                    # 假信道
                    plt.bar(i, 1, color=fake_channel_color)
                else:
                    # 未使用的信道
                    plt.bar(i, 1, color=unused_channel_color)

            # 显示图形
            plt.show()
        else:
            super(CommunicationEnv, self).render(mode=mode)

    def close(self):
        """
        清理环境资源
        """
        pass
