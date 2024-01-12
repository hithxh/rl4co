import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import argparse
import gym

buffer = deque(maxlen=1000)

class QNet(nn.Module):
    def __init__(self, dim_input: int, num_action: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN(nn.Module):
    def __init__(self, dim_input=None, num_action=None, discount_factor=0.99):
        super().__init__()
        self.discount_factor = discount_factor
        self.q_net = QNet(dim_input, num_action)
        self.target_q_net = QNet(dim_input, num_action)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def choose_action(self, state):
        action = self.q_net(state)
        return action.argmax().item()

    def learn(self, batch):
        transitions = random.sample(buffer,batch)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.cat(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones)

        # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        # 根据actions索引在训练网络的输出的第1维度上获取对应索引的q值（state_value）
        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1
    def soft_update(self, target, source, tau=0.99):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1 - tau))

def train(args, env, agent):
    agent.q_net.train()
    for i in range(args.episodes):
        episode_reward = 0
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            buffer.append((state, action, reward, done, next_state))
            state = next_state

            if len(buffer) > 1000:
                agent.learn(100)

            if done:
                break
        agent.soft_update(agent.target_Q, agent.Q)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument("--dim_state", default=4, type=int, help="Dimension of state.")
    parser.add_argument("--num_action", default=2, type=int, help="Number of action.")
    parser.add_argument("--discount", default=0.99, type=float, help="Discount coefficient.")
    parser.add_argument("--max_steps", default=100_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Number of")
    args = parser.parse_args()
    env = gym.make('CartPole-v1')
    agent = DQN(dim_input=args.dim_state, num_action=args.num_action, discount_factor=args.discount)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    # agent.Q.to(device)
    # agent.target_Q.to(device)
    train(args, env, agent)

if __name__ == "__main__":
    main()