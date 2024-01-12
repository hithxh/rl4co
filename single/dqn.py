import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple,deque
import argparse
import gym
import numpy as np
# from torchrl.data import ReplayBuffer

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward','done','next_state'))
class ReplayBuffer():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


# class ReplayMemory():
#     def __init__(self, capacity):
#         # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
#         self.buffer = deque(maxlen=capacity)
#     # 将数据以元组形式添加进经验池
#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#     # 随机采样batch_size行数据
#     def sample(self, batch_size):
#         transitions = random.sample(self.buffer, batch_size)  # list, len=32
#         # *transitions代表取出列表中的值，即32项
#         state, action, reward, next_state, done = zip(*transitions)
#         return np.array(state), action, reward, np.array(next_state), done
#     # 目前队列长度
#     def size(self):
#         return len(self.buffer)

class QNet(nn.Module):
    def __init__(self, dim_input:int, num_action:int):
        super().__init__()
        self.fc1 = nn.Linear(dim_input,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_action)
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN(nn.Module):
    def __init__(self, dim_input=None,num_action=None,discount_factor=0.99):
        super().__init__()
        self.discount_factor = discount_factor
        self.Q = QNet(dim_input,num_action)
        self.target_Q = QNet(dim_input,num_action)
        self.target_Q.load_state_dict(self.Q.state_dict())
    def choose_action(self,state):
        action = self.Q(state)
        return action.argmax()
    def learn(self, state, action, reward, done,next_state):
        qvals = self.Q(state).gather(1, action.unsqueeze(1)).squeeze()
        target_qvals = self.target_Q(next_state).detach().max(dim=1)
        loss = F.mse_loss(reward+self.discount_factor* target_qvals*(1-done),qvals)
        return loss
    def soft_update(self, target, source,tau=0.99):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data*tau + param.data*(1-tau))




def train(args,env,agent):
    optimizer = torch.optim.Adam(agent.Q.parameters(), lr=args.lr)
    memory = ReplayBuffer(1000)
    agent.Q.train()


    for i in range(args.episodes):
        if i < args.warmup_steps:
            # episode_reward = 0
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            action = agent.choose_action(state)
            action = action.item()
            next_state, reward,terminated,truncated, _ = env.step(action)
            done = terminated or truncated
            # episode_reward += reward
            # action = torch.tensor(action, dtype=torch.float32)
            # reward = torch.tensor(reward, dtype=torch.float32)
            memory.add(state, action, reward, done, next_state)
            state = next_state
            # state = torch.tensor(next_state, dtype=torch.float32)
        else:
            # transitions = memory.sample(batch_size=args.batch_size)
            # batch = Transition(*zip(*transitions))
            # state_batch = torch.cat(batch.state)
            # action_batch = torch.cat(batch.action)
            # reward_batch = torch.cat(batch.reward)
            # next_state_batch = torch.cat(batch.next_state)
            # done_batch = torch.cat(batch.done)

            state, action, reward, next_state, done = memory.sample(batch_size=args.batch_size)

            loss = agent.learn(state, action, reward, done, next_state)

            # loss = agent.learn(state_batch, action_batch, reward_batch, done_batch, next_state_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    env = gym.make('CartPole-v1')
    agent = DQN(dim_input=args.dim_state, num_action=args.num_action, discount_factor=args.discount)
    # agent.Q.to(device)
    # agent.target_Q.to(device)
    train(args,env,agent)


if __name__ == "__main__":
    main()