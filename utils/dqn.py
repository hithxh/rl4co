import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
import configargparse
import random
import gym

@dataclass
class ReplayBuffer:
    maxsize: int
    size: int = 0
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    next_state: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)

    def push(self, state, action, reward, done, next_state):
        if self.size < self.maxsize:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.done.append(done)
            self.next_state.append(next_state)
        else:
            position = self.size % self.maxsize
            self.state[position] = state
            self.action[position] = action
            self.reward[position] = reward
            self.done[position] = done
            self.next_state[position] = next_state
        self.size += 1

    def sample(self, n):
        total_number = self.size if self.size < self.maxsize else self.maxsize
        indices = np.random.randint(total_number, size=n)
        state = [self.state[i] for i in indices]
        action = [self.action[i] for i in indices]
        reward = [self.reward[i] for i in indices]
        done = [self.done[i] for i in indices]
        next_state = [self.next_state[i] for i in indices]
        return state, action, reward, done, next_state




class QNet(nn.Module):
    """QNet.
    Input: feature
    Output: num_act of values
    """

    def __init__(self, dim_state, num_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DQN(nn.Module):
    def __init__(self, dim_state, num_action,discount=0.99):
        super().__init__()
        self.qnet = QNet(dim_state, num_action)
        self.discount = discount
        self.target_qnet = QNet(dim_state, num_action)
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def action(self,state):
        with torch.no_grad():
            return self.qnet(state).argmax(dim=1).item()
        
    def loss(self, batch):
        state, action, reward, next_state, done = batch
        q = self.qnet(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = reward + self.discount * self.target_qnet(next_state).max(dim=1)[0] * (1 - done)
        return F.mse_loss(q, target_q.detach())
    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)

def train(args,env,agent):
    replaybuffer = ReplayBuffer(args.replay_size)
    optimizer = torch.optim.Adam(agent.qnet.parameters(), lr=args.lr)
    state = env.reset()

    for step in range(args.max_steps):
        if step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.action(torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0)).item()
        next_state, reward, done, _ = env.step(action)
        replaybuffer.push(state, action, reward, done, next_state)


    if step >= args.warmup_steps:
        batch = replaybuffer.sample(args.batch_size)
        loss = agent.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step: {step}, Loss: {loss.item()}")
        if step % 1000 == 0:
            agent.target_qnet.load_state_dict(agent.qnet.state_dict())
        state = next_state
        if done:
            state = env.reset()

def eval(args,env,agent):
    pass
        
    

if __name__ == "__main__":
    # parser = configargparse.ArgumentParser(default_config_files=['./config/dqn.yaml'])
    parser = configargparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--replay_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--warmup_steps", default=10_000, type=int, help="Warmup steps without training.")


    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env)
    set_seed(args)
    agent = DQN(args.dim_state, args.num_action, args.discount).to(args.device)
    if args.tain:
        train(args,env,agent)
    elif args.eval:
        eval(args,env,agent)