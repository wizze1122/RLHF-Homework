import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from Gridworld import Gridworld

# ———— 通用超参 ————
SIZE       = 4
MODE       = 'player'
GAMMA      = 0.9
LR         = 1e-3
EPS_START  = 1.0
EPS_END    = 0.1
EPS_DECAY  = 1000      # linear decay steps
MEM_SIZE   = 2000
BATCH_SIZE = 128
MAX_EPISODES = 2000
TARGET_SYNC_FREQ = 100  # Double DQN: 每 100 episodes 同步一次 target 网络

ACTION_MAP = {0:'u',1:'d',2:'l',3:'r'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ———— (A) 网络结构定义 ————

class BasicDQN(nn.Module):
    def __init__(self, in_dim=64, h1=150, h2=100, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1,   h2), nn.ReLU(),
            nn.Linear(h2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=64, h1=150, h2=100, out_dim=4):
        super().__init__()
        # 共享前置层
        self.fc_shared = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1,   h2), nn.ReLU()
        )
        # Value 分支
        self.fc_value = nn.Sequential(
            nn.Linear(h2, 1)
        )
        # Advantage 分支
        self.fc_adv = nn.Sequential(
            nn.Linear(h2, out_dim)
        )
    def forward(self, x):
        x = self.fc_shared(x)
        v = self.fc_value(x)             # [batch, 1]
        a = self.fc_adv(x)               # [batch, 4]
        # Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        return v + (a - a.mean(dim=1, keepdim=True))


# ———— (B) 经验回放缓冲区 ————
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            torch.cat(s).to(device),
            torch.tensor(a, dtype=torch.int64, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.cat(s2).to(device),
            torch.tensor(done, dtype=torch.float32, device=device)
        )
    def __len__(self):
        return len(self.buffer)


# ———— (C) Agent 基类 ————
class DQNAgent:
    def __init__(self, net, double=False):
        self.net       = net.to(device)
        self.target_net= net.__class__().to(device) if double else None
        if self.target_net:
            self.target_net.load_state_dict(self.net.state_dict())
        self.opt       = optim.Adam(self.net.parameters(), lr=LR)
        self.replay    = ReplayBuffer(MEM_SIZE)
        self.eps       = EPS_START
        self.double    = double
        self.steps     = 0

    def select_action(self, state):
        # ε-greedy
        if random.random() < self.eps:
            return random.randrange(4)
        else:
            with torch.no_grad():
                q = self.net(state)
                return q.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return None
        s, a, r, s2, done = self.replay.sample(BATCH_SIZE)
        # 当前 Q(s,a)
        q_vals = self.net(s).gather(1, a.unsqueeze(1)).squeeze()
        # 计算 target Q
        with torch.no_grad():
            if self.double:
                # Double DQN: 选动作用 online net，评估用 target net
                next_a = self.net(s2).argmax(dim=1, keepdim=True)
                q_next = self.target_net(s2).gather(1, next_a).squeeze()
            else:
                # Basic / Dueling 一般DQN
                q_next = self.net(s2).max(dim=1)[0]
            target = r + GAMMA * q_next * (1 - done)
        loss = nn.MSELoss()(q_vals, target)
        # 反向传播
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # ε 线性衰减
        self.steps += 1
        self.eps = max(EPS_END, EPS_START - self.steps / EPS_DECAY)
        return loss.item()

    def sync_target(self):
        if self.target_net:
            self.target_net.load_state_dict(self.net.state_dict())


# ———— (D) 训练与比较 ————

def run_training(agent, label):
    episode_rewards = []
    losses = []
    for ep in range(1, MAX_EPISODES+1):
        game = Gridworld(size=SIZE, mode=MODE)
        s = torch.from_numpy(game.board.render_np().reshape(1,64)).float().to(device)
        total_r = 0
        done = False
        while not done:
            a = agent.select_action(s)
            game.makeMove(ACTION_MAP[a])
            r = game.reward()
            s2 = torch.from_numpy(game.board.render_np().reshape(1,64)).float().to(device)
            done_flag = (abs(r) == 10)
            agent.replay.push(s, a, r, s2, done_flag)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            total_r += r
            s = s2
            if done_flag:
                break
        episode_rewards.append(total_r)
        # Double DQN 同步 target
        if agent.double and ep % TARGET_SYNC_FREQ == 0:
            agent.sync_target()
    return episode_rewards, losses


if __name__ == "__main__":
    # 1) Basic DQN
    basic_agent = DQNAgent(BasicDQN(), double=False)
    r1, l1 = run_training(basic_agent, "Basic")

    # 2) Double DQN
    double_agent = DQNAgent(BasicDQN(), double=True)
    r2, l2 = run_training(double_agent, "Double")

    # 3) Dueling DQN （本质也是 DQN，但网络结构换成 Dueling）
    dueling_agent = DQNAgent(DuelingDQN(), double=False)
    r3, l3 = run_training(dueling_agent, "Dueling")

    # ———— 绘图对比 ————
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(r1, label="Basic")
    plt.plot(r2, label="Double")
    plt.plot(r3, label="Dueling")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend()
    plt.title("Reward Comparison (player mode)")

    plt.subplot(1,2,2)
    plt.plot(l1, label="Basic", alpha=0.5)
    plt.plot(l2, label="Double", alpha=0.5)
    plt.plot(l3, label="Dueling", alpha=0.5)
    plt.xlabel("Training Steps"); plt.ylabel("Loss"); plt.legend()
    plt.title("Loss Curves")
    plt.tight_layout()
    plt.show()
