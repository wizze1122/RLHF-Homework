#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
naive_dqn_static.py

在 Gridworld 的 static 模式下訓練一個最簡 DQN，並測試它能否穩定通過迷宮。
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from Gridworld import Gridworld

# 1. 超參數
L1, L2, L3, L4 = 64, 150, 100, 4
gamma      = 0.9
epsilon    = 1.0
learning_rate = 1e-3
epochs     = 1000

# 2. 建立 Q-網路
model = torch.nn.Sequential(
    torch.nn.Linear(L1, L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2, L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3, L4)
)
loss_fn   = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3. 動作對照表
action_set = {0:'u', 1:'d', 2:'l', 3:'r'}

# 4. 主訓練迴圈 (static mode)
losses = []
for ep in range(epochs):
    game = Gridworld(size=4, mode='static')
    state_np = game.board.render_np().reshape(1,64) + np.random.rand(1,64)*0.1
    state = torch.from_numpy(state_np).float()
    done = False

    while not done:
        # 1) 選動作
        qvals = model(state)
        if random.random() < epsilon:
            a = random.randint(0,3)
        else:
            a = torch.argmax(qvals).item()

        # 2) 執行動作
        game.makeMove(action_set[a])
        next_np = game.board.render_np().reshape(1,64) + np.random.rand(1,64)*0.1
        next_s  = torch.from_numpy(next_np).float()
        r = game.reward()

        # 3) 計算 target Q 值 (先轉成 Python float)
        with torch.no_grad():
            max_next_q = model(next_s).max().item()
        # 這裡用 abs(r)!=1 判斷是否為 terminal (r=10或-10)
        if abs(r) != 1:
            target_val = float(r)             # terminal: Q_target = r
        else:
            target_val = r + gamma * max_next_q  # non-terminal: Q_target = r + γ·maxQ

        # 4) 轉成 float32 scalar tensor
        y = torch.tensor(target_val, dtype=torch.float32)

        # 5) 取出本網路對 a 的預測 Q(s,a)
        x = qvals.squeeze()[a]  # shape=[]，也是 float32

        # 6) 計算並回傳梯度
        loss = loss_fn(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_s
        if abs(r) == 10:  # r=+10 或 -10 結束
            done = True

    losses.append(loss.item())
    # ε-greedy 漸減
    if epsilon > 0.1:
        epsilon -= 1.0/epochs

# 5. 繪製 Loss 曲線
plt.figure(figsize=(8,5))
plt.plot(losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Naive DQN Loss (static mode)")
plt.tight_layout()
plt.show()

# 6. 測試函式
def test_static(model, max_steps=20):
    game = Gridworld(size=4, mode='static')
    s_np = game.board.render_np().reshape(1,64)
    s = torch.from_numpy(s_np).float()
    for step in range(max_steps):
        q = model(s)
        a = torch.argmax(q).item()
        game.makeMove(action_set[a])
        print(f"Step {step:2d}: move {action_set[a]}")
        print(game.display())
        r = game.reward()
        if abs(r)==10:
            print("→ 遊戲結束，Reward =", r)
            return r>0
        s = torch.from_numpy(game.board.render_np().reshape(1,64)).float()
    print("→ 超過最大步數，視為失敗")
    return False

print("Static mode 測試結果：", test_static(model))
