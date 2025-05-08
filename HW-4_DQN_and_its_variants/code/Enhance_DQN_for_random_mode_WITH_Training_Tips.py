#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dqn_keras_random_fixed.py

在 Gridworld random 模式下訓練 DQN (Keras 實作)，
並加入 gradient clipping 與 learning rate scheduling。
修正了输入/输出维度，确保正确索引 Q 值。
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from Gridworld import Gridworld

# === 1. 超參數 ===
STATE_DIM    = 64
ACTION_DIM   = 4
GAMMA        = 0.9
EPS_START    = 1.0
EPS_END      = 0.1
EPS_DECAY    = 1e-3           # 每 step 減少 ε
LR_INIT      = 1e-3
MEM_SIZE     = 10000
BATCH_SIZE   = 128
EPISODES     = 2000
MAX_STEPS    = 50

# Learning rate schedule: 指數衰減
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR_INIT,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    clipnorm=1.0              # gradient clipping
)
mse_loss = tf.keras.losses.MeanSquaredError()

# 動作對照
action_map = {0:'u', 1:'d', 2:'l', 3:'r'}

# === 2. 定義 Q 網路 ===
class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(150, activation='relu')
        self.fc2 = tf.keras.layers.Dense(100, activation='relu')
        self.out = tf.keras.layers.Dense(ACTION_DIM)

    def call(self, x):
        # x: [batch, 64]
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

# === 3. Experience Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, s, a, r, s2, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        # store state without extra batch dimension
        self.buffer.append((s.squeeze(), a, r, s2.squeeze(), done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s.astype(np.float32), a, r.astype(np.float32), s2.astype(np.float32), d.astype(np.float32)

    def __len__(self):
        return len(self.buffer)

# === 4. 主訓練迴圈 ===
agent = QNetwork()
replay = ReplayBuffer(MEM_SIZE)
eps = EPS_START

# 用於記錄
all_rewards = []
all_losses  = []

for ep in range(1, EPISODES+1):
    game = Gridworld(size=4, mode='random')
    state = game.board.render_np().reshape(STATE_DIM).astype(np.float32)
    total_r = 0

    for step in range(MAX_STEPS):
        # ε-greedy
        if random.random() < eps:
            a = random.randint(0, ACTION_DIM-1)
        else:
            qvals = agent(state[None, :])  # [1,4]
            a = int(tf.argmax(qvals[0]).numpy())

        game.makeMove(action_map[a])
        r = game.reward()
        next_state = game.board.render_np().reshape(STATE_DIM).astype(np.float32)
        done = abs(r)==10

        # 存入回放
        replay.push(state, a, r, next_state, done)
        state = next_state
        total_r += r

        # 只有 buffer 滿了才開始更新
        if len(replay) >= BATCH_SIZE:
            s_batch, a_batch, r_batch, s2_batch, d_batch = replay.sample(BATCH_SIZE)

            with tf.GradientTape() as tape:
                # 預測 Q(s,a)
                q_pred = agent(s_batch)                   # [B,4]
                idx    = tf.stack([tf.range(BATCH_SIZE, dtype=tf.int32), a_batch.astype(np.int32)], axis=1)
                q_sa   = tf.gather_nd(q_pred, idx)       # [B,]

                # 計算 target Q
                q_next = tf.reduce_max(agent(s2_batch), axis=1)  # [B,]
                q_target = r_batch + GAMMA * q_next * (1 - d_batch)

                loss = mse_loss(q_target, q_sa)

            grads = tape.gradient(loss, agent.trainable_variables)
            optimizer.apply_gradients(zip(grads, agent.trainable_variables))
            all_losses.append(float(loss.numpy()))

        if done:
            break

    # ε 緩減
    eps = max(EPS_END, eps - EPS_DECAY)
    all_rewards.append(total_r)

    if ep % 100 == 0:
        print(f"Episode {ep:4d}, Reward: {total_r:.1f}, ε: {eps:.3f}")

# === 5. 畫圖 ===
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(all_rewards)
plt.xlabel("Episode"); plt.ylabel("Total Reward")
plt.title("DQN (Keras) on Random Mode")

plt.subplot(1,2,2)
plt.plot(all_losses)
plt.xlabel("Training Steps"); plt.ylabel("Loss")
plt.title("Loss Curve")
plt.tight_layout()
plt.show()

# === 6. 測試函式 ===
def test_random(model, games=100):
    wins = 0
    for _ in range(games):
        game = Gridworld(size=4, mode='random')
        s = game.board.render_np().reshape(STATE_DIM).astype(np.float32)
        for _ in range(MAX_STEPS):
            q = model(s[None, :])
            a = int(tf.argmax(q[0]).numpy())
            game.makeMove(action_map[a])
            if abs(game.reward())==10:
                if game.reward()>0: wins+=1
                break
            s = game.board.render_np().reshape(STATE_DIM).astype(np.float32)
    print(f"Win rate over {games} games: {wins/games*100:.1f}%")

test_random(agent)
