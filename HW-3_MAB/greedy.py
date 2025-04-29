import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy(k=10, steps=1000, epsilon=0.1):
    true_rewards = np.random.normal(0, 1, k)
    Q = np.zeros(k)
    N = np.zeros(k)
    rewards = []
    actions = []

    for t in range(steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(k)
        else:
            a = np.argmax(Q)
        r = np.random.normal(true_rewards[a], 1)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards.append(r)
        actions.append(a)

    return rewards, actions

rewards, actions = epsilon_greedy()
average_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

plt.plot(average_rewards)
plt.title("Epsilon-Greedy: Average Reward")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.show()

plt.hist(actions, bins=np.arange(11)-0.5, edgecolor='black')
plt.title("Epsilon-Greedy: Action Count")
plt.xlabel("Action")
plt.ylabel("Count")
plt.show()
