import numpy as np
import matplotlib.pyplot as plt
def softmax(Q, tau):
    exp_Q = np.exp(Q / tau)
    return exp_Q / np.sum(exp_Q)

def softmax_bandit(k=10, steps=1000, tau=0.5):
    true_rewards = np.random.normal(0, 1, k)
    Q = np.zeros(k)
    N = np.zeros(k)
    rewards = []
    actions = []

    for _ in range(steps):
        probs = softmax(Q, tau)
        a = np.random.choice(k, p=probs)
        r = np.random.normal(true_rewards[a], 1)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards.append(r)
        actions.append(a)

    return rewards, actions

rewards, actions = softmax_bandit()
avg_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

plt.plot(avg_rewards)
plt.title("Softmax: Average Reward")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.show()

plt.hist(actions, bins=np.arange(11)-0.5, edgecolor='black')
plt.title("Softmax: Action Count")
plt.xlabel("Action")
plt.ylabel("Count")
plt.show()
