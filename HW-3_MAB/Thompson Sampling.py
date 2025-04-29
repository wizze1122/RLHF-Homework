import numpy as np
import matplotlib.pyplot as plt
def thompson_sampling(k=10, steps=1000):
    true_probs = np.random.rand(k)
    alpha = np.ones(k)
    beta_vals = np.ones(k)
    rewards = []
    actions = []

    for _ in range(steps):
        theta = np.random.beta(alpha, beta_vals)
        a = np.argmax(theta)
        reward = np.random.rand() < true_probs[a]
        alpha[a] += reward
        beta_vals[a] += 1 - reward
        rewards.append(reward)
        actions.append(a)

    return rewards, actions

rewards, actions = thompson_sampling()
avg_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

plt.plot(avg_rewards)
plt.title("Thompson Sampling: Average Reward")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.show()

plt.hist(actions, bins=np.arange(11)-0.5, edgecolor='black')
plt.title("Thompson Sampling: Action Count")
plt.xlabel("Action")
plt.ylabel("Count")
plt.show()
