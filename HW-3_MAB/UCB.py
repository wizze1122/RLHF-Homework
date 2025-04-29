import numpy as np
import matplotlib.pyplot as plt
def ucb(k=10, steps=1000, c=2):
    true_rewards = np.random.normal(0, 1, k)
    Q = np.zeros(k)
    N = np.ones(k)
    rewards = []
    actions = []

    for t in range(1, steps+1):
        ucb_values = Q + c * np.sqrt(np.log(t) / N)
        a = np.argmax(ucb_values)
        r = np.random.normal(true_rewards[a], 1)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards.append(r)
        actions.append(a)

    return rewards, actions

rewards, actions = ucb()
avg_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

plt.plot(avg_rewards)
plt.title("UCB: Average Reward")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.show()

plt.hist(actions, bins=np.arange(11)-0.5, edgecolor='black')
plt.title("UCB: Action Count")
plt.xlabel("Action")
plt.ylabel("Count")
plt.show()
