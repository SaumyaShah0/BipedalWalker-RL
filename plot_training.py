import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("training_rewards.npy")

plt.figure(figsize=(10,5))
plt.plot(rewards, label="Episode Reward")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Training Performance of PPO on BipedalWalker-v3")
plt.legend()
plt.grid(True)
plt.show()