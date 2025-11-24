import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Custom callback to log rewards
class RewardLogger(BaseCallback):
    def __init__(self, log_path, verbose=1):
        super().__init__(verbose)
        self.log_path = log_path
        self.rewards = []

    def _on_step(self) -> bool:
        if 'episode' in self.locals['infos'][0]:
            ep_reward = self.locals['infos'][0]['episode']['r']
            self.rewards.append(ep_reward)
            np.save(self.log_path, self.rewards)
        return True

# Create environment
env = gym.make("BipedalWalker-v3")
env = Monitor(env)

# PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=64,
    learning_rate=0.0003,
    n_steps=2048,
    gamma=0.99,
    device="cuda"
)

# Train
logger = RewardLogger("training_rewards.npy")
model.learn(total_timesteps=500_000, callback=logger)

# Save model
model.save("ppo_bipedalwalker")

env.close()