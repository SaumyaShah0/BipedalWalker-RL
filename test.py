import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("BipedalWalker-v3", render_mode="human")

model = PPO.load("ppo_bipedalwalker", device="cuda")

obs, info = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()