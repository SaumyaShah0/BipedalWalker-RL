# 🤖 BipedalWalker-v3 Reinforcement Learning Project

A Reinforcement Learning project using **Proximal Policy Optimization (PPO)** algorithm to train an agent to walk in the **BipedalWalker-v3** environment (from Gymnasium / Box2D physics engine).

This project demonstrates:
- Gymnasium + Box2D physics simulation
- PPO training & testing pipeline using Stable-Baselines3
- GPU acceleration (CUDA PyTorch)
- Reward logging and plotting
- Cross-platform setup (Windows/Linux)
- Clean and reproducible project structure

---

## ⚙️ Tech Stack

| Component | Version |
|------------|----------|
| **Python** | 3.10.19 |
| **Gymnasium** | Latest |
| **Stable-Baselines3** | Latest |
| **PyTorch** | CUDA 12.1 (for GPU) / CPU fallback |
| **OS** | Windows 10/11 / Zorin OS 18 (Linux) |
| **GPU** | NVIDIA RTX 3050 Laptop GPU |

---

## 📂 Project Structure

BipedalWalker-RL/  
│  
├── train.py                # Main training script  
├── test.py                 # Testing and rendering trained model  
├── plot_training.py        # Plot training rewards  
├── check_gpu.py            # Check CUDA/GPU availability  
│  
├── requirements.txt        # Dependencies list  
├── training_rewards.npy    # Stored reward data (auto-generated)  
├── models/                 # Saved trained models  
├── results/                # Optional log/plot directory  
├── .gitignore              # Git ignore file  
└── README.md               # Project documentation  

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
https://github.com/SaumyaShah0/BipedalWalker-RL/


### 2️⃣ Create Virtual Environment

**Windows:**
python -m venv walker_env
walker_env\Scripts\activate


### 3️⃣ Install Dependencies
pip install --upgrade pip
pip install gymnasium[box2d] stable-baselines3 torch matplotlib pygame


---

## 🧠 Training the Model

Run:
python train.py


This will:
- Create the BipedalWalker-v3 environment
- Train PPO for 500,000 timesteps
- Save rewards to training_rewards.npy
- Export trained model as ppo_bipedalwalker.zip in /models/

---

## 🎮 Testing the Trained Model

Run:
python test.py

This:
- Loads the trained model
- Opens the simulation window (2D graphical)
- Lets the agent walk in the terrain

---

## 🧩 PPO Algorithm Overview

**Full Form:** Proximal Policy Optimization  
PPO is a policy gradient method that balances exploration and stability by limiting updates to the policy function.  
It is widely used for continuous control problems such as robotics.

**🔍 Why PPO?**
- Stable updates (clip function prevents large policy changes)
- Works with both discrete & continuous actions
- Efficient for high-dimensional environments (like BipedalWalker)
- Easier to tune than TRPO or A2C

**⚔️ Comparison**

| Algorithm | Full Form                         | Key Feature                  | Suitable For            | Comparison                                   |
|-----------|-----------------------------------|------------------------------|------------------------|-----------------------------------------------|
| PPO       | Proximal Policy Optimization      | Stable policy updates        | Continuous/Discrete    | ✅ Best balance of stability & performance    |
| A2C       | Advantage Actor-Critic            | Synchronous actor-critic     | Simpler problems       | ❌ Less stable for long training              |
| DDPG      | Deep Deterministic Policy Gradient| Continuous deterministic control| Robotics arms      | ⚠️ Unstable w/o tuning                       |
| SAC       | Soft Actor-Critic                 | Entropy-regularized exploration| Continuous tasks    | ⚡ Fast but heavier compute cost              |

---

## 🧱 Hard Mode

To switch to hard terrain (with pits, stumps, and gaps):

In train.py and test.py:
env = gym.make("BipedalWalkerHardcore-v3")


---

## ⚙️ Performance

| Mode     | Hardware         | Training Time (500k Steps) |
|----------|------------------|---------------------------|
| Normal   | CPU              | ~50–60 minutes            |
| Normal   | GPU (RTX 3050)   | ~20–30 minutes            |
| Hardcore | GPU              | 2–4 hours                 |

---

## 🧮 GPU / CPU Usage

You can check CUDA or GPU availability using check_gpu.py:
import torch

print("Torch version:", torch.version)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
print("GPU Name:", torch.cuda.get_device_name(0))
else:
print("Running on CPU only")

**🔧 Run on GPU**

If your system has CUDA and PyTorch with GPU support installed:

- It will automatically use GPU.
- No code changes needed.

If not detected:
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


---

## 📊 Plot Training Rewards

Run:
python plot_training.py

Example code:
import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("training_rewards.npy")
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("PPO on BipedalWalker-v3")
plt.grid(True)
plt.show()


## 📚 References

- Gymnasium Documentation
- Stable-Baselines3 Docs
- PyTorch CUDA Install Guide
- PyLessons PPO Tutorial

## 🏁 Summary

This project trains a Bipedal robot to walk using PPO (Proximal Policy Optimization) with GPU acceleration.  
It demonstrates continuous control learning, policy gradient algorithms, and environment simulation — a foundational RL project for robotics and AI.
