# 🏃‍♂️ BipedalWalker-v3 Reinforcement Learning using PPO (Stable-Baselines3)

This project trains a Bipedal Walker agent using **Proximal Policy Optimization (PPO)** in the **Gymnasium BipedalWalker-v3** environment.  
The agent learns stable locomotion using neural network policies, trained on **Windows 11** with support for both **CPU** and **GPU (NVIDIA RTX)**.

---

# 📌 Project Features
- PPO policy-gradient algorithm  
- Gymnasium + Box2D physics simulation  
- Training & testing pipeline  
- GPU acceleration (CUDA PyTorch)  
- Reward logging and plotting  
- Clean and reproducible project structure  

---

# 🧩 Tech Stack

| Component | Version |
|----------|---------|
| Python | 3.10.x |
| Gymnasium | Latest |
| Stable-Baselines3 | Latest |
| PyTorch | CPU or CUDA 12.1 |
| OS | Windows 10/11 |
| GPU | NVIDIA RTX 3050 Laptop GPU |

---

# 📁 Project Structure

```
BipedalWalker-RL/
│── train.py
│── test.py
│── plot_training.py
│── check_gpu.py
│── requirements.txt
│── .gitignore
│── training_rewards.npy
│── models/                (optional)
│── results/               (optional)
```

---

# 🛠 Installation

## 1️⃣ Clone the Repository
```
git clone https://github.com/YOUR_USERNAME/BipedalWalker-RL.git  
cd BipedalWalker-RL
```

## 2️⃣ Create Virtual Environment  
```
python -m venv venv
```

Activate:

**Windows**  
```
venv\Scripts\activate
```

**Linux**  
```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies (CPU)
```
pip install -r requirements.txt
```

---

# ⚡ GPU Setup (CUDA)

## Step 1 — Remove CPU PyTorch  
```
pip uninstall torch torchvision torchaudio -y
```

## Step 2 — Install CUDA PyTorch 12.1  
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 3 — Check GPU 
```
python check_gpu.py
```

Expected:

Torch version: x.x.x  
CUDA available: True  
GPU Name: NVIDIA RTX 3050

---

# 🚀 Running the Project

## ⭐ Train Agent
```
python train.py
```

To force GPU:
```
model = PPO("MlpPolicy", env, device="cuda")
```

## ⭐ Test Agent
```
python test.py
```

## ⭐ Plot Rewards  
```
python plot_training.py
```

---

# 🤖 Algorithm Used — PPO

**Proximal Policy Optimization (PPO)**  
- Stable policy updates  
- Works well on continuous control  
- Less sensitive to hyperparameters  
- Best choice for Bipedal Walker tasks  

---

# 🌍 Gymnasium Environments

### Normal Mode  
```
gym.make("BipedalWalker-v3")
```

### Hard Mode  
```
gym.make("BipedalWalkerHardcore-v3")
```

---

# 📈 Performance (Approx)

| Mode | Hardware | Time (500k steps) |
|------|----------|-------------------|
| Normal | CPU | ~50–60 mins |
| Normal | GPU | ~20–30 mins |
| Hardcore | GPU | 2–4 hours |

---

# 🧪 check_gpu.py
```
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU only")
```
---

# 📦 requirements.txt

- gymnasium[box2d]  
- stable-baselines3  
- pygame  
- matplotlib  
- numpy  

---

# 🧹 .gitignore  
- venv/  
- __pycache__/  
- *.npy  
- *.npz  
- *.zip  
- *.pt  
- *.pth  
- *.log  
- .DS_Store  

---

# 🙌 Credits  
- Gymnasium  
- Stable-Baselines3  
- PyTorch  
- Box2D  

---

# 🏁 Summary  
This repository contains a full PPO training pipeline for BipedalWalker-v3 with CPU & GPU support, clean structure, reward logging, and complete reproducibility.
