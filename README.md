# 🧠 PyTorch PPO & SAC Implementations for Continuous Control

![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## Overview

This repository contains clean, modular implementations of two foundational **Deep Reinforcement Learning** algorithms:  
- **Proximal Policy Optimization (PPO)**
- **Soft Actor-Critic (SAC)**

Both are implemented using **PyTorch** and are suited for **continuous action spaces**.

This repo is designed for:
- 🔬 Researchers
- 📚 Students
- 🧑‍💻 Developers building DRL pipelines

**Keywords**: `Soft Actor-Critic`, `PPO`, `Reinforcement Learning`, `PyTorch`, `DRL`, `RL`, `Actor Critic`, `Continuous Control`, `OpenAI Gym`, `Off-policy`, `On-policy`, `Deep RL`, `SOTA`

---

## ✨ Features

- 🧱 Modular structure with reusable components (Actor, Critic, Memory)
- 🧮 PPO implementation with GAE, clipping, entropy regularisation
- 🔁 SAC with automatic entropy tuning and twin Q-networks
- 🔍 Logging of rewards and action distributions
- ⚙️ Easy to integrate into any RL environment

---

## 📦 Installation

```bash
git clone https://github.com/your-username/ppo-sac-pytorch.git
cd ppo-sac-pytorch
pip install -r requirements.txt
