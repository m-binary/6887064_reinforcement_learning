# Autonomous Highway Driving with Double Deep Q-Learning
| **Author**        | Max Spannagel                                     |
|-------------------|---------------------------------------------------|
| **Student ID**    | 6887064                                           |
| **Course**        | Reinforcement Learning                            |
| **Environment**   | Highway-v0 (highway-env)                          |
| **Algorithm**     | Double Deep Q-Network (DDQN)                      |

The reinforcement learning project implements a Double Deep Q-Network (DDQN) for autonomous highway driving simulation using OpenAI Gymnasium and Highway-Env.

## Overview

This project demonstrates the application of Deep Reinforcement Learning to autonomous driving in a highway environment. The agent learns to navigate traffic, change lanes safely, and maintain high speeds while avoiding collisions through trial and error interactions with the simulated environment.

## Features

- The project uses **Double Deep Q-Learning** to reduce overestimation bias compared to standard DQN.
- **Experience Replay** enables efficient learning from stored experiences.
- A balanced exploration vs exploitation strategy is achieved through **Epsilon-Greedy Exploration**.
- Stable training is ensured by **Target Network Updates** with periodic synchronization.
- The **Hyperparameter Configuration** allows easy experimentation with different settings.

## Environment

The project uses the `highway-env` environment, which simulates realistic highway driving scenarios:

- **State Space**: Kinematic features of nearby vehicles (position, velocity)
- **Action Space**: Discrete actions (lane changes, acceleration/deceleration)
- **Reward Function**: 
  - Collision penalty: -3.0
  - High speed reward: +1.0
  - Right lane preference: +0.1
  - Lane change cost: -0.3

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, but recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/mbinary/6887064_reinforcement_learning.git
cd 6887064_reinforcement_learning
```

2. **Install dependencies using uv (recommended)**
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Dependencies
- `gymnasium>=1.2.0` - Reinforcement learning environment interface
- `highway-env>=1.10.1` - Highway driving simulation environment
- `torch>=2.7.1` - PyTorch for neural networks
- `numpy>=2.3.1` - Numerical computations
- `matplotlib` - Visualization and plotting
- `tqdm>=4.67.1` - Progress bars
- `ipykernel>=6.29.5` - Jupyter notebook support

## Usage

### Training

Open the `training.ipynb` notebook and run all cells to start training:

```bash
jupyter notebook training.ipynb
```

### Testing

After training, test the agent's performance:
- Load the latest or a specific model version
- Run test episodes without exploration
- Evaluate average performance


## References

- [Highway-Env Documentation](https://github.com/Farama-Foundation/HighwayEnv)
- [Gymnasium Documentation](https://gymnasium.farama.org)