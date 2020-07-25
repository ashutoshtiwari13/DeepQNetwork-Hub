## A Repo for Classical to State-of-the-art Deep Q-Network Algorithms
DQN is an algorithm that addresses the question : How do we make reinforcement learning look more like **Supervised learning** ? Common problems that consisitently show up in value-based reinforcement learning are:  
1. Data is not independent and identically distributed (The IID Assumption )
2. Non-stationarity of targets 

It's obvious that we needed to make the neccessary tweaks in the algorithms to overcome these problems i.e to make the data look more IID and the targets fixed.

**Solutions** (which form the part of the DQN's we see today):
> In order to make the target values more stationary we can have a separate network that we can fix for multiple steps and reserve it for calculating more stationary targets i.e making use of a *Target network*

> Use "replay" of already seen experiences (Experience Replay) , often referred to as the replay buffer or a replay memory and holds experience samples for several steps, allowing the sampling of mini-batches from a broad-set of past experiences.

## Papers associated with Novel-Algorithms
### Playing Atari with Deep Reinforcement Learning [[Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)]
"""TODO"""
### Deep Reinforcement Learning with Double Q-learning [[Paper](https://arxiv.org/abs/1509.06461)]
"""TODO"""
### Dueling Network Architectures for Deep Reinforcement Learning [[Paper](https://arxiv.org/abs/1511.06581)]
"""TODO"""
### Prioritized Experience Replay [[Paper](https://arxiv.org/abs/1511.05952)]
"""TODO"""
### Noisy Networks for Exploration [[Paper](https://arxiv.org/abs/1706.10295)]
"""TODO"""
### A Distributional Perspective on Reinforcement Learning [[Paper](https://arxiv.org/pdf/1707.06887.pdf)]
"""TODO"""
### Rainbow: Combining Improvements in Deep Reinforcement Learning [[Paper](https://arxiv.org/abs/1710.02298)]
"""TODO"""
### Distributional Reinforcement Learning with Quantile Regression[[Paper](https://arxiv.org/pdf/1710.10044.pdf)]
"""TODO"""
### Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation [[Paper](https://arxiv.org/abs/1604.06057)]
"""TODO"""


