## A Repo for Classical to State-of-the-art Deep Q-Network Algorithms
DQN is an algorithm that addresses the question : How do we make reinforcement learning look more like **Supervised learning** ? Common problems that consisitently show up in value-based reinforcement learning are:  
1. Data is not independent and identically distributed (The IID Assumption )
2. Non-stationarity of targets 

It's obvious that we needed to make the neccessary tweaks in the algorithms to overcome these problems i.e to make the data look more IID and the targets fixed.

**Solutions** (which form the part of the DQN's we see today):
> In order to make the target values more stationary we can have a separate network that we can fix for multiple steps and reserve it for calculating more stationary targets i.e making use of a *Target network*

> Use "replay" of already seen experiences (Experience Replay) , often referred to as the replay buffer or a replay memory and holds experience samples for several steps, allowing the sampling of mini-batches from a broad-set of past experiences.

  <p align="center">DQN with Replay memory <p align="center">
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/DQN.png" width="450px" height="300px"/></p>

## Papers associated with Novel-Algorithms
### Playing Atari with Deep Reinforcement Learning [[Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/DQN_image1.png" width="650px" height="200px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/DQN_algo.png" width="650px" height="300px"/></p>

### Deep Reinforcement Learning with Double Q-learning [[Paper](https://arxiv.org/abs/1509.06461)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/DDQN1.png" width="450px" height="100px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/DDQN2.png" width="450px" height="100px"/></p>

### Dueling Network Architectures for Deep Reinforcement Learning [[Paper](https://arxiv.org/abs/1511.06581)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Dueling1.png" width="450px" height="100px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Dueling2.png" width="450px" height="100px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Dueling3.png" width="450px" height="100px"/></p>

### Prioritized Experience Replay [[Paper](https://arxiv.org/abs/1511.05952)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/PER1.png" width="550px" height="400px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/PER_compare.png" width="450px" height="100px"/></p>

### Noisy Networks for Exploration [[Paper](https://arxiv.org/abs/1706.10295)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Noisy_loss.png" width="550px" height="200px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Noisy_compare.png" width="450px" height="150px"/></p>

### A Distributional Perspective on Reinforcement Learning (Categorical DQN) [[Paper](https://arxiv.org/pdf/1707.06887.pdf)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Categorical.png" width="550px" height="350px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Categorical2.png" width="450px" height="350px"/></p>
### Rainbow: Combining Improvements in Deep Reinforcement Learning [[Paper](https://arxiv.org/abs/1710.02298)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Rainbow1.png" width="550px" height="350px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/Rainbow_compare.png" width="450px" height="300px"/></p>
### Distributional Reinforcement Learning with Quantile Regression[[Paper](https://arxiv.org/pdf/1710.10044.pdf)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/QR1.png" width="550px" height="350px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/QR2.png" width="450px" height="350px"/></p>

### Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation [[Paper](https://arxiv.org/abs/1604.06057)]
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/HR1.png" width="600px" height="400px"/></p>
<p align="center"><img src="https://github.com/ashutoshtiwari13/DeepQNetwork-Hub/blob/master/images/HR2.png" width="450px" height="150px"/></p>


