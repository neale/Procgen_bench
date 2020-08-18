# DeepRL (procgen edition)


Modularized implementation of popular deep RL algorithms in PyTorch.  
Easy switch between toy tasks and challenging games.

Implemented algorithms:
* (Double/Dueling/Prioritized) Deep Q-Learning (DQN)
* Categorical DQN (C51)
* Quantile Regression DQN (QR-DQN)
* (Continuous/Discrete) Synchronous Advantage Actor Critic (A2C)
* Synchronous N-Step Q-Learning (N-Step DQN)
* Deep Deterministic Policy Gradient (DDPG)
* Proximal Policy Optimization (PPO)
* The Option-Critic Architecture (OC)
* Twined Delayed DDPG (TD3)

The DQN agent, as well as C51 and QR-DQN, has an asynchronous actor for data generation and an asynchronous replay buffer for transferring data to GPU.
Using 1 RTX 2080 Ti and 3 threads, the DQN agent runs for 10M steps (40M frames, 2.5M gradient updates) for Breakout within 6 hours.

# Dependency
* PyTorch v1.5.1

# Usage

```
pip install -r requirements.txt
```


```examples.py``` contains examples for all the implemented algorithms.  
```Dockerfile``` contains the environment for generating the curves below.  

# Credit

DeepRL framework orginally authored by: 

```
@misc{deeprl,
  author = {Zhang, Shangtong},
  title = {Modularized Implementation of Deep RL Algorithms in PyTorch},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/ShangtongZhang/DeepRL}},
}
```
