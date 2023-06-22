# Learning2Learning (L2L)

<p align="center"><img src="https://github.com/IureRosa/learningVis/blob/main/images/logo_vector.png" width="650px"/></p>
 <!-- 
 <p align="center"><img src="https://github.com/IureRosa/learningVis/blob/main/images/homepagev2.png" width="650px"/></p> 
 -->

### Abstract

The L2L project aims to develop an interactive web app that allows users to explore and understand the fundamental concepts of reinforcement learning applied to autonomous navigation. The web app will be able to run different reinforcement learning algorithms and will provide a friendly interface so that users can interact with the code, manipulate the learning parameters and observe how these changes affect the results obtained. Data visualization will play a key role in presenting the results, allowing users to understand the importance of parameters in the learning process.

## Folder Structure
~~~
.
├── images   # app related images
├── agents    # RL agents and environment for the proposed project
├── results    # (Preliminary)
├── LICENSE
├── .gitignore
└── README.md
~~~

## Methods

The training environments are OpenAI Gym's own. In this project, environments with continuous and discrete action space are used, so that the use of more developed algorithms is necessary.

For discrete and simple environments like FrozenLake, tabular algorithms like SARSA and Q-Learning are efficient. However, as the complexity of the environment increases, DRL techniques become necessary to efficiently solve the proposed environments.

In addition, some OpenAI Gym environments, such as Pendulum, have a continuous action space, making it necessary to use algorithms such as DQN or PPO to solve such problems, since the Actor Critic algorithms (TD3, DDPG and SAC) are exclusive to discrete action spaces.

In the agents folder it is possible to find the codes used in this project. We encourage users to create and test their own training functions. This will mean that, in addition to understanding the relationship between the parameters and the results obtained, the user will have a dimension of how efficient the agent developed by him is.

## Dependencies
- [OpenAI Gym](https://github.com/openai/gym)==0.21.0 or newer
- Python==3.7.12 or newer
- (Optional) [Stable Baselines3](https://stable-baselines.readthedocs.io/en/master/index.html#)==1.3.0 or newer

