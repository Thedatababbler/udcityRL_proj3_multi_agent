### Learning Algorithm
For this project, we encounter a new challenging problems in Reinforcement Learning, that is how to train multiple agents to coportate or compete in the same game. Though we've learned
many techniques and models by far, all of them are designed for a single agent learning tasks. So, I decide to choose MADDPG(Multi-Agent Deep Deterministic Policy Gradient) algorithm
we just learned in this chapter to complete the task.

Comparing to DDPG, MADDPG could be applied to multi-agent tasks like this project. The reason that the original DDPG is not suitable for multi-agents learning is beacuse the partial observation problem.
For a single agent in the multi-agent tasks, they usually cannot capture the full observation of the environments, that is to say the states no longer varied only due to the single agent's actions.
So, in this situation, we can no longer train a single agent by learning the state-value function. To solve this problem, MADDPG gives the critics a full insight of the state and the action values generated from other agents.
The critics observed the full interactions between each agent and the environments, and the observation from all the agents, so it can learn the state-value function correctly.
For each agent, we train a independent critics by providing the full state observation and all actions to it. So each agent can learn from the full observation. This the core differences
between DDPG and MADDPG.

However, we only provide full observation to the critics during the training stage. After the agents learned a efficient policy during the training, we will no longer provide full observation
to the critics during the testing stage, in fact, we don't even need the critics model during the testing stage.

## Loss functions
MADDPG is very much identical to the DDPG algorithm in terms of loss functions, except we need to feed the full observation and actions from all agents to the critics in MADDPG. 
Just like the DDPG, we need to train the critics and actors during the training. To do that, we need to train a critic module which can evaluate the expected reward return given an action.
The loss function for the critic module is the MSE loss of the TD error which equals to (y - Q(s,a)) where y = reward + discount * Q(s+1,a+1). To alleviate the over-confidence problem
in this method, we use a seperate network Q-target network to compute the Q(s,a) and Q(s+1, a+1) values from above. 

To train the actors, we only need to train them to maximize the critics value of their determined actions. Thus, the loss function is simple the negative critics value given the actions.

## Experience replay
We also need to implement a memory buffer to store all the past states, actions, rewards we've been through. And we sample from it to train our agents. We do this because we don't want
the correlation of consecutive states affect our training.

## OU Noise
To let our agent to explore different strategies at the begining stage, we need to add some random noise to the models. From what we've learned from the papper, the OU Noise is the best 
noise for our DDPG models.

### Results
After training more than 1500 episodes, our agents finally learned to solve this task. From this experince I've learned that tunning the initial state and decay rate of the OU Noise is
quite important in training a MADDPG model. A different initial state may lead to a different training length in this task. To let the agent to explore the stratgies is quite important 
for the training, so we can not decay the noise too fast. 

Here is the resualt graph:
