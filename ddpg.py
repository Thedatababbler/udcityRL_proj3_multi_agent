import torch
import numpy as np
from torch.optim import Adam
#from utilities import hard_update, toTorch
from collections import namedtuple, deque

from models import Actor, Critic
#from utilities import toTorch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents,
                 hidden_actor, hidden_critic, lr_actor, lr_critic,
                 buffer_size, agent_id, use_PER=False, seed=0) -> None:
        super(DDPGAgent， self).__init__()

        self.seed = torch.manual_seed(seed)

        self.actor_local = Actor(state_size, hidden_actor, action_size,
                                 seed=seed).to(device)
        self.actor_target = Actor(state_size, hidden_actor, action_size,
                                  seed=seed).to(device)
        self.critic_local = Critic(state_size, num_agents*action_size,
                                  hidden_critic, 1, seed=seed).to(device)
        self.critic_target = Critic(state_size, num_agents*action_size,
                                    hidden_critic, 1, seed=seed).to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(),
                                    lr= lr_actor)
        self.critic_optimizer = Adam(self.critic_local.parameters(),
                                     lr=lr_actor)
        
        # initialize targets same as original networks
        self.noise = OUNoise(out_actor, scale=1.0 )
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)


    def act(self, obs, noise):
        obs = obs.to(device)

        if len(obs.shape)==1: 
            obs = obs.unsqueeze(0)

        return self.actor_local(obs) + noise*self.noise.noise()

    def target_act(self, obs, noise):
        obs = obs.to(device)

        if len(obs.shape)==1:
            obs = obs.unsqueeze(0)
        
        return self.actor_target(obs) + noise*self.noise.noise()



