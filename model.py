import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim, seed=0) -> None:
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.linear(2*state_dim, hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])

        self.fc2 = nn.linear(hidden_dim[0], hidden_dim[1])
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])

        self.fc3 = nn.linear(hidden_dim[-1], output_dim)

        self.activation = f.leaky_relu

        self.register_parameter()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))

        a = torch.tanh(self.fc3(x))
        return a

class Critic(nn.Moddule):
    def __init__(self, full_state_dim, full_action_dim, hidden_dim,
                 output_dim, seed=0, output_act=False):

            super(Critic, self).__init__()
            self.seed = torch.manual_seed(seed)

            self.fc1 = nn.linear(full_state_dim, hidden_dim[0])
            self.bn1 = nn.BatchNorm1d(hidden_dim[0])

            self.fc2 = nn.linear(hidden_dim[0]+full_action_dim, 
                                 hidden_dim[1])
            self.bn2 = nn.BatchNorm1d(hidden_dim[1])

            self.fc3 = nn.linear(hidden_dim[-1], output_dim)

            self.activation = f.leaky_relu
            self.out_act = output_act
            
            self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, actions):
        h1 = self.activation(self.bn1(self.fc1(state)))
        concated = torch.cat((h1, actions), dim=-1)
        h2 = self.activation(self.bn2(self.fc2(concated)))

        output = self.fc3(h2)
        return output

