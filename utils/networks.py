import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.in_fn = nn.BatchNorm1d(input_dim)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))

        return out

class DoubleMLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(DoubleMLPNetwork, self).__init__()

        self.in_fn = nn.BatchNorm1d(input_dim)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.fc4 = nn.Linear(input_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, out_dim)

        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.fc6.weight.data.uniform_(-3e-3, 3e-3)

            self.out_fn = torch.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        norm_in_X = self.in_fn(X)

        h1 = self.nonlin(self.fc1(norm_in_X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))

        h1_2 = self.nonlin(self.fc4(norm_in_X))
        h2_2 = self.nonlin(self.fc5(h1_2))
        out_2 = self.out_fn(self.fc6(h2_2))

        return out, out_2

    def Q1(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        norm_in_X = self.in_fn(X)

        h1 = self.nonlin(self.fc1(norm_in_X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))

        return out
    



# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

from .diff_helpers import SinusoidalPosEmb


class DiffMLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(DiffMLP, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim   # state + action + time emb
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)
        self.device = device

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        t = t.to(state.device)
        x = x.to(state.device)

        x = torch.cat([x, t, state], dim=1)   # 这里的t应该是denoising step的 X_t，epsilon_t
        x = self.mid_layer(x)

        return self.final_layer(x)



