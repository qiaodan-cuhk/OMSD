import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'
    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim, layers=2):
        super().__init__()
        dims = [state_dim + action_dim] +[256]*layers +[1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))


class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)


class Dirac_Policy(nn.Module):
    def __init__(self, action_dim, state_dim, layer=2):
        super().__init__()
        self.net = mlp([state_dim] + [256]*layer + [action_dim], output_activation=nn.Tanh)

    def forward(self, state):
        return self.net(state)
    
    def select_actions(self, state):
        return self(state)


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act  # F.relu()
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)
            # self.layer_norm = nn.BatchNorm1d(features)

        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0.0 else None

    def forward(self, x, training=False):
        # dropout -> layer norm -> 4x -> activate -> x | residual(x_old) + x(x_new) 
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x

class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, out_dim, dropout_rate=None, use_layer_norm=False, hidden_dim=512, activations=F.relu):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks  # default 3
        self.out_dim = out_dim   # action dim
        self.dropout_rate = dropout_rate   # 0.1
        self.use_layer_norm = use_layer_norm   # True
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim, self.hidden_dim)  # 老算法是inputdim+128, 128 是 t embed，现在改成新的直接在idql里修改维度

        self.blocks = nn.ModuleList([MLPResNetBlock(self.hidden_dim, self.activations, self.dropout_rate, self.use_layer_norm)
                                     for _ in range(self.num_blocks)])   # UNet blocks 

        self.out_fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, training=False):
        # x -> hidden dim -> [ hidden ->  hidden ->  hidden ](3 resnet x+res_x) -> activate -> out dim
        x = self.fc(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.activations(x)
        x = self.out_fc(x)

        return x
    
    
class ScoreNet_IDQL(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=None, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim))
        self.device=args.device
        self.marginal_prob_std = marginal_prob_std
        self.args=args

        t_embed_dim = args.t_embed_dims  # 64
        xa_embed_dim = args.sa_embed_dims  # 32

        # [32, 64, 64]  && [3+7, 32]
        self.cond_model = mlp([embed_dim, t_embed_dim, t_embed_dim], output_activation=None, activation=nn.ReLU)
        self.input_embed = mlp([input_dim, xa_embed_dim], output_activation=None, activation=nn.ReLU)

        self.new_input_dim = xa_embed_dim + t_embed_dim   # 32 + 64  = embed(s+a) + embed(t)
        self.main = MLPResNet(args.actor_blocks, self.new_input_dim, output_dim, dropout_rate=0.1, use_layer_norm=True, hidden_dim=args.resnet_hidden_dim, activations=nn.ReLU())

        # pytorch2.2 can use nn.Mish, similar to Relu
        # https://pytorch.org/docs/stable/generated/torch.nn.Mish.html

        # The swish activation function
        # self.act = lambda x: x * torch.sigmoid(x)
        
    # (perturbed_x, random_t, all_s)
    def forward(self, x, t, condition):
        # self.embed = t * self.W * 2\pi, input t output = embedding dim 32
        # cond_model = input 32 -> 64 -> 64
        embed = self.cond_model(self.embed(t))  
        input_xa = torch.cat([x, condition], dim=-1)
        embed_xa = self.input_embed(input_xa)
        all = torch.cat([embed_xa, embed], dim=-1)  # dim = 32+64

        # all = torch.cat([x, condition, embed], dim=-1)  # pertubed x, all_s, t_embed
        h = self.main(all)
        return h
