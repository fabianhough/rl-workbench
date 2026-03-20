
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

from .base import Agent
from ..model import SimpleLinearNet



class AgentA2C(SimpleLinearNet):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        policy_hidden_dims: list,
        value_hidden_dims: list,
        activation=nn.ReLU,
        device='cpu',
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99
    ):
        
        self._device = device
        self._env = env

        self.policy_net = SimpleLinearNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=policy_hidden_dims
        )
        self.value_net = SimpleLinearNet(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=value_hidden_dims
        )

        self.policy_net.to(device)
        self.value_net.to(device)

        self.policy_optimizer = Adam(
            params=self.policy_net.parameters(),
            lr=policy_lr
        )
        self.value_optimizer = Adam(
            params=self.value_net.parameters(),
            lr=value_lr
        )



    def act(self, observ, **kwargs): ...


    def train(self, sample): ...


    def post_episode(self): ...



