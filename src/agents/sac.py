

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from .base import Agent
from ..model import SimpleLinearNet


LOG_STD_MIN = -5
LOG_STD_MAX = 2


class AgentSAC(Agent):
    def __init__(self,
        state_dim: int,
        action_dim: int,
        actor_hidden_dims: list,
        q_hidden_dims: list,
        actor_output_dim: int,
        action_gain: float,
        action_bias: float,
        env,
        device,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        q_lr=1e-3
    ):
        self._device = device
        self._env = env

        self.gamma = gamma
        self.tau = tau

        # Action Scaling
        self.action_gain = action_gain
        self.action_bias = action_bias

        # Actor Net
        self.actorNet = SimpleLinearNet(
            input_dim=state_dim,
            output_dim=actor_output_dim,
            hidden_dims=actor_hidden_dims,
            activation=nn.ReLU
        )
        self.actor_mean = nn.Linear(actor_output_dim, action_dim)
        self.actor_logstd = nn.Linear(actor_output_dim, action_dim)

        # Q Nets
        self.qNet0 = SimpleLinearNet(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_dims=q_hidden_dims,
            activation=nn.ReLU
        )
        self.qNet1 = SimpleLinearNet(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_dims=q_hidden_dims,
            activation=nn.ReLU
        )

        # Q Targets
        self.qTgt0 = SimpleLinearNet(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_dims=q_hidden_dims,
            activation=nn.ReLU
        )
        self.qTgt1 = SimpleLinearNet(
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_dims=q_hidden_dims,
            activation=nn.ReLU
        )

        # Moving onto device
        self.actorNet.to(device)
        self.qNet0.to(device)
        self.qNet1.to(device)
        self.qTgt0.to(device)
        self.qTgt1.to(device)

        # Copying Q Net weights
        self.qTgt0.load_state_dict(self.qNet0.state_dict())
        self.qTgt1.load_state_dict(self.qNet1.state_dict())

        # Optimizers
        self.actor_optim = Adam(
            params=self.actorNet.parameters(),
            lr=actor_lr
        )
        self.q_optim = Adam(
            params=list(self.qNet0.parameters()) + list(self.qNet1.parameters()),
            lr=q_lr
        )

    def actor_forward(self, x):
        # Using cleanRL implementation
        x = self.actorNet(x)
        # Distributing to mean and log_std
        mean = self.actor_mean(x)
        log_std = self.actor_logstd(x)
        # Constraining log_std between [-1,1]
        log_std = torch.tanh(log_std)
        # Scaling between [-5,2]
        # 'log_std + 1' moves between [0,2], '0.5 *' moves it [0,1]; rest is scaling to [-5,2]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def act(self, observ):
        self.actorNet.eval()
        with torch.no_grad():
            # Prepare observ
            observ_tensor = torch.tensor(observ, dtype=torch.float32).to(self._device)

            # Sample an action
            mean, log_std = self.actor_forward(observ_tensor.unsqueeze(0))
            std = log_std.exp()
            normal = Normal(mean, std)
            # Only using sample here as we are only interestedin the action and not the gradient
            action = torch.tanh(normal.sample()) * self.action_gain + self.action_bias
            action = action.item()
        self.actorNet.train()
        return action, None


    def train(self, sample):
        pass

    def post_episode(self):
        pass