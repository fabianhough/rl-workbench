
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.distributions import Categorical

from .base import Agent
from ..model import SimpleLinearNet



class AgentPPO(Agent):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        policy_hidden_dims: list,
        value_hidden_dims: list,
        env,
        device='cpu',
        activation=nn.Tanh,
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        critic_coeff=0.5,
        entropy_coeff=0.05
    ):
        
        self._device = device
        self._env = env

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff

        self.policy_net = SimpleLinearNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=policy_hidden_dims,
            activation=activation
        )
        self.value_net = SimpleLinearNet(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=value_hidden_dims,
            activation=activation
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

    def policy(self, observs):
        # Unnormalized log probabilities
        logits = self.policy_net(observs)

        # Includes Softmax using log-sum-exp
        return Categorical(logits=logits)

    def value(self, observs):
        return self.value_net(observs)

    def act(self, observ, **kwargs):
        self.policy_net.eval()
        self.value_net.eval()
        with torch.no_grad():
            # Prepare observ
            observ_tensor = torch.tensor(observ, dtype=torch.float32).to(self._device)

            # Sample an action
            action = self.policy(observ_tensor.unsqueeze(0)).sample().item()

            # Estimate a value
            value = self.value(observ_tensor.unsqueeze(0)).item()
        self.policy_net.train()
        self.value_net.train()
        return action

    def train(self, sample):
        # Unwrapping sample
        observs, actions, rewards, next_observs, dones, values = sample

        # Preparing tensors
        observs = torch.tensor(observs).to(self._device)
        actions = torch.tensor(actions).to(self._device)
        rewards = torch.tensor(rewards).to(self._device)
        next_observs = torch.tensor(next_observs).to(self._device)
        dones = torch.tensor(dones).to(self._device)
        values = torch.tensor(values).to(self._device)

        # Preparing optimizer
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        # GAE Calculation


        # Back-propagation
        loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()

        # Returning loss metrics
        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

    def post_episode(self): ...



