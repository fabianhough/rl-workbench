
import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Categorical
from torch.optim import Adam

from .base import Agent
from ..model import SimpleLinearNet



def discounted_rewards_to_go(rewards, dones, gamma):
    returns = np.zeros_like(rewards)
    running_return = 0
    for i in reversed(range(len(rewards))):
        running_return = rewards[i] + gamma * (1 - dones[i]) * running_return
        returns[i] = running_return
    return returns


class AgentReinforce(Agent):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        activation=nn.ReLU,
        device='cpu',
        gamma=0.99,
        lr=3e-4
    ):
        '''
        '''
        
        self._device = device
        self.gamma = gamma

        self.net = SimpleLinearNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
        self.net.to(self._device)

        self.optimizer = Adam(
            params=self.net.parameters(),
            lr=lr
        )

    def policy(self, observs):
        # Unnormalized log probabilities
        logits = self.net.forward(observs)

        # Includes Softmax using log-sum-exp
        return Categorical(logits=logits)

    def act(self, observ, **kwargs):
        self.net.eval()
        with torch.no_grad():
            # Prepare observ
            observ_tensor = torch.tensor(observ, dtype=torch.float32).to(self._device)

            # Sample an action
            action = self.policy(observ_tensor.unsqueeze(0)).sample().item()
        self.net.train()
        return action

    def loss(self, observs, actions, weights):
        # Generating the log_probs
        log_probs = self.policy(observs).log_prob(actions)
        # Calculating the negative loss
        return -(log_probs * weights).mean()

    def train(self, sample):
        # Preparing optimizer
        self.optimizer.zero_grad()

        # Unwrapping sample
        observs, actions, rewards, _, dones = sample

        # Discounted returns to go
        returns = discounted_rewards_to_go(rewards, dones, self.gamma)

        # Preparing tensors
        observs = torch.tensor(observs).to(self._device)
        actions = torch.tensor(actions).to(self._device)
        returns = torch.tensor(returns).to(self._device)

        # Calculating loss
        loss = self.loss(observs, actions, returns)

        # Back-propagation
        loss.backward()
        self.optimizer.step()

        # Returning loss metrics
        return {'loss': loss.item()}

    def post_episode(self): ...
