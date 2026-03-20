
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam

from .base import Agent
from ..model import SimpleLinearNet



class AgentDQN(Agent):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        env,
        activation=nn.ReLU,
        device='cpu',
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=10
    ):
        '''
        '''

        self._device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._tgt_update_freq = target_update_freq
        self._tgt_counter = 0
        self._env = env

        self.q_net = SimpleLinearNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
        self.tgt_net = SimpleLinearNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
        self.q_net.to(self._device)
        self.tgt_net.to(self._device)
        self._tgt_update()

        self.optimizer = Adam(
            params=self.q_net.parameters(),
            lr=lr
        )

    def _tgt_update(self):
        self.tgt_net.load_state_dict(self.q_net.state_dict())

    def policy(self, observs):
        # Generate q values
        q_vals = self.q_net.forward(observs)
        return q_vals

    def act(self, observ, explore=True, **kwargs):
        self.q_net.eval()
        with torch.no_grad():
            # Prepare observ
            observ_tensor = torch.tensor(observ, dtype=torch.float32).to(self._device)

            # Sample an action
            if explore and random.random() < self.epsilon:
                action = self._env.action_space.sample()
            else:
                action = self.policy(observ_tensor.unsqueeze(0)).argmax().item()
        self.q_net.train()
        return action

    def train(self, sample):
        # Unwrapping sample
        observs, actions, rewards, next_observs, dones = sample

        # Preparing tensors
        observs = torch.tensor(observs).to(self._device)
        actions = torch.tensor(actions).to(self._device)
        rewards = torch.tensor(rewards).to(self._device)
        next_observs = torch.tensor(next_observs).to(self._device)
        dones = torch.tensor(dones).to(self._device)

        # Preparing optimizer
        self.optimizer.zero_grad()

        # Calculating loss
        with torch.no_grad():
            # DQN
            # next_q_max, _ = self.q_net.forward(s_next_observs).max(dim=1)

            # DDQN
            next_q_idxs = self.q_net.forward(next_observs).argmax(dim=1)
            next_q_max = self.tgt_net.forward(next_observs).gather(1, next_q_idxs.unsqueeze(1)).squeeze()

            target_q = rewards + self.gamma * next_q_max * (1 - dones)

        loss = self.loss(observs, actions, target_q)

        # Back-propagation
        loss.backward()
        self.optimizer.step()

        # Target update
        self._tgt_counter += 1
        if self._tgt_counter >= self._tgt_update_freq:
            self._tgt_update()
            self._tgt_counter = 0

        # Returning loss metrics
        return {'loss': loss.item()}

    def loss(self, observs, actions, target_q):
        # observs (#, *observ_space)
        # actions (#, )
        # target_q (#, )

        # Generating q_vals
        q_vals = self.policy(observs)

        # Gathering q values associated with actions
        q_vals = q_vals.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        # Generating the loss
        loss = F.mse_loss(input=q_vals, target=target_q.squeeze())
        return loss

    def post_episode(self):
        self.epsilon = max(self._epsilon_end, self.epsilon * self._epsilon_decay)
