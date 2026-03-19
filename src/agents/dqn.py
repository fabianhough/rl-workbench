import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam

from ..model import SimpleLinearNet



class AgentDQN():
    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        activation=nn.ReLU,
        device='cpu',
        gamma=0.995,
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
        # Generate q values
        q_vals = self.forward(observs)
        return q_vals

    def act(self, observ):
        self.net.eval()
        with torch.no_grad():
            # Prepare observ
            observ_tensor = torch.tensor(observ, dtype=torch.float32).to(self._device)

            # Sample an action
            action = self.policy(observ_tensor.unsqueeze(0)).argmax().item()
        self.net.train()
        return action

    def train(self, sample):
        # Preparing optimizer
        self.optimizer.zero_grad()

        # Unwrapping sample
        observs, actions, rewards, next_observs, dones = sample

        # Preparing tensors
        observs = torch.tensor(observs).to(self._device)
        actions = torch.tensor(actions).to(self._device)
        rewards = torch.tensor(rewards).to(self._device)
        next_observs = torch.tensor(rewards).to(self._device)
        dones = torch.tensor(dones).to(self._device)

        # Calculating loss
        with torch.no_grad():
            # DQN
            # next_q_max, _ = self.net.forward(s_next_observs).max(dim=1)

            # DDQN
            next_q_idxs = self.net.policy(next_observs).argmax(dim=1)
            next_q_max = self.net.policy(next_observs).gather(1, next_q_idxs.unsqueeze(1)).squeeze()

            target_q = rewards + self.gamma * next_q_max * (1 - dones)

        loss = self.net.loss(observs, actions, target_q)

        # Back-propagation
        loss.backward()
        optimizer.step()

    def loss(self, observs, actions, target_q):
        # observs (#, *observ_space)
        # actions (#, )
        # target_q (#, )

        # Generating q_vals
        q_vals = self.policy(observs)

        # Gathering q values associated with actions
        q_vals = q_vals.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        # Generating the loss
        loss = F.mse_loss(input=q_vals, target=target_q)
        return loss
