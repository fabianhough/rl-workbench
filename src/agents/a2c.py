
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

from .base import Agent
from ..model import SimpleLinearNet



class AgentA2C():
    def __init__(self,
        input_dim: int,
        output_dim: int,
        policy_hidden_dims: list,
        value_hidden_dims: list,
        env,
        device='cpu',
        activation=nn.ReLU,
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99,
        critic_coeff=0.5,
        entropy_coeff=0.05
    ):
        
        self._device = device
        self._env = env

        self.gamma = gamma
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff

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

    def act(self, observ, **kwargs):
        self.policy_net.eval()
        with torch.no_grad():
            # Prepare observ
            observ_tensor = torch.tensor(observ, dtype=torch.float32).to(self._device)

            # Sample an action
            action = self.policy(observ_tensor.unsqueeze(0)).sample().item()
        self.policy_net.train()
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
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        # Calculating n-step TD rewards-to-go
        n = len(rewards)
        G_t = 0
        done_i = None
        for i in reversed(range(n)):
            if dones[i]:
                done_i = i
            G_t = rewards[i] + self.gamma * (1 - dones[i]) * G_t

        # Bootstrap if episode didn't end within buffer
        if done_i is None:
            G_t += self.gamma**n * value_net(next_observs[-1])

        # Critic
        # critic_td = rewards + self.gamma * (1 - dones) * value_net(next_observs)
        critic_td = G_t
        critic_y_h = value_net(observs[0])

        print(critic_td)
        print(critic_y_h)
        raise Exception

        # Actor
        advantage = (critic_td - critic_y_h).detach()
        actor_dist = policy_net.policy(observs[0])
        actor_log_prob = actor_dist.log_prob(actions[0])
        actor_entropy = actor_dist.entropy()

        # Calculating loss
        critic_loss = F.mse_loss(critic_y_h, critic_td.detach())
        actor_loss = -(actor_log_prob * advantage).mean() - self.entropy_coeff * actor_entropy.mean()
        loss = actor_loss + self.critic_coeff * critic_loss

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



