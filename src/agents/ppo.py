
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
        clip_eps=0.2,
        critic_coeff=0.5,
        entropy_coeff=0.05,
        mini_batches=4,
        mini_batch_size=20,
    ):
        
        self._device = device
        self._env = env

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff

        # Has an internal mini_batch system, once rollout buffer is received
        self.mini_batches = mini_batches
        self.mini_batch_size = mini_batch_size

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
        return action, value

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

        # Preparing next values and log probs
        with torch.no_grad():
            next_values = self.value(next_observs).squeeze()
            log_probs = self.policy(observs).log_prob(actions)

        # GAE Calculation
        advantages = torch.zeros_like(rewards).to(self._device)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            # TD error
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
        returns = advantages + values

        # Mini-batches
        for mb in range(self.mini_batches):
            # Preparing optimizer
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            # Random sample
            mb_idxs = np.random.randint(0, len(observs), size=self.mini_batch_size)

            # Policy Loss
            new_dist = self.policy(observs[mb_idxs])
            new_log_probs = new_dist.log_prob(actions[mb_idxs])
            log_ratio = new_log_probs - log_probs[mb_idxs]
            ratio = log_ratio.exp()

            entropy = new_dist.entropy().mean()
            actor_loss = -torch.min(advantages[mb_idxs] * ratio, advantages[mb_idxs]* torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps)).mean()

            # Value Loss
            critic_loss = F.mse_loss(self.value_net(observs[mb_idxs]).squeeze(), returns[mb_idxs])
            
            # Loss
            loss = actor_loss + self.critic_coeff * critic_loss - self.entropy_coeff * entropy

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



