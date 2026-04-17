

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
        policy_hidden_dims: list,
        q_hidden_dims: list,
        policy_output_dim: int,
        action_gain: float,
        action_bias: float,
        env,
        device,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        policy_lr=3e-4,
        q_lr=1e-3
    ):
        self._device = device
        self._env = env

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Action Scaling
        self.action_gain = action_gain
        self.action_bias = action_bias

        # Policy Net
        self.policyNet = SimpleLinearNet(
            input_dim=state_dim,
            output_dim=policy_output_dim,
            hidden_dims=policy_hidden_dims,
            activation=nn.ReLU
        )
        self.policy_mean = nn.Linear(policy_output_dim, action_dim)
        self.policy_logstd = nn.Linear(policy_output_dim, action_dim)

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
        self.policyNet.to(device)
        self.policy_mean.to(device)
        self.policy_logstd.to(device)
        self.qNet0.to(device)
        self.qNet1.to(device)
        self.qTgt0.to(device)
        self.qTgt1.to(device)

        # Copying Q Net weights
        self.qTgt0.load_state_dict(self.qNet0.state_dict())
        self.qTgt1.load_state_dict(self.qNet1.state_dict())

        # Optimizers
        self.policy_optim = Adam(
            params=list(self.policyNet.parameters()) + \
                list(self.policy_mean.parameters()) + \
                list(self.policy_logstd.parameters()),
            lr=policy_lr
        )
        self.q_optim = Adam(
            params=list(self.qNet0.parameters()) + list(self.qNet1.parameters()),
            lr=q_lr
        )

    def policy_forward(self, x):
        # Using cleanRL implementation
        x = self.policyNet(x)
        # Distributing to mean and log_std
        mean = self.policy_mean(x)
        log_std = self.policy_logstd(x)
        # Constraining log_std between [-1,1]
        log_std = torch.tanh(log_std)
        # Scaling between [-5,2]
        # 'log_std + 1' moves between [0,2], '0.5 *' moves it [0,1]; rest is scaling to [-5,2]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def sample_policy(self, observ):
        mean, log_std = self.policy_forward(observ)
        normal = Normal(mean, log_std.exp())
        raw_action = normal.rsample()
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_gain + self.action_bias

        log_prob = normal.log_prob(raw_action)
        # Change of variables
        log_prob = log_prob - torch.log(self.action_gain * (1 - tanh_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def act(self, observ):
        self.policyNet.eval()
        with torch.no_grad():
            # Prepare observ
            observ_tensor = torch.tensor(observ, dtype=torch.float32).to(self._device)

            # Sample an action
            mean, log_std = self.policy_forward(observ_tensor.unsqueeze(0))
            std = log_std.exp()
            normal = Normal(mean, std)
            # Only using sample here as we are only interestedin the action and not the gradient
            action = torch.tanh(normal.sample()) * self.action_gain + self.action_bias
            action = action.item()
        self.policyNet.train()
        return action, None


    def train(self, sample):
        # Unwrapping sample
        observs, actions, rewards, next_observs, dones, _ = sample

        # Preparing tensors
        observs = torch.tensor(observs).to(self._device)
        actions = torch.tensor(actions).to(self._device)
        rewards = torch.tensor(rewards).to(self._device)
        next_observs = torch.tensor(next_observs).to(self._device)
        dones = torch.tensor(dones).to(self._device)

        with torch.no_grad():
            # Getting next actions and next log_probs
            next_actions, next_log_probs = self.sample_policy(next_observs)
            qtgt0 = self.qTgt0(torch.cat([next_observs, next_actions], 1))
            qtgt1 = self.qTgt1(torch.cat([next_observs, next_actions], 1))
            
            q_tgts = torch.min(qtgt0, qtgt1) - self.alpha * next_log_probs
            td_tgts = rewards + (1 - dones) * self.gamma * q_tgts.squeeze(1)

        # Q Training
        q_vals0 = self.qNet0(torch.cat([observs, actions], 1)).squeeze(1)
        q_vals1 = self.qNet1(torch.cat([observs, actions], 1)).squeeze(1)
        q_loss = F.mse_loss(q_vals0, td_tgts) + F.mse_loss(q_vals1, td_tgts)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # Policy Training
        # Using Pi to denote actions from the current policy
        pi, log_pi = self.sample_policy(observs)
        q_pi_0 = self.qNet0(torch.cat([observs, pi], 1))
        q_pi_1 = self.qNet1(torch.cat([observs, pi], 1))
        min_q_pi = torch.min(q_pi_0, q_pi_1)

        # Goal is to minimize low-entropy, and maximize Q value
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Copying over to target Q
        for param, target_param in zip(self.qNet0.parameters(), self.qTgt0.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qNet1.parameters(), self.qTgt1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            

    def post_episode(self): ...
