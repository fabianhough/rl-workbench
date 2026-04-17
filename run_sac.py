
import yaml

import torch
import mlflow
import gymnasium as gym

from src import experiment
from src.agents import AgentSAC
from src.learn import train, TrainFreq, SamplingType




def agent_builder_sac(config, env, device):
    action_gain = (env.action_space.high - env.action_space.low) / 2.0
    action_bias = (env.action_space.high + env.action_space.low) / 2.0
    agent = AgentSAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        policy_hidden_dims=config.pop('policy_hidden_dims'),
        policy_output_dim=config.pop('policy_output_dim'),
        q_hidden_dims=config.pop('q_hidden_dims'),
        action_gain=action_gain,
        action_bias=action_bias,
        env=env,
        device=device,
        policy_lr=config.pop('policy_lr'),
        q_lr=config.pop('q_lr'),
        gamma=config.pop('gamma'),
        tau=config.pop('tau'),
        alpha=config.pop('alpha')
    )
    return agent, config


if __name__ == '__main__':
    # Loading config
    with open('config-sac.yaml', 'r') as f:
        config = yaml.safe_load(f)

    experiment(config, agent_builder_sac)