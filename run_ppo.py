
import yaml

import torch
import mlflow
import gymnasium as gym

from src import experiment
from src.agents import AgentPPO
from src.learn import train, TrainFreq, SamplingType



def agent_builder_ppo(config, env, device):
    agent = AgentPPO(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        policy_hidden_dims=config.pop('policy_hidden_dims'),
        value_hidden_dims=config.pop('value_hidden_dims'),
        env=env,
        device=device,
        policy_lr=config.pop('policy_lr'),
        value_lr=config.pop('value_lr'),
        gamma=config.pop('gamma'),
        gae_lambda=config.pop('lambda'),
        clip_eps=config.pop('clip_epsilon'),
        critic_coeff=config.pop('critic_coeff'),
        entropy_coeff=config.pop('entropy_coeff'),
        mini_batches=config.pop('mini_batches'),
        mini_batch_size=config.pop('mini_batch_size')
    )
    return agent, config


if __name__ == '__main__':
    # Loading config
    with open('config-ppo.yaml', 'r') as f:
        config = yaml.safe_load(f)

    experiment(config, agent_builder_ppo)
