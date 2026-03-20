
import yaml

import torch
import mlflow
import gymnasium as gym

from src import experiment
from src.agents import AgentReinforce
from src.learn import train, TrainFreq, SamplingType



def agent_builder_reinforce(config, env):
    agent = AgentReinforce(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        hidden_dims=config.pop('hidden_dims'),
        device=device,
        gamma=config.pop('gamma'),
        lr=config.pop('lr')
    )
    return agent, config


if __name__ == '__main__':
    # Loading config
    with open('config-reinforce.yaml', 'r') as f:
        config = yaml.safe_load(f)

    experiment(config, agent_builder_reinforce)
