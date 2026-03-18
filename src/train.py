'''
Training
'''

from enum import Enum

import mlflow
import gymnasium as gym

from .buffer import ReplayBuffer


class TrainFreq(Enum):
    STEP = 'step'
    EPISODE = 'episode'
    BATCH = 'batch'


class SamplingType(Enum):
    REPLAY = 'replay'
    NSTEP = 'nstep'
    BATCH = 'batch'




def train(
    agent,
    env_name,
    num_episodes: int=1,
    num_batches: int=1,
    train_freq: TrainFreq=TrainFreq.STEP,
    sampling: SamplingType=SamplingType.NSTEP,
    sample_size: int=1,
    mlflow_log: bool=True,
    env_eval_seed: int=42
):
    '''

    '''

    env = gym.make(id=env_name)

    if sampling == SamplingType.NSTEP:
        pass
    elif sampling == SamplingType.REPLAY:
        pass
    elif sampling == SamplingType.BATCH:
        pass

    global_steps = 0
    for batch in range(1, num_batches+1):



        for episode in range(1, num_episodes+1):

            ep_rewards = []

            observ, info = env.reset()
            steps = 0
            
            while True:
                action = agent.act(observ)

                next_observ, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if train_freq == TrainFreq.STEP:
                    agent.train()

                ep_rewards.append(reward)
                steps += 1
                global_steps += 1

                if done:
                    observ, info = env.reset()

                    if mlflow_log:
                        mlflow.log_metrics({
                            'episode_reward': sum(ep_rewards),
                            'episode_steps': steps
                        }, step=episode)

                    if train_freq == TrainFreq.EPISODE:
                        agent.train()

                    break
                else:
                    observ = next_observ


        if train_freq == TrainFreq.BATCH:
            agent.train()


