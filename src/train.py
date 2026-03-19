'''
Training
'''

from enum import Enum

import mlflow
import gymnasium as gym

from .buffer import ReplayBuffer, RolloutBuffer, NStepBuffer


class TrainFreq(Enum):
    STEP = 'step'
    EPISODE = 'episode'
    BATCH = 'batch'


class SamplingType(Enum):
    REPLAY = 'replay'
    NSTEP = 'nstep'
    ROLLOUT = 'rollout'




def train(
    agent,
    env_name,
    num_episodes: int=1,
    num_batches: int=1,
    train_freq: TrainFreq=TrainFreq.STEP,
    sampling: SamplingType=SamplingType.ROLLOUT,
    sample_size: int=1,
    mlflow_log: bool=True,
    env_eval_seed: int=42
):
    '''

    '''

    # Generating Environment
    env = gym.make(id=env_name)

    # Creating buffer based on Sampling Type
    buffer = None
    if sampling == SamplingType.ROLLOUT:
        buffer = RolloutBuffer()
    elif sampling == SamplingType.NSTEP:
        buffer = NStepBuffer(n=sample_size)
    elif sampling == SamplingType.REPLAY:
        buffer = ReplayBuffer(
            buffer_len=10000,
            sample_size=sample_size,
            observ_space=env.observation_space,
            action_space=env.action_space
        )

    # Global step tracking
    global_steps = 0
    for batch in range(1, num_batches+1):
        for episode in range(1, num_episodes+1):

            # Trajectory Rewards
            ep_rewards = []

            # Initializing state and trajectory steps
            observ, info = env.reset()
            steps = 0
            
            # Runs through episode until terminated or truncated
            while True:
                # Retrieve action from agent
                action = agent.act(observ)

                # Step through environment with action
                next_observ, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Adding experience to buffer
                buffer.add(observ, action, reward, next_observ, done)

                # Training per step
                if train_freq == TrainFreq.STEP:
                    agent.train(buffer.sample())

                # Incrementing rewards and steps
                ep_rewards.append(reward)
                steps += 1
                global_steps += 1

                if done:
                    # Resetting environment and initial variables
                    observ, info = env.reset()
                    steps = 0

                    # Logging episode specific metrics
                    if mlflow_log:
                        mlflow.log_metrics({
                            'episode_reward': sum(ep_rewards),
                            'episode_steps': steps
                        }, step=episode)

                    # Training per episode
                    if train_freq == TrainFreq.EPISODE:
                        agent.train(buffer.sample())

                    # Ending the episode
                    break
                else:
                    # Setting next state to current
                    observ = next_observ


        # Training per batch
        if train_freq == TrainFreq.BATCH:
            agent.train(buffer.sample())


