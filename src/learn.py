'''
Training
'''

import os
from enum import Enum

import mlflow
from gymnasium import Env

from .buffer import ReplayBuffer, RolloutBuffer, NStepBuffer
from .utils.image import save_gif


class TrainFreq(Enum):
    STEP = 'step'
    EPISODE = 'episode'
    BATCH = 'batch'


class SamplingType(Enum):
    REPLAY = 'replay'
    NSTEP = 'nstep'
    ROLLOUT = 'rollout'



def evaluate_episode(agent, env, env_seed=42):
    ## Evaluate policy
    observ, info = env.reset(seed=env_seed)
    total_rewards = []
    frames = []
    while True:
        frames.append(env.render())
        action, _ = agent.act(observ, explore=False)
        observ, reward, terminated, truncated, info = env.step(action)
        total_rewards.append(reward)
        if terminated or truncated:
            break

    return total_rewards, frames



def train(
    agent,
    env: Env,
    num_episodes: int=1,
    num_batches: int=1,
    train_freq: TrainFreq=TrainFreq.STEP,
    sampling: SamplingType=SamplingType.ROLLOUT,
    sample_size: int=1,
    mlflow_log: bool=True,
    env_eval_seed: int=42,
    episodes_per_visual: int=100
):
    '''

    '''

    # Creating buffer based on Sampling Type
    buffer = None
    reset_on_sample = False
    if sampling == SamplingType.ROLLOUT:
        buffer = RolloutBuffer()
        reset_on_sample = True
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
    global_episodes = 0
    for batch in range(num_batches):
        for episode in range(num_episodes):
            print(f'Batch: {batch:03d}\tEpisode {episode:04d}', end='\r')

            # Trajectory Rewards
            ep_rewards = []

            # Initializing state and trajectory steps
            observ, info = env.reset()
            steps = 0
            
            # Runs through episode until terminated or truncated
            while True:
                # Retrieve action from agent
                action, value = agent.act(observ)

                # Step through environment with action
                next_observ, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Adding experience to buffer
                buffer.add(observ, action, reward, next_observ, done, value)

                # Training per step
                if train_freq == TrainFreq.STEP:
                    if buffer.ready():
                        train_metrics = agent.train(buffer.sample())
                        mlflow.log_metrics(train_metrics, step=global_steps)

                # Incrementing rewards and steps
                ep_rewards.append(reward)
                steps += 1
                global_steps += 1

                if done:
                    # Logging episode specific metrics
                    if mlflow_log:
                        mlflow.log_metrics({
                            'episode_reward': sum(ep_rewards),
                            'episode_steps': steps
                        }, step=episode)

                    # Training per episode
                    if train_freq == TrainFreq.EPISODE:
                        if buffer.ready():
                            train_metrics = agent.train(buffer.sample())
                            mlflow.log_metrics(
                                train_metrics,
                                step=batch*num_episodes+episode
                            )
                            if reset_on_sample:
                                buffer.reset()

                    # Resetting environment and initial variables
                    observ, info = env.reset()
                    steps = 0
                    global_episodes += 1

                    # Ending the episode
                    break
                else:
                    # Setting next state to current
                    observ = next_observ

            agent.post_episode()

            # Evaluate Agent
            total_rewards, frames = evaluate_episode(
                agent=agent,
                env=env,
                env_seed=env_eval_seed
            )

            if global_episodes % episodes_per_visual == 0:
                # Saving gif of performance
                eval_gif_fn = f'eval_{batch:03d}_{episode:03d}.gif'
                save_gif(frames=frames, path=eval_gif_fn)
                mlflow.log_artifact(eval_gif_fn, artifact_path=f'eval')
                os.remove(eval_gif_fn)

            # Log evaluation metrics
            mlflow.log_metrics(
                {
                    'eval_reward': sum(total_rewards),
                    'eval_steps': len(total_rewards)
                }, step=batch*num_episodes + episode
            )

        # Training per batch
        if train_freq == TrainFreq.BATCH:
            if buffer.ready():
                train_metrics = agent.train(buffer.sample())
                mlflow.log_metrics(train_metrics, step=batch)
                if reset_on_sample:
                    buffer.reset()


        # Register Model



