'''
Training
'''

from enum import Enum

import mlflow
from gymnasium import Env



class TrainFreq(Enum):
    STEP = 'step'
    EPISODE = 'episode'
    BATCH = 'batch'




def train(
    agent,
    env: Env,
    train_freq: TrainFreq = TrainFreq.STEP,
    num_episodes: int=1,
    batch_size: int=1,
    mlflow_log: bool=True
):
    '''

    '''

    global_steps = 0
    for batch in range(1, batch_size+1):



        for episode in range(1, num_episodes+1):

            ep_rewards = []

            observ, info = env.reset()
            steps = 0
            
            while True:
                action = agent.act(observ)

                next_observ, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if train_freq == TrainFreq.STEP:
                    pass

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
                        pass

                    break
                else:
                    observ = next_observ


        if train_freq == TrainFreq.BATCH:
            pass




