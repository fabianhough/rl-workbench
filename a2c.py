'''
Inspired by SpinningUp, Huggingface, and cleanRL
'''

import os
import yaml

import gymnasium as gym

import mlflow
from mlflow.models import infer_signature

import logging
logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
# NOTE: Specifically for mlflow export model complaints

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

from src.utils.image import save_gif
from src.episode import training_episode, evaluate_episode
from src.reward import discounted_rewards_to_go
from src.model import SimpleLinearPolicyNet




class A2CPolicyNet(SimpleLinearPolicyNet):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, activation=nn.ReLU):
        '''
        '''
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )

    def policy(self, observs):
        # Unnormalized log probabilities
        logits = self.forward(observs)

        # Includes Softmax using log-sum-exp
        return Categorical(logits=logits)

    def act(self, observ):
        # Sample an action
        return self.policy(observ.unsqueeze(0)).sample().item()


class A2CValueNet(SimpleLinearPolicyNet):
    def __init__(self, input_dim: int, hidden_dims: list, activation=nn.ReLU):
        '''
        '''
        super().__init__(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation
        )

    


def rl_a2c(
    num_episodes,
    log_episode,
    env_str,
    env_test_seed,
    model_name,
    policy_hidden_dims,
    value_hidden_dims,
    policy_lr,
    value_lr,
    gamma,
    entropy_coeff,
    critic_coeff
):

    ### REINFORCE/VPG

    ## Setup

    # Creating environment and rendering for gifs in MLFlow
    env = gym.make(env_str, render_mode="rgb_array")

    # Creating policy and value
    input_dim = env.observation_space.shape[0] # NOTE: Only for flat states
    output_dim = env.action_space.n # NOTE: Only for discrete actions

    policy_net = A2CPolicyNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=policy_hidden_dims
    )
    value_net = A2CValueNet(
        input_dim=input_dim,
        hidden_dims=value_hidden_dims
    )

    policy_net.to(device)
    value_net.to(device)

    policy_optimizer = Adam(
        params=policy_net.parameters(),
        lr=policy_lr
    )
    value_optimizer = Adam(
        params=value_net.parameters(),
        lr=value_lr
    )

    with mlflow.start_run():

        mlflow.log_params({
            'env': env_str,
            'gamma': gamma,
            'entropy_coeff': entropy_coeff,
            'num_episodes': num_episodes,
            'device': device,
            'policy_hidden_dims': policy_hidden_dims,
            'policy_lr': policy_lr,
            'value_hidden_dims': value_hidden_dims,
            'value_lr': value_lr
        })

        ## Training

        global_steps = 0
        for episode in range(num_episodes):
            print(f'Episode {episode}', end='\r')

            # Reset Env
            observ, info = env.reset()
            steps = 0
            ep_reward = 0

            # Running through episode
            while True: # NOTE: Based on episode limit
                # Get Action
                # Assumes action is a single action
                observ = torch.tensor(observ, dtype=torch.float32).to(device)
                action = policy_net.act(observ)

                # Step through environment
                next_observ, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ## Training
                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()

                critic_td = reward + gamma * (1 - done) * value_net(torch.tensor(next_observ, dtype=torch.float32).to(device))
                critic_y_h = value_net(observ)
                critic_loss = F.mse_loss(critic_y_h, critic_td.detach())

                advantage = (critic_td - critic_y_h).detach()
                action = torch.tensor(action, dtype=torch.int64).to(device)
                actor_dist = policy_net.policy(observ)
                actor_log_prob = actor_dist.log_prob(action)
                actor_entropy = actor_dist.entropy()
                actor_loss = -(actor_log_prob * advantage).mean() - entropy_coeff * actor_entropy.mean()

                loss = actor_loss + critic_coeff * critic_loss

                loss.backward()
                policy_optimizer.step()
                value_optimizer.step()

                mlflow.log_metrics(
                    {
                        'loss': loss,
                        'actor_loss': actor_loss.item(),
                        'critic_loss': critic_loss.item(),
                        'advantage': advantage.item(),
                        'critic_td': critic_td.item(),
                        'critic_y_h': critic_y_h.item(),
                        'actor_log_prob': actor_log_prob,
                        'actor_entropy': actor_entropy
                    },
                    step=global_steps
                )

                ep_reward += reward
                steps += 1
                global_steps += 1
                
                if done:
                    # Resetting environment on done
                    observ, info = env.reset()

                    # Log in MLFlow
                    mlflow.log_metrics(
                        {
                            'episode_reward': ep_reward,
                            'episode_steps': steps
                        },
                        step=episode
                    )
                    break
                else:
                    # Setting state for next state
                    observ = next_observ


            with torch.no_grad():
                total_rewards, frames = evaluate_episode(
                    env=env,
                    net=policy_net,
                    env_seed=env_test_seed,
                    device=device
                )


            if episode > 0 and (episode+1) % log_episode == 0:
                # Saving gif of performance
                eval_gif_fn = f'eval_episode_{episode:03d}.gif'
                save_gif(frames=frames, path=eval_gif_fn)
                mlflow.log_artifact(eval_gif_fn, artifact_path=f'eval')
                os.remove(eval_gif_fn)

        # Logging model
        input_sample = torch.zeros(1, input_dim).to(device)
        output_sample = policy_net.forward(input_sample).detach().cpu().numpy()
        input_sample = input_sample.cpu().numpy()
        signature = infer_signature(
            model_input=input_sample,
            model_output=output_sample
        )
        policy_net.cpu()
        model_info = mlflow.pytorch.log_model(
            pytorch_model=policy_net,
            name=model_name,
            input_example=input_sample,
            signature=signature
        )
        policy_net.to(device)
        mlflow.register_model(model_info.model_uri, model_name)

            

if __name__ == '__main__':
    with open('a2c-config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config.pop('tracking_uri'))
    mlflow.set_experiment(config.pop('experiment'))
    rl_a2c(**config)