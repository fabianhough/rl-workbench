'''
Inspired by SpinningUp and HuggingFace
'''



import torch
import torch.nn as nn

from torch.distributions import Categorical



class ModelRLReinforcePolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, activation=nn.ReLU):
        '''
        '''
        super().__init__()

        # Dynamically add linear projections and adds activations in between
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        ops = []
        for i in range(len(layer_dims) - 1): # Iterate two i at a time
            ops.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                ops.append(activation())

        # Applying operations to policy
        self._policy_params = nn.Sequential(*ops)


    def forward(self, x):
        return self._policy_params(x)

    def policy(self, obs):
        # Unnormalized log probabilities
        logits = self.forward(obs)

        # Includes Softmax using log-sum-exp
        return Categorical(logits=logits)

    def act(self, observ):
        # Sample an action
        return self.policy(observ).sample().item()

    def loss(self, observs, actions, weights):
        # Generating the log_probs
        log_probs = self.policy(obs).log_prob(actions)
        # Calculating the negative loss
        return -(log_probs * weights).mean()




class Batch():
    observs = []
    actions = []
    weights = []




def run():
    pass

    # REINFORCE/VPG

    # Hyperparameters
    # NOTE: Turn into adjustable values
    disc_gamma = 0.99
    batch_size = 2000

    batch_

    ep_rewards = []

    while True:
        pass
        # Get Action

        # Step through environment

        if done:

            # Calculate the discounted rewards-to-go
            # Generates: G_t = r_(t) + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
            for i in range(len(rewards))[::-1]:
                rewards[i] += disc_gamma * rewards[i+1] if i+1 < len(rewards) else 0


            # Calculate the loss
