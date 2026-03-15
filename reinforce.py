import torch
import torch.nn as nn
import torch.nn.functional as F

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
        logits = self.forward(obs)
        return Categorical(logits=logits)

    def act(self, obs):
        '''
        Args:
            obs (torch.Tensor):   A given set of states

        Returns:
            (int, torch.Tensor):
        '''
        # Sample an action
        dist = self.policy(obs)
        action = dist.sample()

        # Return a single action and the log_prob of the action
        return action.item(), dist.log_prob(action)





def run():
    pass

    # REINFOR