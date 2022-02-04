import torch
import torch.nn as nn
from torch.nn import Sequential

print(f'Torch version :{torch.__version__}')


class ActorCriticNetwork(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(ActorCriticNetwork, self).__init__()

        self.net = Sequential(
            nn.Linear(in_dims, 256),  # can put *in_dims if the input is iterable
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dims)
        )

        self.init_weights()

    def forward(self, obs):
        output = self.net(obs)
        return output

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)
