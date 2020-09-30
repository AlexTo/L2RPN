from abc import ABC
from collections import OrderedDict

from torch import nn


class Critic(nn.Module, ABC):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()

        self.model = nn.Sequential(OrderedDict([
            ('hidden1', nn.Linear(in_features=input_dim, out_features=768)),
            ('relu1', nn.ReLU()),
            ('hidden2', nn.Linear(in_features=768, out_features=768)),
            ('relu2', nn.ReLU()),
            ('output', nn.Linear(in_features=768, out_features=output_dim)),
        ]))

    def forward(self, state_batch):
        return self.model(state_batch)
