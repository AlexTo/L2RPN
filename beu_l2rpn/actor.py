import torch
from abc import ABC
from collections import OrderedDict

from torch import nn


class Actor(nn.Module, ABC):
    def __init__(self, input_dim, action_mappings):
        super(Actor, self).__init__()

        self.action_mappings = action_mappings

        self.model = nn.Sequential(OrderedDict([
            ('hidden1', nn.Linear(in_features=input_dim, out_features=2048)),
            ('relu1', nn.ReLU()),
            ('hidden2', nn.Linear(in_features=2048, out_features=4096)),
            ('relu2', nn.ReLU()),
            ('hidden3', nn.Linear(in_features=4096, out_features=action_mappings.shape[1])),
        ]))

    def forward(self, state_batch):
        out = self.model(state_batch)
        out = out.matmul(self.action_mappings.T)
        out = torch.softmax(out, dim=-1)
        return out
