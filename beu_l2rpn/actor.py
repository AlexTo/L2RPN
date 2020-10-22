from abc import ABC
from collections import OrderedDict

import torch.nn.functional as F
from torch import nn


class Actor(nn.Module, ABC):
    def __init__(self, input_dim, action_mappings, config):
        super(Actor, self).__init__()

        self.action_mappings = action_mappings

        fc_layers_conf = config["hidden_layers"]

        fc_layers = OrderedDict()
        fc_layers.update({'hidden0': nn.Linear(in_features=input_dim, out_features=fc_layers_conf[0])})
        fc_layers.update({'relu0': nn.ReLU()})
        for i in range(len(fc_layers_conf)):
            if i < len(fc_layers_conf) - 1:
                fc_layers.update(
                    {f'hidden{i + 1}': nn.Linear(in_features=fc_layers_conf[i], out_features=fc_layers_conf[i + 1])})
                fc_layers.update({f'relu{i + 1}': nn.ReLU()})

        fc_layers.update(
            {f'hidden{len(fc_layers_conf)}': nn.Linear(in_features=fc_layers_conf[-1],
                                                       out_features=action_mappings.shape[0])})

        self.fc_layers = nn.Sequential(fc_layers)

    def forward(self, state_batch):
        out = self.fc_layers(state_batch)
        out = out.matmul(self.action_mappings)
        out = F.gumbel_softmax(out, dim=-1)
        return out
