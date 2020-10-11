from __future__ import division

from abc import ABC

import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module, ABC):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(in_features=state_size, out_features=896)
        self.linear2 = nn.Linear(in_features=896, out_features=896)
        self.linear3 = nn.Linear(in_features=896, out_features=action_size)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        return self.linear3(out)


class Critic(nn.Module, ABC):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(in_features=state_size, out_features=896)
        self.linear2 = nn.Linear(in_features=896, out_features=896)
        self.linear3 = nn.Linear(in_features=896, out_features=1)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        return self.linear3(out)


class A3C(nn.Module, ABC):
    def __init__(self, state_size, action_size):
        super(A3C, self).__init__()
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)

    def forward(self, x):
        return self.critic(x), self.actor(x)
