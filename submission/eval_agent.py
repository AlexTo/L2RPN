import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from grid2op.Agent import BaseAgent


class Net(nn.Module, ABC):
    def __init__(self, s_dim, action_mappings, action_line_mappings):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.action_mappings = action_mappings
        self.action_line_mappings = action_line_mappings

        self.pi1 = nn.Linear(s_dim, 896)
        self.pi2 = nn.Linear(896, 896)
        self.pi3 = nn.Linear(896, action_mappings.shape[0])

        self.v1 = nn.Linear(s_dim, 896)
        self.v2 = nn.Linear(896, 896)
        self.v3 = nn.Linear(896, 1)

        self.distribution = torch.distributions.Categorical

    def forward(self, x):

        pi1 = torch.tanh(self.pi1(x))
        pi2 = torch.tanh(pi1)
        pi3 = self.pi3(pi2)

        logits = torch.matmul(pi3, self.action_mappings)

        return logits

    def choose_action(self, s, attention, k=1):
        self.eval()
        logits = self.forward(s)
        prob = F.softmax(logits * attention, dim=1).data
        m = self.distribution(prob)
        return np.unique(m.sample([k]).cpu().numpy())

class Agent(BaseAgent): 
    def __init__(self, env, state_size, action_mappings, action_line_mappings):
        self.env = env
        self.state_size = state_size
