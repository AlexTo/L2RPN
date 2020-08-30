import torch
import torch.nn as nn


class BeUNet(nn.Module):
    def __init__(self, observation_size, action_size):
        super(BeUNet, self).__init__()
        self.linear1 = nn.Linear(in_features=observation_size, out_features=800)
        self.linear2 = nn.Linear(in_features=800, out_features=800)
        self.linear3 = nn.Linear(in_features=800, out_features=800)
        self.linear4 = nn.Linear(in_features=800, out_features=494)
        self.linear5 = nn.Linear(in_features=494, out_features=494)
        self.linear6 = nn.Linear(in_features=494, out_features=action_size)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.relu(self.linear2(out))
        out = torch.relu(self.linear3(out))
        out = torch.relu(self.linear4(out))
        out = torch.relu(self.linear5(out))
        out = self.linear6(out)
        return out
