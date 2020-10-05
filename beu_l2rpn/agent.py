import numpy as np
import torch
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from beu_l2rpn.utils import init_obs_extraction, load_action_mappings


class BeUAgent(AgentWithConverter):

    def __init__(self, config, env, action_mappings=None):
        self.env = env
        self.config = config
        self.hyper_parameters = config["hyper_parameters"]

        AgentWithConverter.__init__(self, self.env.action_space, action_space_converter=IdToAct)
        # self.action_space.filter_action(self.filter_action)
        self.all_actions = np.array(self.action_space.all_actions)

        if action_mappings is None:
            self.action_mappings = load_action_mappings(env, config, all_actions=self.all_actions)
        else:
            self.action_mappings = action_mappings

        self.action_mappings = torch.tensor(self.action_mappings, requires_grad=False).float()

        self.observation_space = self.env.observation_space

        obs_idx, obs_size = init_obs_extraction(self.observation_space, self.hyper_parameters['selected_attributes'])
        self.obs_idx = obs_idx

        self.action_size = int(self.action_space.size())
        self.state_size = int(obs_size)
        print(f"State size: {obs_size}")
        print(f"Action size: {self.action_size}")
        self.hyper_parameters["action_size"] = self.action_size
        self.hyper_parameters["state_size"] = self.state_size

    def my_act(self, transformed_observation, reward, done=False):
        return 0
