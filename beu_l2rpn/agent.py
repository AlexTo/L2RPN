import os

import numpy as np
import torch
import torch.multiprocessing as mp
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from beu_l2rpn.a3c import Net, Worker
from beu_l2rpn.shared_adam import SharedAdam
from beu_l2rpn.utils import get_action_mappings, init_obs_extraction, create_env


class BeUAgent(AgentWithConverter):

    def __init__(self, config, action_mappings_matrix=None, env=None):

        if env is None:
            self.env = create_env(config["env"], config["seed"])
        else:
            self.env = env

        self.config = config
        self.hyper_parameters = config["hyper_parameters"]
        AgentWithConverter.__init__(self, self.env.action_space, action_space_converter=IdToAct)
        # self.action_space.filter_action(self.filter_action)
        self.all_actions = np.array(self.action_space.all_actions)

        self.load_action_mappings(action_mappings_matrix)

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

    def load_action_mappings(self, action_mappings_matrix):
        config = self.config
        if action_mappings_matrix is not None:
            self.action_mappings = action_mappings_matrix
        elif os.path.exists(config['action_mappings_matrix']):
            with open(config['action_mappings_matrix'], 'rb') as f:
                self.action_mappings = np.load(f)
        else:
            self.action_mappings = get_action_mappings(self.env, self.all_actions,
                                                       self.hyper_parameters["selected_action_types"])
            np.save(config['action_mappings_matrix'], self.action_mappings)

    def my_act(self, transformed_observation, reward, done=False):
        return 0

    def train(self):
        global_net = Net(self.state_size, self.action_size)
        global_net.share_memory()
        opt = SharedAdam(global_net.parameters(), lr=self.hyper_parameters["learning_rate"])  # global optimizer

        global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
        workers = [
            Worker(global_net=global_net, opt=opt, global_ep=global_ep, global_ep_r=global_ep_r,
                   res_queue=res_queue, worker_id=i, env=self.config["env"],
                   nb_episodes=self.hyper_parameters["train_num_episodes"],
                   state_size=self.state_size, action_size=self.action_size,
                   update_global_iter=self.hyper_parameters["update_global_iter"],
                   gamma=self.hyper_parameters["discount_rate"], all_actions=self.all_actions,
                   selected_attributes=self.hyper_parameters["selected_attributes"], obs_idx=self.obs_idx,
                   seed=i + self.config["seed"], feature_scalers=self.hyper_parameters["feature_scalers"])
            for i in range(self.hyper_parameters["num_workers"])]

        [w.start() for w in workers]
        res = []
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]
        torch.save(global_net.state_dict(), "model.pth")
