"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import logging
import math
import os
from abc import ABC
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from setproctitle import setproctitle as ptitle

from grid2op.Converter import IdToAct
from grid2op.Environment import MultiMixEnvironment
from torch import Tensor
from torch.nn import init

from utils import v_wrap, set_init, push_and_pull, record, convert_obs, create_env, cuda, setup_worker_logging


class TrainableElementWiseLayer(nn.Module):
    weight: Tensor

    def __init__(self, c, h, w):
        super(TrainableElementWiseLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, c, h, w))  # define the trainable parameter
        self.reset_parameters()

    def forward(self, x):
        # assuming x is of size b-1-h-w
        return x * self.weight  # element-wise multiplication

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class Net(nn.Module, ABC):
    def __init__(self, s_dim, a_dim, action_mappings=None):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_mappings = action_mappings

        self.pi1 = nn.Linear(s_dim, 896)
        self.pi2 = nn.Linear(896, 896)
        self.pi3 = nn.Linear(896, action_mappings.shape[0])

        self.point_wise_mul1 = TrainableElementWiseLayer(8, action_mappings.shape[0], action_mappings.shape[1])

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

        self.v1 = nn.Linear(s_dim, 896)
        self.v2 = nn.Linear(896, 896)
        self.v3 = nn.Linear(896, 1)
        set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        pi2 = torch.tanh(pi1)
        pi3 = self.pi3(pi2)

        am = F.relu(self.point_wise_mul1(self.action_mappings))
        am = self.conv1(am)

        logits = torch.matmul(pi3, am)

        v1 = torch.tanh(self.v1(x))
        v2 = torch.tanh(v1)
        values = self.v3(v2)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().cpu().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss, a_loss.mean(), c_loss.mean()


class Agent(mp.Process):
    def __init__(self, global_net, opt, global_ep, global_ep_r, res_queue, rank, config, state_size,
                 obs_idx, log_queue, action_mappings):
        super(Agent, self).__init__()
        self.rank = rank
        self.seed = config["seed"] + rank
        self.gpu_id = config["gpu_ids"][rank % len(config["gpu_ids"])]

        torch.manual_seed(self.seed)
        if self.gpu_id >= 0:
            torch.cuda.manual_seed(self.seed)

        self.config = config
        self.state_size = state_size
        self.name = 'w%02i' % rank
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.opt = global_net, opt

        self.num_episodes = config["num_episodes"]
        self.update_global_iter = config["update_global_iter"]
        self.gamma = config["gamma"]
        self.obs_idx = obs_idx
        self.selected_attributes = config["selected_attributes"]
        self.feature_scalers = config["feature_scalers"]
        self.log_queue = log_queue
        self.action_mappings = action_mappings

    def convert_obs(self, observation):
        return convert_obs(observation, self.obs_idx, self.selected_attributes, self.feature_scalers)

    def run(self):
        ptitle('Training Agent: {}'.format(self.rank))
        config = self.config
        check_point_episodes = config["check_point_episodes"]
        check_point_folder = os.path.join(config["check_point_folder"], config["env"])
        setup_worker_logging(self.log_queue)

        self.env = create_env(config["env"], self.seed)

        action_space = IdToAct(self.env.action_space)
        action_space.init_converter(all_actions=os.path.join("data", f"{config['env']}_action_space.npy"))
        self.action_space = action_space
        self.local_net = Net(self.state_size, action_space.size(), self.action_mappings)  # local network
        self.local_net = cuda(self.gpu_id, self.local_net)

        total_step = 1
        l_ep = 0
        while self.g_ep.value < self.num_episodes:
            self.print(f"{self.env.name} - {self.env.chronics_handler.get_name()}")
            if isinstance(self.env, MultiMixEnvironment):
                s = self.env.reset(random=True)
            else:
                s = self.env.reset()

            connectivity = s.connectivity_matrix()

            s = self.convert_obs(s)
            s = v_wrap(s[None, :])
            s = cuda(self.gpu_id, s)

            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            ep_step = 0
            while True:
                a = self.local_net.choose_action(s)
                logging.info(f"{self.name}_act|||{a}")
                act = self.action_space.convert_act(a)

                s_, r, done, info = self.env.step(act)

                connectivity_ = s_.connectivity_matrix()

                topo_diff = np.sum(abs(connectivity_ - connectivity))

                if topo_diff == 0:
                    r = r - 1.5

                connectivity = connectivity_

                s_ = self.convert_obs(s_)
                s_ = v_wrap(s_[None, :])
                s_ = cuda(self.gpu_id, s_)

                if done:
                    if len(info["exception"]) > 0:
                        r = -10
                    else:
                        r = 10

                r += 10
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.update_global_iter == 0 or done:  # update global and assign to local net
                    # sync

                    buffer_a = cuda(self.gpu_id, torch.tensor(buffer_a, dtype=torch.int32))
                    buffer_s = cuda(self.gpu_id, torch.cat(buffer_s))

                    push_and_pull(self.opt, self.local_net, check_point_episodes, check_point_folder, self.g_ep, l_ep,
                                  self.name, self.rank, self.global_net, done, s_, buffer_s, buffer_a, buffer_r,
                                  self.gamma, self.gpu_id)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, ep_step)
                        break
                s = s_
                total_step += 1
                ep_step += 1
            l_ep += 1
        self.res_queue.put(None)

    def print(self, msg):
        print(f"{self.name} - {msg}")
