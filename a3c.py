"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import logging
import os
from abc import ABC
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from grid2op.Converter import IdToAct
from grid2op.Environment import MultiMixEnvironment
from setproctitle import setproctitle as ptitle
from torch.distributions import Binomial
from expert_rules import expert_rules

from utils import v_wrap, set_init, push_and_pull, record, convert_obs, create_env, cuda, setup_worker_logging, lreward, forecast, forecast_actions, get_bad_lines


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

        set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):

        pi1 = torch.tanh(self.pi1(x))
        pi2 = torch.tanh(pi1)
        pi3 = self.pi3(pi2)

        logits = torch.matmul(pi3, self.action_mappings)

        v1 = torch.tanh(self.v1(x))
        v2 = torch.tanh(v1)
        values = self.v3(v2)
        return logits, values

    def choose_action(self, s, attention, k=1):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits * attention, dim=1).data
        m = self.distribution(prob)
        return np.unique(m.sample([k]).cpu().numpy())

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()

        a_loss = -exp_v + 0.05 * m.entropy()
        total_loss = (c_loss + a_loss).mean()
        return total_loss, a_loss.mean(), c_loss.mean()


class Agent(mp.Process):
    def __init__(self, global_net, opt, global_ep, global_step, global_ep_r, res_queue, global_num_candidate_acts, rank, config, log_queue, action_mappings, action_line_mappings):
        super(Agent, self).__init__()
        self.rank = rank
        self.seed = config["seed"] + rank
        self.gpu_id = config["gpu_ids"][rank % len(config["gpu_ids"])]

        torch.manual_seed(self.seed)
        if self.gpu_id >= 0:
            torch.cuda.manual_seed(self.seed)

        self.config = config
        self.state_size = config["state_size"]
        self.name = 'w%02i' % rank
        self.g_ep, self.g_step, self.g_ep_r, self.res_queue, self.g_num_candidate_acts = global_ep, global_step, global_ep_r, res_queue, global_num_candidate_acts

        self.global_net, self.opt = global_net, opt

        self.num_episodes = config["num_episodes"]
        self.update_global_iter = config["update_global_iter"]
        self.gamma = config["gamma"]
        self.selected_attributes = config["selected_attributes"]
        self.feature_scalers = config["feature_scalers"]
        self.log_queue = log_queue
        self.action_mappings = action_mappings
        self.action_line_mappings = action_line_mappings

    def convert_obs(self, observation_space, observation):
        return convert_obs(observation_space, observation, self.selected_attributes, self.feature_scalers)

    def run(self):
        ptitle('Training Agent: {}'.format(self.rank))
        config = self.config
        check_point_episodes = config["check_point_episodes"]
        check_point_folder = os.path.join(
            config["check_point_folder"], config["env"])
        setup_worker_logging(self.log_queue)

        self.env = create_env(config["env"], self.seed)
        observation_space = self.env.observation_space
        action_space = IdToAct(self.env.action_space)
        with open(os.path.join("data", f"{config['env']}_action_space.npz"), 'rb') as f:
            archive = np.load(f)
            action_space.init_converter(all_actions=archive[archive.files[0]])

        self.action_space = action_space
        all_actions = np.array(action_space.all_actions)

        self.local_net = Net(
            self.state_size, self.action_mappings, self.action_line_mappings)  # local network
        self.local_net = cuda(self.gpu_id, self.local_net)

        total_step = 1
        l_ep = 0
        while self.g_ep.value < self.num_episodes:
            self.print(
                f"{self.env.name} - {self.env.chronics_handler.get_name()}")
            if isinstance(self.env, MultiMixEnvironment):
                obs = self.env.reset(random=True)
            else:
                obs = self.env.reset()

            maintenance_list = obs.time_next_maintenance + obs.duration_next_maintenance

            s = self.convert_obs(observation_space, obs)
            s = v_wrap(s[None, :])
            s = cuda(self.gpu_id, s)

            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            ep_step = 0
            ep_agent_num_dmd = 0
            ep_agent_num_acts = 0
            while True:
                bad_lines = get_bad_lines(obs.rho, config["danger_threshold"])

                expert_act = expert_rules(
                    self.name, maintenance_list, ep_step, action_space, obs)

                if expert_act is not None:
                    a = np.where(all_actions == expert_act)[0][0]
                    choosen_actions = np.array([a])
                    #print(f"Expert act: {a}")
                elif bad_lines.size <= 0:
                    choosen_actions = np.array([0])
                else:
                    choosen_actions = np.array([])
                    for bad_line in bad_lines:
                        attention = self.action_line_mappings[bad_line]
                        choosen_actions = np.append(choosen_actions, self.local_net.choose_action(s, attention, self.g_num_candidate_acts.value))
                    ep_agent_num_dmd += 1

                obs_previous = obs
                a, obs_forecasted, obs_do_nothing = forecast_actions(
                    np.unique(choosen_actions), self.action_space, obs, min_threshold=0.95)

                #logging.info(f"{self.name}_act|||{a}")
                act = self.action_space.convert_act(a)

                obs, r, done, info = self.env.step(act)

                r = lreward(
                    a, self.env, obs_previous, obs_do_nothing, obs_forecasted, obs, done, info, threshold_safe=0.9)

                if a > 0:
                    if r > 0:
                        print("+", end="")
                    elif r < 0:
                        print("-", end="")
                    elif len(choosen_actions) > 0:
                        print("*", end="")
                    else:
                        print("x", end="")
                else:
                    if len(choosen_actions) > 0:
                        print("o", end="")
                    else:
                        print("0", end="")

                if r > 0:
                    ep_agent_num_acts += 1

                s_ = self.convert_obs(observation_space, obs)
                s_ = v_wrap(s_[None, :])
                s_ = cuda(self.gpu_id, s_)

                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                
                logging.info(f"{self.name}_log|||{ep_step},{self.env.name},{self.env.chronics_handler.get_name()},{a},{r}")

                if total_step % self.update_global_iter == 0 or done:  # update global and assign to local net
                    # sync

                    # if len(buffer_r) > 0 and np.mean(np.abs(buffer_r)) > 0:
                    buffer_a = cuda(self.gpu_id, torch.tensor(
                        buffer_a, dtype=torch.long))
                    buffer_s = cuda(self.gpu_id, torch.cat(buffer_s))
                    push_and_pull(self.opt, self.local_net, check_point_episodes, check_point_folder, self.g_ep, l_ep,
                                  self.name, self.rank, self.global_net, done, s_, buffer_s, buffer_a, buffer_r, self.gamma, self.gpu_id)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        print("")
                        record(config["starting_num_candidate_acts"], config["num_candidate_acts_decay_iter"], self.g_ep, self.g_step, self.g_num_candidate_acts,
                               self.g_ep_r, ep_r, self.res_queue, self.name, ep_step, ep_agent_num_dmd, ep_agent_num_acts)
                        break
                s = s_
                total_step += 1
                ep_step += 1
            l_ep += 1
        self.res_queue.put(None)

    def print(self, msg):
        print(f"{self.name} - {msg}")
