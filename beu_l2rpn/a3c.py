"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
from abc import ABC

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from grid2op.Environment import MultiMixEnvironment

from beu_l2rpn.utils import v_wrap, set_init, push_and_pull, record, convert_obs, create_env


class Net(nn.Module, ABC):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

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
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, opt, global_ep, global_ep_r, res_queue, worker_id, nb_episodes, env, state_size,
                 action_size, update_global_iter, gamma, all_actions, obs_idx, selected_attributes, seed,
                 feature_scalers):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.name = 'w%02i' % worker_id
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.opt = global_net, opt
        self.local_net = Net(state_size, action_size)  # local network
        self.env = create_env(env, seed)
        self.nb_episodes = nb_episodes
        self.update_global_iter = update_global_iter
        self.gamma = gamma
        self.all_actions = all_actions
        self.obs_idx = obs_idx
        self.selected_attributes = selected_attributes
        self.feature_scalers = feature_scalers

    def convert_obs(self, observation):
        return convert_obs(observation, self.obs_idx, self.selected_attributes, self.feature_scalers)

    def run(self):
        total_step = 1
        while self.g_ep.value < self.nb_episodes:
            self.print(f"{self.env.name} - {self.env.chronics_handler.get_name()}")
            if isinstance(self.env, MultiMixEnvironment):
                s = self.env.reset(random=True)
            else:
                s = self.env.reset()
            s = self.convert_obs(s)

            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            ep_step = 0
            while True:
                a = self.local_net.choose_action(v_wrap(s[None, :]))
                act = self.all_actions[a]
                s_, r, done, info = self.env.step(act)
                s_ = self.convert_obs(s_)
                if done:
                    if len(info["exception"]) > 0:
                        r = -100
                    else:
                        r = 100
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.update_global_iter == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.local_net, self.global_net, done, s_, buffer_s, buffer_a, buffer_r,
                                  self.gamma)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, ep_step)
                        break
                s = s_
                total_step += 1
                ep_step += 1
        self.res_queue.put(None)

    def print(self, msg):
        print(f"{self.name} - {msg}")
