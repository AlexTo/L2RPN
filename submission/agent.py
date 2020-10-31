import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from grid2op.Agent import BaseAgent
from .expert_rules import expert_rules


class Net(nn.Module):
    def __init__(self, state_size, action_mappings, action_line_mappings):
        super(Net, self).__init__()
        self.action_mappings = action_mappings
        self.action_line_mappings = action_line_mappings

        self.pi1 = nn.Linear(state_size, 896)
        self.pi2 = nn.Linear(896, 896)
        self.pi3 = nn.Linear(896, action_mappings.shape[0])

        self.v1 = nn.Linear(state_size, 896)
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
    def __init__(self, env, config, action_space, action_mappings, action_line_mappings):
        super(Agent, self).__init__(action_space)
        self.env = env
        self.config = config
        self.obs_space = env.observation_space
        self.state_size = config["state_size"]
        self.selected_attributes = config["selected_attributes"]
        self.feature_scalers = config["feature_scalers"]
        self.action_space = action_space

        self.action_mappings = torch.tensor(action_mappings).float()
        self.action_line_mappings = torch.tensor(action_line_mappings).float()
        self.net = Net(self.state_size, self.action_mappings,
                       self.action_line_mappings)
        self.ep_step = 0

    def reset(self, obs):
        self.maintenance_list = obs.time_next_maintenance + obs.duration_next_maintenance
        self.ep_step = 0

    def act(self, obs, reward, done):

        expert_act = expert_rules(
            self.maintenance_list, self.ep_step, self.action_space, obs)

        if expert_act is not None:
            self.ep_step += 1
            return expert_act

        rho = obs.rho.copy()
        rho[rho == 0.0] = 1.0
        lines_overload = rho > self.config["danger_threshold"]

        if not np.any(lines_overload):
            choosen_actions = np.array([0])
        else:
            s = torch.tensor(self.convert_obs(obs)).float()
            lines_overload = torch.tensor(lines_overload.astype(int)).float()
            attention = torch.matmul(lines_overload.reshape(
                1, -1), self.action_line_mappings)
            attention[attention > 1] = 1
            choosen_actions = self.net.choose_action(s, attention)

        a, _, _ = self.forecast_actions(choosen_actions, self.action_space, obs)
        self.ep_step += 1
        return self.action_space.convert_act(a)

    def convert_obs(self, obs):
        selected_attributes = self.selected_attributes
        obs_space = self.obs_space
        scalers = self.feature_scalers

        vect = obs.to_vect()
        feature_vects = []
        for attr in selected_attributes:
            if not selected_attributes[attr]:
                continue
            beg_, end_, _ = obs_space.get_indx_extract(attr)
            if attr == "topo_vect":
                topo_vect = vect[beg_:end_].astype("int")
                feature_vect = np.zeros((topo_vect.size, 2))
                feature_vect[np.arange(topo_vect.size), topo_vect - 1] = 1
                feature_vect = feature_vect.flatten()
            elif attr == "month":
                feature_vect = np.zeros(12)
                feature_vect[obs.month - 1] = 1
            elif attr == "day_of_week":
                feature_vect = np.zeros(7)
                feature_vect[obs.day_of_week] = 1
            elif attr == "hour_of_day":
                feature_vect = np.zeros(24)
                feature_vect[obs.hour_of_day] = 1
            elif attr == "timestep_overflow":
                timestep_overflow = vect[beg_:end_].astype("int")
                feature_vect = np.zeros((timestep_overflow.size, 4))
                feature_vect[np.arange(timestep_overflow.size),
                             timestep_overflow] = 1
                feature_vect = feature_vect.flatten()
            elif attr == "time_before_cooldown_line" or attr == "time_before_cooldown_sub" \
                    or attr == "time_next_maintenance" or attr == "duration_next_maintenance":
                v = vect[beg_:end_].astype("int")
                if attr == "time_next_maintenance":
                    v = v + 1  # because time_next_maintenance can be -1

                v[v > 12] = 12
                feature_vect = np.zeros((v.size, 13))
                feature_vect[np.arange(v.size), v] = 1
                feature_vect = feature_vect.flatten()
            else:
                feature_vect = vect[beg_:end_]

            feature_vect = feature_vect / scalers[attr]
            feature_vects.append(feature_vect)
        return np.concatenate(feature_vects)

    def forecast_actions(self, actions, action_space, obs, min_threshold=0.9, normalised=True):
        try:
            obs_do_nothing, _, done_do_nothing, _ = obs.simulate(
                action_space.convert_act(0))
            if done_do_nothing:
                obs_do_nothing = obs
        except:
            obs_do_nothing = obs


        best_action = 0
        best_impact = 0
        best_obs = obs_do_nothing

        for action in actions[actions > 0]:
            try:
                obs_forecasted, _, done_forecasted, _ = obs.simulate(
                    action_space.convert_act(action))
                if done_forecasted:
                    obs_forecasted = obs_do_nothing
            except:
                obs_forecasted = obs_do_nothing


            impact = self.compute_impact(obs_forecasted.rho, obs_do_nothing.rho)

            if impact < best_impact:
                best_action = action
                best_impact = impact
                best_obs = obs_forecasted

        return best_action, best_obs, obs_do_nothing
    
    def transform_rho(self, rho, min_threshold=0.02, normalised=True):
        rho_t = np.copy(rho)
        rho_t[rho_t == 0.0] = 1.01
        rho_t[rho_t > 1.0] = 1.01
        rho_t[rho_t < min_threshold] = min_threshold
        rho_t = -np.log(1.02 - rho_t)
        return rho_t if not normalised else (rho_t / -np.log(0.01))


    def compute_impact(self, rho1, rho2, min_threshold=0.02, normalised=True, eps=0.0000001):
        #impact = transform_rho(rho1, min_threshold, normalised) - \
        #    transform_rho(rho2, min_threshold, normalised)
        impact = (self.transform_rho(rho1) -  self.transform_rho(rho2))[np.logical_or(np.logical_or(rho1 == 0, rho1 > min_threshold), 
                                                                            np.logical_or(rho2 == 0, rho2 > min_threshold))]
        return impact.sum() / ((impact != 0).sum() + eps)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
