import os
import random

import grid2op
import numpy as np
import torch
from grid2op.Environment import MultiMixEnvironment
from grid2op.Reward import CombinedReward, RedispReward, EconomicReward, CloseToOverflowReward, GameplayReward, \
    LinesReconnectedReward, L2RPNReward
from lightsim2grid import LightSimBackend
from torch import nn
from torch.optim import Adam

from beu_l2rpn.actor import Actor
from beu_l2rpn.critic import Critic


def shuffle(x):
    lx = len(x)
    s = np.random.choice(lx, size=lx, replace=False)
    return x[s]


def shuffle_env_chronics(environment):
    if isinstance(environment, MultiMixEnvironment):
        for mix in environment:
            mix.chronics_handler.shuffle(shuffler=shuffle)
    else:
        environment.chronics_handler.shuffle(shuffler=shuffle)


def create_env(env, seed):
    environment = grid2op.make(env, reward_class=CombinedReward, backend=LightSimBackend())
    cr = environment.reward_helper.template_reward
    cr.addReward("redisp", RedispReward(), 0.1)
    cr.addReward("economic", EconomicReward(), 0.1)
    cr.addReward("overflow", CloseToOverflowReward(), 0.1)
    cr.addReward("gameplay", GameplayReward(), 0.1)
    cr.addReward("recolines", LinesReconnectedReward(), 0.1)
    cr.addReward("l2rpn", L2RPNReward(), .6 / float(environment.n_line))
    # Initialize custom rewards
    # cr.set_range(-1.0, 1.0)
    cr.initialize(environment)
    environment.seed(seed)
    shuffle_env_chronics(environment)
    return environment


def filter_action(action):
    impacts = action.impact_on_objects()
    if impacts['force_line']['changed'] and impacts['force_line']['disconnections']['count'] > 0:
        return False
    return True


def init_obs_extraction(observation_space, selected_attributes):
    idx = np.zeros(0, dtype=np.uint)
    size = 0
    for obs_attr_name in selected_attributes:
        if not selected_attributes[obs_attr_name]:
            continue
        beg_, end_, dtype_ = observation_space.get_indx_extract(obs_attr_name)
        idx = np.concatenate((idx, np.arange(beg_, end_, dtype=np.uint)))
        size += end_ - beg_  # no "+1" needed because "end_" is exclude by python convention
    return idx, size


def has_overflow(obs):
    return any(obs.rho > 1.02)


def get_topo_pos_vect(env, obj_type):
    pos_vect = env.line_or_pos_topo_vect
    if obj_type == 'line (origin)':
        pos_vect = env.line_or_pos_topo_vect
    elif obj_type == 'line (extremity)':
        pos_vect = env.line_ex_pos_topo_vect
    elif obj_type == 'load':
        pos_vect = env.load_pos_topo_vect
    elif obj_type == 'generator':
        pos_vect = env.gen_pos_topo_vect
    return pos_vect


def convert_obs(observation_space, observation, selected_attributes, scalers, obs_is_vect=False):
    if not obs_is_vect:
        vect = observation.to_vect()
    else:
        vect = observation

    feature_vects = []
    for attr in selected_attributes:
        if not selected_attributes[attr]:
            continue
        beg_, end_, _ = observation_space.get_indx_extract(attr)
        if attr == "topo_vect":
            topo_vect = vect[beg_:end_].astype("int")
            feature_vect = np.zeros((topo_vect.size, 2))
            feature_vect[np.arange(topo_vect.size), topo_vect - 1] = 1
            feature_vect = feature_vect.flatten()
        elif attr == "timestep_overflow":
            timestep_overflow = vect[beg_:end_].astype("int")
            feature_vect = np.zeros((timestep_overflow.size, 4))
            feature_vect[np.arange(timestep_overflow.size), timestep_overflow] = 1
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


def set_random_seeds(env, random_seed):
    """Sets all possible random seeds so results can be reproduced"""
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    env.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


def create_action_mappings(env, all_actions, selected_action_types):
    action_tensors = []
    i = 0
    for act in all_actions:

        impacts = act.impact_on_objects()

        switch_line_tensor = np.zeros(env.n_line)
        if selected_action_types["switch_line"]:
            switch_line_tensor[impacts['switch_line']['powerlines']] = 1

        force_line_disconnect_vector = np.zeros(env.n_line)
        if selected_action_types["force_line_disconnect"]:
            force_line_disconnect_vector[impacts['force_line']['disconnections']['powerlines']] = 1

        force_line_reconnect_vector = np.zeros(env.n_line)
        if selected_action_types["force_line_reconnect"]:
            force_line_reconnect_vector[impacts['force_line']['reconnections']['powerlines']] = 1

        set_bus_1_vector = np.zeros(env.dim_topo)
        set_bus_2_vector = np.zeros(env.dim_topo)

        if selected_action_types["set_bus"]:
            for bus_assign in impacts['topology']['assigned_bus']:
                if bus_assign['bus'] == 1:
                    bus_vector = set_bus_1_vector
                else:
                    bus_vector = set_bus_2_vector

                obj_id = bus_assign['object_id']
                obj_type = bus_assign['object_type']

                pos_vect = get_topo_pos_vect(env, obj_type)

                bus_vector[pos_vect[obj_id]] = 1

        switch_bus_vector = np.zeros(env.dim_topo)
        if selected_action_types["switch_bus"]:
            for bus_switch in impacts['topology']['bus_switch']:
                obj_id = bus_switch['object_id']
                obj_type = bus_switch['object_type']
                pos_vect = get_topo_pos_vect(env, obj_type)
                switch_bus_vector[pos_vect[obj_id]] = 1

        redisp_vector = np.zeros(env.n_gen * 8)
        if selected_action_types["redispatch"]:
            for redisp in impacts['redispatch']['generators']:
                obj_id = redisp['gen_id']
                dispatch_levels = np.linspace(-env.gen_max_ramp_down[obj_id], env.gen_max_ramp_up[obj_id], 9)
                level = np.argwhere(np.abs(dispatch_levels - redisp['amount']) < 0.01)
                if level > 4:
                    level = level - 1
                redisp_vector[obj_id * 8 + level] = 1

        action_tensor = np.concatenate(
            [switch_line_tensor, force_line_reconnect_vector, force_line_disconnect_vector, set_bus_1_vector,
             set_bus_2_vector, switch_bus_vector, redisp_vector])

        if i == 0:
            action_tensor = 1 - action_tensor
        action_tensor = action_tensor / action_tensor.sum()
        action_tensors.append(action_tensor)
        i += 1
        if i % 1000 == 0:
            print(f"Action mappings: Processed {i} actions")

    return np.array(action_tensors)


def take_optim_step(optimizer, network, loss, clipping_norm=None, retain_graph=False):
    """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
    if not isinstance(network, list):
        network = [network]
    optimizer.zero_grad()  # reset gradients to 0
    loss.backward(retain_graph=retain_graph)  # this calculates the gradients
    if clipping_norm is not None:
        for net in network:
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                           clipping_norm)  # clip gradients to help stabilise training
    optimizer.step()  # this applies the gradients


def soft_update(local_model, target_model, tau):
    """Updates the target network in the direction of the local network but by taking a step size
    less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def copy_model(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())


def create_networks(config, action_size, state_size, action_mappings, device):
    critic_1 = nn.DataParallel(Critic(input_dim=state_size,
                                      action_mappings=action_mappings,
                                      config=config["Critic"])).to(device)

    critic_2 = nn.DataParallel(Critic(input_dim=state_size,
                                      action_mappings=action_mappings,
                                      config=config["Critic"])).to(device)

    critic_target_1 = nn.DataParallel(Critic(input_dim=state_size,
                                             action_mappings=action_mappings,
                                             config=config["Critic"])).to(device)

    critic_target_2 = nn.DataParallel(Critic(input_dim=state_size,
                                             action_mappings=action_mappings,
                                             config=config["Critic"])).to(device)

    actor = nn.DataParallel(Actor(input_dim=state_size,
                                  action_mappings=action_mappings,
                                  config=config["Actor"])).to(device)

    critic_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config["Critic"]["learning_rate"], eps=1e-4)
    critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=config["Critic"]["learning_rate"], eps=1e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["Actor"]["learning_rate"], eps=1e-4)

    copy_model(critic_1, critic_target_1)
    copy_model(critic_2, critic_target_2)

    auto_entropy_tuning = config["auto_entropy_tuning"]
    log_alpha, alpha_optim, target_entropy = None, None, None
    if auto_entropy_tuning:
        # we set the max possible entropy as the target entropy
        target_entropy = -np.log(1.0 / action_size) * 0.98
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp()
        alpha_optim = Adam([log_alpha], lr=config["Actor"]["learning_rate"], eps=1e-4)
    else:
        alpha = config["entropy_term_weight"]

    return actor, critic_1, critic_2, critic_target_1, critic_target_2, actor_optimizer, critic_optimizer, \
           critic_optimizer_2, alpha, log_alpha, alpha_optim, target_entropy
