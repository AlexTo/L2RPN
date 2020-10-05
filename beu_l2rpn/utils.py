import copy
import os
import random

import grid2op
from grid2op.Environment import MultiMixEnvironment
from grid2op.Reward import CombinedScaledReward, CloseToOverflowReward, LinesReconnectedReward, L2RPNReward
from torch import nn
import torch
import numpy as np

from beu_l2rpn.rewards.gameplay_reward import GameplayReward
from beu_l2rpn.rewards.redisp_reward import RedispReward


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, local_net, global_net, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = local_net(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = local_net.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name, ep_step):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
        f"| steps survived: {ep_step}"
    )


def shuffle(x):
    lx = len(x)
    s = np.random.choice(lx, size=lx, replace=False)
    return x[s]


def create_env(env, seed):
    # environment = grid2op.make(env, reward_class=CombinedScaledReward)
    # cr = environment.reward_helper.template_reward
    # cr.addReward("redisp", RedispReward(), 1.0)
    # cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    # cr.addReward("gameplay", GameplayReward(), 1.0)
    # cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    # cr.addReward("l2rpn", L2RPNReward(), 2.0 / float(environment.n_line))
    # # Initialize custom rewards
    # cr.set_range(-1.0, 1.0)
    # cr.initialize(environment)
    environment = grid2op.make(env)
    environment.seed(seed)
    shuffle_env_chronics(environment)
    return environment


def shuffle_env_chronics(environment):
    if isinstance(environment, MultiMixEnvironment):
        for mix in environment:
            mix.chronics_handler.shuffle(shuffler=shuffle)
    else:
        environment.chronics_handler.shuffle(shuffler=shuffle)


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


def get_action_mappings(env, all_actions, selected_action_types):
    action_tensors = []
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
                level = np.argwhere(dispatch_levels == redisp['amount'])
                if level > 4:
                    level = level - 1
                redisp_vector[obj_id * 8 + level] = 1

        action_tensors.append(np.concatenate(
            [switch_line_tensor, force_line_reconnect_vector, force_line_disconnect_vector, set_bus_1_vector,
             set_bus_2_vector, switch_bus_vector, redisp_vector]))

    return np.array(action_tensors)


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


def convert_obs(observation, obs_idx, selected_attributes, feature_scalers):
    obs = copy.deepcopy(observation)
    for attr in selected_attributes:
        if not selected_attributes[attr]:
            continue
        setattr(obs, attr, getattr(obs, attr) / feature_scalers[attr])
    vect = obs.to_vect()
    return vect[obs_idx]


def set_random_seeds(seed):
    """Sets all possible random seeds so results can be reproduced"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
