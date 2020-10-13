from __future__ import division

import copy
import logging
from logging.handlers import QueueListener, QueueHandler

import numpy as np
import grid2op
import torch
import torch.multiprocessing as mp
from grid2op.Environment import MultiMixEnvironment

from grid2op.Reward import RedispReward, CombinedScaledReward, CloseToOverflowReward, GameplayReward, \
    LinesReconnectedReward, L2RPNReward
from lightsim2grid import LightSimBackend

from beu_l2rpn.logger import NeptuneLogHandler


def setup_worker_logging(log_queue):
    queue_handler = QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.addHandler(queue_handler)
    logger.setLevel(logging.INFO)


def setup_main_logging(config) -> mp.Queue:
    log_queue = mp.Queue()
    log_handler = NeptuneLogHandler(config)
    log_handler.setLevel(logging.INFO)
    listener = QueueListener(log_queue, log_handler, respect_handler_level=True)
    listener.start()
    return log_queue


def shuffle(x):
    lx = len(x)
    s = np.random.choice(lx, size=lx, replace=False)
    return x[s]


def create_env(env, seed):
    env = grid2op.make(env, reward_class=CombinedScaledReward, backend=LightSimBackend())
    env.seed(seed)
    cr = env.reward_helper.template_reward
    cr.addReward("redisp", RedispReward(), 1.0)
    cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("gameplay", GameplayReward(), 1.0)
    cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    cr.addReward("l2rpn", L2RPNReward(), 2.0 / float(env.n_line))
    # Initialize custom rewards
    cr.set_range(-1.0, 1.0)
    cr.initialize(env)

    if isinstance(env, MultiMixEnvironment):
        for mix in env:
            mix.chronics_handler.shuffle(shuffler=shuffle)
    else:
        env.chronics_handler.shuffle(shuffler=shuffle)
    return env


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


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
                level = np.argwhere(dispatch_levels == redisp['amount'])
                if level > 4:
                    level = level - 1
                redisp_vector[obj_id * 8 + level] = 1

        action_tensor = np.concatenate(
            [switch_line_tensor, force_line_reconnect_vector, force_line_disconnect_vector, set_bus_1_vector,
             set_bus_2_vector, switch_bus_vector, redisp_vector])

        if i == 0:
            action_tensor = 1 - action_tensor
        action_tensor = action_tensor / action_tensor.sum()
        if action_tensor.sum() == 0:
            v = 0
        action_tensors.append(action_tensor)
        i += 1
    return np.array(action_tensors)
