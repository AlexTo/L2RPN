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
