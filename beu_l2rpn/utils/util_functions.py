from abc import ABCMeta

import numpy as np
import torch
from torch.distributions import Categorical, normal


def abstract(cls):
    return ABCMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))


def save_score_results(file_path, results):
    """Saves results as a numpy file at given path"""
    np.save(file_path, results)


def normalise_rewards(rewards):
    """Normalises rewards to mean 0 and standard deviation 1"""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return (rewards - mean_reward) / (std_reward + 1e-8)  # 1e-8 added for stability


def create_actor_distribution(action_types, actor_output, action_size):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:, action_size:].squeeze(0)
        if len(means.shape) == 2:
            means = means.squeeze(-1)
        if len(stds.shape) == 2:
            stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution


def get_scalers():
    # heuristic to scale observation attributes
    return {
        "year": 2000,
        "month": 12,
        "day": 30,
        "day_of_week": 7,
        "hour_of_day": 24,
        "minute_of_hour": 60,
        "prod_p": 100,
        "prod_q": 100,
        "prod_v": 100,
        "load_p": 100,
        "load_q": 100,
        "load_v": 100,
        "p_or": 100,
        "q_or": 100,
        "v_or": 100,
        "a_or": 1000,
        "p_ex": 100,
        "q_ex": 100,
        "v_ex": 100,
        "a_ex": 1000,
        "rho": 1,
        "topo_vect": 1,
        "line_status": 1,
        "timestep_overflow": 1,
        "time_before_cooldown_line": 10,
        "time_before_cooldown_sub": 10,
        "time_next_maintenance": 1,
        "duration_next_maintenance": 1,
        "actual_dispatch": 1,
        "target_dispatch": 1
    }


def shuffle(x):
    lx = len(x)
    s = np.random.choice(lx, size=lx, replace=False)
    return x[s]


def has_overflow(obs):
    return any(obs.rho > 1.02)
