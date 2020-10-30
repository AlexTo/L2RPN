import logging
import logging
import os
import random
import time
import math
from logging.handlers import QueueHandler, QueueListener

import grid2op
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from grid2op.Environment import MultiMixEnvironment
from grid2op.Reward import ConstantReward
from lightsim2grid import LightSimBackend
from torch import nn

from logger import NeptuneLogHandler


def setup_worker_logging(log_queue):
    queue_handler = QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.addHandler(queue_handler)
    logger.setLevel(logging.INFO)


def setup_main_logging(config) -> mp.Queue:
    log_queue = mp.Queue()
    log_handler = NeptuneLogHandler(config)
    log_handler.setLevel(logging.INFO)
    listener = QueueListener(log_queue, log_handler,
                             respect_handler_level=True)
    listener.start()
    return log_queue


def create_env(env, seed):
    environment = grid2op.make(
        env, reward_class=ConstantReward, backend=LightSimBackend())
    # for mix in list(environment.keys()):
    # cr = environment[mix].get_reward_instance()
    # cr.addReward("redisp", RedispReward(), 0.1)
    # cr.addReward("economic", EconomicReward(), 0.1)
    # cr.addReward("overflow", CloseToOverflowReward(), 0.1)
    # cr.addReward("gameplay", GameplayReward(), 0.1)
    # cr.addReward("recolines", LinesReconnectedReward(), 0.1)
    # cr.addReward("l2rpn", L2RPNReward(), .6 / float(environment.n_line))
    # Initialize custom rewards
    # cr.set_range(-1.0, 1.0)

    # cr.initialize(environment[mix])
    environment.seed(seed)
    return environment


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def cuda(gpu_id, obj):
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            return obj.cuda()


def push_and_pull(opt, local_net, check_point_episodes, check_point_folder, g_ep, l_ep, name, rank, global_net, done, s_, bs, ba, br, gamma, gpu_id=-1):
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = local_net(s_)[1].detach().cpu().numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    buffer_v_target = cuda(gpu_id, v_wrap(np.array(buffer_v_target)[:, None]))

    loss, a_loss, c_loss = local_net.loss_func(bs, ba, buffer_v_target)
    # logging.info(f"{name}_actor_loss|||{a_loss.item()}")
    # logging.info(f"{name}_critic_loss|||{c_loss.item()}")
    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        if gp.grad is not None and gpu_id < 0:
            return
        elif gpu_id < 0:
            gp._grad = lp.grad
        elif lp.grad is not None:
            gp._grad = lp.grad.cpu()

    opt.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())
    if done:
        with g_ep.get_lock():
            if g_ep.value > 0 and g_ep.value % check_point_episodes == 0:
                torch.save(global_net.state_dict(),
                           f"{check_point_folder}/model_{int(time.time())}_gep_{g_ep.value}_w{rank}_{l_ep}_.pth")


def record(starting_num_candidate_acts, num_candidate_acts_decay_iter, global_ep, global_step, global_num_candidate_acts, global_ep_r, ep_r, res_queue, name,
           ep_step, ep_agent_num_acts):
    with global_step.get_lock():
        global_step.value += ep_step
        with global_num_candidate_acts.get_lock():
            global_num_candidate_acts.value = starting_num_candidate_acts - \
                math.floor(global_step.value / num_candidate_acts_decay_iter)
            if global_num_candidate_acts.value < 1:
                global_num_candidate_acts.value = 1
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
    logging.info(f"{name}_eps_reward|||{ep_r}")
    logging.info(f"{name}_eps_steps|||{ep_step}")
    logging.info(f"{name}_eps_agent_num_acts|||{ep_agent_num_acts}")
    logging.info(f"global_expected_returns|||{global_ep_r.value}")
    logging.info(f"num_candidate_acts|||{global_num_candidate_acts.value}")


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


def filter_action(action):
    impacts = action.impact_on_objects()
    if impacts['force_line']['changed'] and impacts['force_line']['disconnections']['count'] > 0:
        return False
    return True


def convert_obs(observation_space, obs, selected_attributes, scalers):
    vect = obs.to_vect()
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


def set_random_seeds(seed):
    """Sets all possible random seeds so results can be reproduced"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def create_action_mappings(env, all_actions, selected_action_types):
    action_tensors = []
    i = 0
    for act in all_actions:

        impacts = act.impact_on_objects()
        action_tensor = []
        if selected_action_types["switch_line"]:
            switch_line_tensor = np.zeros(env.n_line)
            switch_line_tensor[impacts['switch_line']['powerlines']] = 1
            action_tensor.append(switch_line_tensor)

        if selected_action_types["force_line_disconnect"]:
            force_line_disconnect_vector = np.zeros(env.n_line)
            force_line_disconnect_vector[impacts['force_line']
                                         ['disconnections']['powerlines']] = 1
            action_tensor.append(force_line_disconnect_vector)

        if selected_action_types["force_line_reconnect"]:
            force_line_reconnect_vector = np.zeros(env.n_line)
            force_line_reconnect_vector[impacts['force_line']
                                        ['reconnections']['powerlines']] = 1
            action_tensor.append(force_line_reconnect_vector)

        if selected_action_types["set_bus"]:
            set_bus_1_vector = np.zeros(env.dim_topo)
            set_bus_2_vector = np.zeros(env.dim_topo)
            for bus_assign in impacts['topology']['assigned_bus']:
                if bus_assign['bus'] == 1:
                    bus_vector = set_bus_1_vector
                else:
                    bus_vector = set_bus_2_vector

                obj_id = bus_assign['object_id']
                obj_type = bus_assign['object_type']

                pos_vect = get_topo_pos_vect(env, obj_type)
                bus_vector[pos_vect[obj_id]] = 1
            action_tensor.append(set_bus_1_vector)
            action_tensor.append(set_bus_2_vector)

        if selected_action_types["switch_bus"]:
            switch_bus_vector = np.zeros(env.dim_topo)
            for bus_switch in impacts['topology']['bus_switch']:
                obj_id = bus_switch['object_id']
                obj_type = bus_switch['object_type']
                pos_vect = get_topo_pos_vect(env, obj_type)
                switch_bus_vector[pos_vect[obj_id]] = 1
            action_tensor.append(switch_bus_vector)

        if selected_action_types["redispatch"]:
            redisp_vector = np.zeros(env.n_gen * 8)

            for redisp in impacts['redispatch']['generators']:
                obj_id = redisp['gen_id']
                dispatch_levels = np.linspace(
                    -env.gen_max_ramp_down[obj_id], env.gen_max_ramp_up[obj_id], 9)
                level = np.argwhere(
                    np.abs(dispatch_levels - redisp['amount']) < 0.01)
                if level > 4:
                    level = level - 1
                redisp_vector[obj_id * 8 + level] = 1
            action_tensor.append(redisp_vector)

        action_tensor = np.concatenate(action_tensor)

        if i == 0:
            action_tensor = 1 - action_tensor
        action_tensor = action_tensor / action_tensor.sum()
        action_tensors.append(action_tensor)
        i += 1
        if i % 1000 == 0:
            print(f"Action mappings: Processed {i} actions")

    return np.array(action_tensors)


def create_action_line_mappings(env, all_actions):
    action_line_mappings = np.zeros((len(all_actions), env.n_line))
    for i, a in enumerate(all_actions):
        impacts = a.impact_on_objects()
        if impacts['force_line']['changed']:
            reconn_lines = impacts['force_line']['reconnections']['powerlines']
            if len(reconn_lines) > 0:
                action_line_mappings[i, reconn_lines] = 1
        if impacts['topology']['changed']:
            set_bus_lines = [obj['object_id'] for obj in impacts['topology']
                             ['assigned_bus'] if obj['object_type'].startswith('line')]
            if len(set_bus_lines) > 0:
                action_line_mappings[i, set_bus_lines] = 1
        if i % 1000 == 0:
            print(f"Action line mappings: Processed {i} actions")
    return action_line_mappings


TRIVIAL = "Trivial"
GOOD = "Good"
BAD = "Bad"
DANGER = "Danger"


def forecast(action, env, obs):
    try:
        obs_do_nothing, _, done_do_nothing, _ = obs.simulate(
            env.action_space())
    except:
        obs_do_nothing = obs

    if done_do_nothing:
        obs_do_nothing = obs

    try:
        obs_forecasted, _, done_forecasted, _ = obs.simulate(action)
    except:
        obs_forecasted = obs_do_nothing

    if done_forecasted:
        obs_forecasted = obs_do_nothing

    return obs_do_nothing, obs_forecasted


def assess_impact(rho, rho_baseline, threshold_trivial=0.01, threshold_safe=0.95):
    impact = rho - rho_baseline
    impact_level = TRIVIAL
    if np.sum(np.logical_and(rho_baseline <= threshold_safe,
                             rho > threshold_safe)) > 0:  # if the action happens to change a power line from safe to danger -> this action is very bad
        impact_level = DANGER
    elif np.mean(np.abs(
            rho - rho_baseline)) <= threshold_trivial:  # The whole network with less then 1% movements each
        impact_level = TRIVIAL
    elif np.mean(impact) > 0:
        impact_level = BAD
    elif np.mean(impact) < 0:  # reducing line flow in general
        impact_level = GOOD

    return impact_level


def transform_rho(rho, min_threshold=0.02, normalised=True):
    rho_t = np.copy(rho)
    rho_t[rho_t == 0.0] = 1.01
    rho_t[rho_t > 1.0] = 1.01
    rho_t[rho_t < min_threshold] = 0.02
    rho_t = -np.log(1.02 - rho_t)
    return rho_t if not normalised else (rho_t / -np.log(0.01))


def compute_impact(rho1, rho2, min_threshold=0.02, normalised=True, eps=0.0000001):
    impact = transform_rho(rho1, min_threshold, normalised) - \
        transform_rho(rho2, min_threshold, normalised)
    return impact.sum() / ((impact != 0).sum() + eps)


def forecast_actions(actions, action_space, obs, min_threshold=0.9, normalised=True):
    try:
        obs_do_nothing, _, done_do_nothing, _ = obs.simulate(
            action_space.convert_act(0))
    except:
        obs_do_nothing = obs

    if done_do_nothing:
        obs_do_nothing = obs

    best_action = 0
    best_impact = 0
    best_obs = obs_do_nothing

    for action in actions[actions > 0]:
        try:
            obs_forecasted, _, done_forecasted, _ = obs.simulate(
                action_space.convert_act(action))
        except:
            obs_forecasted = obs_do_nothing

        if done_forecasted:
            obs_forecasted = obs_do_nothing

        impact = compute_impact(obs_forecasted.rho, obs_do_nothing.rho,
                                min_threshold=min_threshold, normalised=normalised)

        if impact < best_impact:
            best_action = action
            best_impact = impact
            best_obs = obs_forecasted

    return best_action, best_obs, obs_do_nothing


def lreward(action, env, obs_previous, obs_do_nothing, obs_forecasted, obs_current, done, info,
            threshold_trivial=0.0, threshold_safe=0.9, eps=0.000001):

    action_impact = compute_impact(
        obs_forecasted.rho, obs_do_nothing.rho, min_threshold=threshold_safe)
    situation_impact = compute_impact(
        obs_current.rho, obs_forecasted.rho, min_threshold=threshold_safe)
    outcome_impact = compute_impact(
        obs_current.rho, obs_do_nothing.rho, min_threshold=threshold_safe)

    if done:
        r = -0.02 if len(info["exception"]) > 0 else threshold_trivial
    else:
        r = -np.mean(outcome_impact) if np.mean(np.abs(action_impact)
                                                ) > threshold_trivial else threshold_trivial

    # print("act:{},r: {},action[rho=({:f},{:f},{:f})],situation[rho=({:f},{:f},{:f})],outcome[rho=({:f},{:f},{:f})]".
    #      format(action, r, np.min(action_impact), np.mean(action_impact), np.max(action_impact), np.min(situation_impact), np.mean(situation_impact),
    #             np.max(situation_impact), np.min(outcome_impact), np.mean(outcome_impact), np.max(outcome_impact)))

    return r
