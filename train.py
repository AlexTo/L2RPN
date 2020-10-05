import json
import os

import grid2op
import torch.multiprocessing as mp
import torch
import numpy as np
from grid2op.Converter import IdToAct

from beu_l2rpn.a3c import Net, Worker

from beu_l2rpn.shared_adam import SharedAdam
from beu_l2rpn.utils import init_obs_extraction, shuffle_env_chronics

os.environ["OMP_NUM_THREADS"] = "1"


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


def train():
    with open('config.json') as json_file:
        config = json.load(json_file)
    if config["use_gpu"] and torch.cuda.is_available():
        mp.set_start_method('spawn')

    hyper_parameters = config["hyper_parameters"]
    env = create_env(config["env"], config["seed"])
    observation_space = env.observation_space
    action_space_converter_class = IdToAct.init_grid(env.action_space)
    action_space = action_space_converter_class(env.action_space)
    action_space.init_converter()
    obs_idx, obs_size = init_obs_extraction(observation_space, hyper_parameters['selected_attributes'])

    state_size = observation_space.size()
    action_size = action_space.size()
    all_actions = np.array(action_space.all_actions)

    global_net = Net(state_size, action_size)
    global_net.share_memory()
    opt = SharedAdam(global_net.parameters(), lr=hyper_parameters["learning_rate"])  # global optimizer

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [
        Worker(global_net=global_net, opt=opt, global_ep=global_ep, global_ep_r=global_ep_r,
               res_queue=res_queue, worker_id=i, env=create_env(config["env"], config["seed"] + i),
               nb_episodes=hyper_parameters["train_num_episodes"],
               state_size=state_size, action_size=action_size,
               update_global_iter=hyper_parameters["update_global_iter"],
               gamma=hyper_parameters["discount_rate"], all_actions=all_actions,
               selected_attributes=hyper_parameters["selected_attributes"], obs_idx=obs_idx,
               feature_scalers=hyper_parameters["feature_scalers"])
        for i in range(hyper_parameters["num_workers"])]

    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    torch.save(global_net.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
