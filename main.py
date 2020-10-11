from __future__ import print_function, division

import json
import os
import neptune
from grid2op.Converter import IdToAct
from beu_l2rpn.model import A3C
from beu_l2rpn.shared_optim import SharedAdam, SharedRMSprop
from beu_l2rpn.test_func import test
from beu_l2rpn.train_func import train
from beu_l2rpn.utils import create_env, init_obs_extraction, setup_main_logging

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp
import time

if __name__ == '__main__':

    with open(os.path.join("data", "config.json"), 'r') as f:
        config = json.load(f)

    torch.manual_seed(config["seed"])
    if config["use_gpu"]:
        torch.cuda.manual_seed(config["seed"])
        mp.set_start_method('spawn')

    check_point_folder = os.path.join(config["check_point_folder"], config["env"])

    if not os.path.exists(check_point_folder):
        os.makedirs(check_point_folder)

    env = create_env(config["env"], config["seed"])

    obs_idx, state_size = init_obs_extraction(env.observation_space, config["selected_attributes"])

    action_space = IdToAct(env.action_space)
    action_space.init_converter(all_actions=os.path.join("data", f"{config['env']}_action_space.npy"))

    shared_model = A3C(state_size, action_space.size())

    shared_model.share_memory()

    optimizer = None
    if config["shared_optimizer"]:
        if config["optimizer"] == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=config["lr"])
            optimizer.share_memory()
        if config["optimizer"] == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=config["lr"], amsgrad=True)
            optimizer.share_memory()

    global_eps = mp.Value('i', 0)

    processes = []

    log_queue = setup_main_logging(config)

    p = mp.Process(target=test, args=(config, shared_model, obs_idx, state_size, log_queue))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, config["nb_workers"]):
        p = mp.Process(
            target=train, args=(rank, config, shared_model, optimizer, obs_idx, state_size, global_eps, log_queue))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()

    log_queue.put_nowait(None)
