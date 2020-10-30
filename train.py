import json
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from grid2op.Converter import IdToAct

from a3c import Net, Agent
from shared_adam import SharedAdam
from utils import create_env, setup_main_logging, cuda

os.environ["OMP_NUM_THREADS"] = "1"


def train():
    with open('data/config.json') as json_file:
        config = json.load(json_file)

    # This will train on CPU with no error if the 2 lines below are commented. However, we need to set start mode to
    # spawn to train on CUDA
    if config["use_gpu"] and torch.cuda.is_available():
        mp.set_start_method('spawn')

    log_queue = setup_main_logging(config)

    check_point_folder = os.path.join(
        config["check_point_folder"], config["env"])
    if not os.path.exists(check_point_folder):
        os.makedirs(check_point_folder)

    env = create_env(config["env"], config["seed"])

    state_size = config["state_size"]

    with open(os.path.join("data", f"{config['env']}_action_mappings.npz"), 'rb') as f:
        archive = np.load(f)
        action_mappings = np.float32(archive[archive.files[0]])

    with open(os.path.join("data", f"{config['env']}_action_line_mappings.npz"), 'rb') as f:
        archive = np.load(f)
        action_line_mappings = np.float32(archive[archive.files[0]])

    action_mappings_tensors = []
    action_line_mappings_tensors = []
    for gpu_id in config["gpu_ids"]:
        action_mappings_copy = np.copy(action_mappings)
        action_mappings_tensor = cuda(gpu_id, torch.tensor(
            action_mappings_copy, requires_grad=False))
        action_mappings_tensors.append(action_mappings_tensor)

        action_line_mappings_copy = np.copy(action_line_mappings)
        action_line_mappings_tensor = cuda(gpu_id, torch.tensor(
            action_line_mappings_copy, requires_grad=False))
        action_line_mappings_tensors.append(action_line_mappings_tensor)

    global_net = Net(state_size, torch.tensor(action_mappings, requires_grad=False),
                     torch.tensor(action_line_mappings, requires_grad=False))

    if os.path.exists(config["load_model"]):
        global_net.load_state_dict(torch.load(config["load_model"]))

    global_net.share_memory()
    opt = SharedAdam(global_net.parameters(),
                     lr=config["learning_rate"])  # global optimizer

    global_step, global_ep, global_ep_r, res_queue, g_num_candidate_acts = mp.Value('i', 0), mp.Value(
        'i', 0), mp.Value('d', 0.), mp.Queue(), mp.Value('i', config["starting_num_candidate_acts"])

    agents = [
        Agent(global_net=global_net, opt=opt, global_ep=global_ep, global_step=global_step, global_ep_r=global_ep_r, res_queue=res_queue,
              global_num_candidate_acts=g_num_candidate_acts, rank=i, config=config, log_queue=log_queue,
              action_mappings=action_mappings_tensors[i % len(
                  config["gpu_ids"])],
              action_line_mappings=action_line_mappings_tensors[i % len(config["gpu_ids"])])

        for i in range(config["num_workers"])]

    [agent.start() for agent in agents]

    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in agents]
    torch.save(global_net.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
