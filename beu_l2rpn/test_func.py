from __future__ import division

import logging
import os
import time

import grid2op
import torch
from grid2op.Converter import IdToAct
from lightsim2grid.LightSimBackend import LightSimBackend
from setproctitle import setproctitle as ptitle

from beu_l2rpn.agent import Agent
from beu_l2rpn.model import A3C
from beu_l2rpn.utils import setup_worker_logging, create_env


def test(config, shared_model, obs_idx, state_size, log_queue):
    ptitle('Test Agent')
    setup_worker_logging(log_queue)
    gpu_id = config["gpu_ids"][-1]
    seed = config["seed"]
    torch.manual_seed(seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(seed)

    save_dir = os.path.join(config["check_point_folder"], config["env"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = create_env(config["env"], seed)

    action_space = IdToAct(env.action_space)
    action_space.init_converter(all_actions=os.path.join("data", f"{config['env']}_action_space.npy"))

    reward_sum = 0
    num_tests = 0
    reward_total_sum = 0
    agent = Agent(None, env, config, obs_idx, action_space, None)
    agent.gpu_id = gpu_id
    agent.model = A3C(state_size, action_space.size())

    agent.state = agent.env.reset()
    agent.state = agent.convert_obs(agent.state)
    agent.eps_len += 2
    agent.state = torch.from_numpy(agent.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            agent.model = agent.model.cuda()
            agent.state = agent.state.cuda()
    flag = True

    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    agent.model.load_state_dict(shared_model.state_dict())
            else:
                agent.model.load_state_dict(shared_model.state_dict())
            agent.model.eval()
            flag = False

        act = agent.action_test()
        logging.info(f"Eval_act|||{act}")
        reward_sum += agent.reward

        if agent.done:
            state = agent.env.reset()
            state = agent.convert_obs(state)
            agent.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    agent.state = agent.state.cuda()

            flag = True
            num_tests += 1
            reward_total_sum += reward_sum

            if num_tests > 0 and num_tests % config["eval_freq"] == 0:
                save_path = os.path.join(config["check_point_folder"], config["env"],
                                         f"model_{int(time.time())}_eps_{num_tests}.pth")
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        torch.save(agent.model.state_dict(), save_path)
                else:
                    torch.save(agent.model.state_dict(), save_path)

            logging.info(f"Eval_eps_rewards|||{reward_sum}")
            logging.info(f"Eval_eps_steps_survived|||{agent.eps_len}")

            reward_sum = 0
            agent.eps_len = 0
            time.sleep(10)
