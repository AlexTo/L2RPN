from __future__ import division

import logging
import os

import torch
from grid2op.Converter import IdToAct
from grid2op.Environment import MultiMixEnvironment
from setproctitle import setproctitle as ptitle
from torch.autograd import Variable

from beu_l2rpn.agent import Agent
from beu_l2rpn.model import A3C
from beu_l2rpn.shared_optim import SharedRMSprop, SharedAdam
from beu_l2rpn.utils import ensure_shared_grads, create_env, setup_worker_logging


def train(rank, config, shared_model, optimizer, obs_idx, state_size, global_eps, log_queue):
    ptitle('Training Agent: {}'.format(rank))

    setup_worker_logging(log_queue)

    gpu_id = config["gpu_ids"][rank % len(config["gpu_ids"])]
    seed = config["seed"] + rank
    torch.manual_seed(seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(seed)

    env = create_env(config["env"], seed)
    action_space = IdToAct(env.action_space)
    action_space.init_converter(all_actions=os.path.join("data", f"{config['env']}_action_space.npy"))

    if optimizer is None:
        if config["optimizer"] == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=config["lr"])
            optimizer.share_memory()
        if config["optimizer"] == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=config["lr"], amsgrad=True)
            optimizer.share_memory()

    model = A3C(state_size, action_space.size())
    agent = Agent(model, env, config, obs_idx, action_space, None)
    agent.gpu_id = gpu_id

    if isinstance(agent.env, MultiMixEnvironment):
        s = agent.env.reset(random=True)
    else:
        s = agent.env.reset()

    agent.state = agent.convert_obs(s)
    agent.state = torch.from_numpy(agent.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            agent.state = agent.state.cuda()
            agent.model = agent.model.cuda()
    agent.model.train()

    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                agent.model.load_state_dict(shared_model.state_dict())
        else:
            agent.model.load_state_dict(shared_model.state_dict())

        for step in range(config["n_steps"]):
            act = agent.action_train()
            logging.info(f"Agent_{rank}_act|||{act}")

            if agent.done:
                break

        if agent.done:
            if isinstance(agent.env, MultiMixEnvironment):
                s = agent.env.reset(random=True)
            else:
                s = agent.env.reset()

            state = agent.convert_obs(s)
            agent.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    agent.state = agent.state.cuda()

            with global_eps.get_lock():
                global_eps.value += 1

        R = torch.zeros(1, 1)
        if not agent.done:
            value, _ = agent.model(Variable(agent.state.unsqueeze(0)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        agent.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(agent.rewards))):
            R = config["gamma"] * R + agent.rewards[i]
            advantage = R - agent.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = agent.rewards[i] + config["gamma"] * agent.values[i + 1].data - agent.values[i].data

            gae = gae * config["gamma"] * config["tau"] + delta_t

            policy_loss = policy_loss - agent.log_probs[i] * Variable(gae) - 0.01 * agent.entropies[i]

            logging.info(f"Agent_{rank}_policy_loss|||{policy_loss.item()}")
            logging.info(f"Agent_{rank}_value_loss|||{value_loss.item()}")

        agent.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(agent.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        agent.clear_actions()
