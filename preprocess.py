import json
import os
import pickle
from multiprocessing import Manager, Process

import joblib
import numpy as np
from grid2op.Converter import IdToAct

from beu_l2rpn.utils import create_action_mappings, create_env


def gen_experience(rank, config, num_exp):
    env = create_env(config["env"], config["seed"] + rank)
    action_space = IdToAct(env.action_space)
    action_space.init_converter(all_actions=os.path.join("data", f"{config['env']}_action_space.npy"))
    action_size = action_space.size()
    experiences = []
    i = 0
    while i < num_exp:
        done = False
        s = env.reset(random=True)
        while not done:
            act = np.random.randint(0, action_size)
            s2, r, done, info = env.step(action_space.convert_act(act))
            experiences.append((s.to_vect(), act, r, s2.to_vect(), done))
            i += 1
            if i % 1000 == 0:
                print(f'Gen_exp_{rank}: {i}/{num_exp} experiences generated.')

    joblib.dump(experiences, os.path.join("data", f"{config['env']}_replay_buffer_{rank}.pickle"), compress=True,
                protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    with open("data/config.json", 'r') as f:
        config = json.load(f)

    if not os.path.exists(os.path.join("data", f"{config['env']}_action_space.npy")):
        env = create_env(config["env"], config["seed"])
        action_space = IdToAct(env.action_space)
        action_space.init_converter()
        action_space.save("data", f"{config['env']}_action_space.npy")

    if not os.path.exists(os.path.join("data", f"{config['env']}_action_mappings.npy")):
        action_mappings = create_action_mappings(env, action_space.all_actions, config["selected_action_types"])
        np.save(os.path.join("data", f"{config['env']}_action_mappings.npy"), action_mappings.T)

    num_exp = config["min_steps_before_learning"]

    processes = []
    for i in range(config["experience_gen_num_processes"]):
        p = Process(target=gen_experience, args=(i, config, num_exp / config["experience_gen_num_processes"]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
