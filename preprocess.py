import json
import os
import pickle
import time
from multiprocessing import Manager, Process

import joblib
import numpy as np
from grid2op.Converter import IdToAct

from beu_l2rpn.utils import create_action_mappings, create_env


def gen_experience(rank, config, num_exp):
    file_name = os.path.join("data", f"{config['env']}_replay_buffer_{rank}.pickle")
    if os.path.exists(file_name):
        return
    start_time = time.time()
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

            if done:
                if len(info['exception']) > 0:
                    r = 0
                else:
                    r = 1.0

            experiences.append((s.to_vect(), act, r, s2.to_vect(), done))
            s = s2
            i += 1
            if i % 1000 == 0 or i >= num_exp:
                elapsed_time = time.time() - start_time
                remaining = (elapsed_time / i) * (num_exp - i)
                print(
                    f'Gen_exp_{rank}: {i}/{num_exp} experiences generated. Estimated time remaining '
                    f'{time.strftime("%H:%M:%S", time.gmtime(remaining))}')

    joblib.dump(experiences, file_name, compress=True, protocol=pickle.HIGHEST_PROTOCOL)


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

    num_exp = config["num_experience_gen"]

    processes = []
    for i in range(config["num_experience_gen_processes"]):
        p = Process(target=gen_experience, args=(i, config, num_exp / config["num_experience_gen_processes"]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
