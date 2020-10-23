import json
import os
import pickle
import time
from multiprocessing import Manager, Process

import joblib
import numpy as np
from grid2op.Converter import IdToAct

from beu_l2rpn.data_structures import ReplayBuffer
from beu_l2rpn.utils import create_action_mappings, create_env, convert_obs


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

    env = create_env(config["env"], config["seed"])

    if not os.path.exists(os.path.join("data", f"{config['env']}_action_space.npy")):
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

    print("Converting obs vectors and merging files...")

    replay_buffer = ReplayBuffer(config["buffer_size"], config["batch_size"], config["seed"])

    i = int(0)
    num_experiences = 0
    while True:
        exp_file = os.path.join("data", f"{config['env']}_replay_buffer_{i}.pickle")
        if os.path.exists(exp_file):
            print(f"Reading {exp_file}")
            with open(exp_file, 'rb') as f:
                expriences = joblib.load(f)
                for exp in expriences:
                    s, a, r, s2, done = exp

                    # convert vector
                    s = convert_obs(env.observation_space, s, config["selected_attributes"], config["feature_scalers"],
                                    True)
                    s2 = convert_obs(env.observation_space, s2, config["selected_attributes"],
                                     config["feature_scalers"], True)
                    replay_buffer.add_experience(s, a, r, s2, done)
                    num_experiences += 1
                    if num_experiences % 1000 == 0:
                        print(f"Converted and merged {num_experiences} experiences")

        else:
            break
        i += 1
    replay_buffer.save(os.path.join("data", f"{config['env']}_replay_buffer.pickle"))
