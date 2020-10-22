import json
import os

import joblib
import numpy as np

from beu_l2rpn.agent import BeUAgent
from beu_l2rpn.data_structures import ReplayBuffer
from beu_l2rpn.utils import create_env, convert_obs


def train():
    with open('data/config.json') as json_file:
        config = json.load(json_file)

    env = create_env(config["env"], config["seed"])
    replay_buffer = ReplayBuffer(config["buffer_size"], config["batch_size"], config["seed"])

    i = 0
    while True:
        exp_file = os.path.join("data", f"{config['env']}_replay_buffer_{i}.pickle")
        if os.path.exists(exp_file):
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
        else:
            break
        i += 1

    with open(os.path.join("data", f"{config['env']}_action_mappings.npy"), 'rb') as f:
        action_mappings_matrix = np.load(f)

    agent = BeUAgent(env, config, action_mappings_matrix, replay_buffer)
    agent.train()


if __name__ == "__main__":
    train()
