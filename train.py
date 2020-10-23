import json
import os

import numpy as np

from beu_l2rpn.agent import BeUAgent
from beu_l2rpn.data_structures import ReplayBuffer
from beu_l2rpn.utils import create_env


def train():
    with open('data/config.json') as json_file:
        config = json.load(json_file)

    env = create_env(config["env"], config["seed"])
    replay_buffer = ReplayBuffer(config["buffer_size"], config["batch_size"], config["seed"])

    print("Loading replay buffer...")
    replay_buffer.load(os.path.join("data", f"{config['env']}_replay_buffer.npy"))

    with open(os.path.join("data", f"{config['env']}_action_mappings.npy"), 'rb') as f:
        print(f"Loading action mappings matrix...")
        action_mappings_matrix = np.load(f)

    agent = BeUAgent(env, config, action_mappings_matrix, replay_buffer)
    agent.train()


if __name__ == "__main__":
    train()
