import json
import os
import numpy as np
import grid2op
from grid2op.Converter import IdToAct
from lightsim2grid.LightSimBackend import LightSimBackend

from utils import create_action_mappings

if __name__ == '__main__':
    with open("config.json", 'r') as f:
        config = json.load(f)

    env = grid2op.make(config["env"], backend=LightSimBackend())
    env.seed(config["seed"])

    action_space = IdToAct(env.action_space)
    action_space.init_converter()

    if not os.path.exists(os.path.join("data", f"{config['env']}_action_space.npy")):
        action_space.save("data", f"{config['env']}_action_space.npy")

    if not os.path.exists(os.path.join("data", f"{config['env']}_action_mappings.npy")):
        action_mappings = create_action_mappings(env, action_space.all_actions, config["selected_action_types"])
        np.save(os.path.join("data", f"{config['env']}_action_mappings.npy"), action_mappings.T)
