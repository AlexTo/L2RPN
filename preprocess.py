import json
import os
import numpy as np
import grid2op
from grid2op.Converter import IdToAct
from lightsim2grid.LightSimBackend import LightSimBackend

from utils import create_action_mappings, create_action_line_mappings, filter_action

if __name__ == '__main__':
    with open("data/config.json", 'r') as f:
        config = json.load(f)

    env = grid2op.make(config["env"], backend=LightSimBackend())
    env.seed(config["seed"])

    selected_action_types = config["selected_action_types"]

    if os.path.exists(os.path.join("data", f"{config['env']}_action_space.npy")):
        action_space = IdToAct(env.action_space)
        action_space.init_converter(all_actions=os.path.join(
            "data", f"{config['env']}_action_space.npy"))
    else:
        action_space = IdToAct(env.action_space)
        action_space.init_converter(
            set_line_status=(selected_action_types["force_line_reconnect"]
                             or selected_action_types["force_line_disconnect"]),
            change_line_status=selected_action_types["switch_line"],
            set_topo_vect=selected_action_types["set_bus"],
            change_bus_vect=selected_action_types["switch_bus"],
            redispatch=selected_action_types["redispatch"])
        action_space.filter_action(filter_action)
        action_space.save("data", f"{config['env']}_action_space.npy")

    if not os.path.exists(os.path.join("data", f"{config['env']}_action_mappings.npy")):
        action_mappings = create_action_mappings(
            env, action_space.all_actions, config["selected_action_types"])
        np.save(os.path.join(
            "data", f"{config['env']}_action_mappings.npy"), action_mappings.T)

    if not os.path.exists(os.path.join("data", f"{config['env']}_action_line_mappings.npy")):
        action_line_mappings = create_action_line_mappings(
            env, action_space.all_actions)
        np.save(os.path.join(
            "data", f"{config['env']}_action_line_mappings.npy"), action_line_mappings.T)
