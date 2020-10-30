import json
import os
import sys
import numpy as np
import json
from grid2op.Converter import IdToAct


def make_agent(env, submission_dir):

    with open(os.path.join(submission_dir, "data", "config.json"), 'r') as f:
        config = json.load(f)

    env_name = config["env"]

    with open(os.path.join(submission_dir, "data", f"{env_name}_action_mappings.npy"), 'rb') as f:
        action_mappings = np.float32(np.load(f))

    with open(os.path.join(submission_dir, "data", f"{env_name}_action_line_mappings.npy"), 'rb') as f:
        action_line_mappings = np.float32(np.load(f))

    action_space = IdToAct(env.action_space)
    action_space.init_converter(all_actions=os.path.join(
        submission_dir, "data", f"{env_name}_action_space.npz"))
