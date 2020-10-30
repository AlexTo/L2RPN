import json
import os
import sys
import numpy as np
import json
from grid2op.Converter import IdToAct
from .agent import Agent

def make_agent(env, submission_dir):

    with open(os.path.join(submission_dir, "data", "config.json"), 'r') as f:
        config = json.load(f)

    env_name = config["env"]

    with open(os.path.join(submission_dir, "data", f"{env_name}_action_mappings.npz"), 'rb') as f:
        archive = np.load(f)
        action_mappings = np.float32(archive[archive.files[0]])

    with open(os.path.join(submission_dir, "data", f"{env_name}_action_line_mappings.npz"), 'rb') as f:
        archive = np.load(f)
        action_line_mappings = np.float32(archive[archive.files[0]])

    action_space = IdToAct(env.action_space)

    with open(os.path.join(submission_dir, "data", f"{env_name}_action_space.npz"), 'rb') as f:
        archive = np.load(f)
        action_space.init_converter(all_actions=archive[archive.files[0]])

    agent = Agent(env, config, action_space, action_mappings, action_line_mappings)
    agent.load(os.path.join(submission_dir, "data", "model.pth"))
    return agent