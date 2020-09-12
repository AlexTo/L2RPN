import json
import os
import sys


def make_agent(env, submission_dir):
    sys.path.insert(0, submission_dir)
    from beu_l2rpn.agent import BeUAgent
    with open(os.path.join(submission_dir, 'config.json')) as json_file:
        config = json.load(json_file)
    agent = BeUAgent(env, config)
    return agent
