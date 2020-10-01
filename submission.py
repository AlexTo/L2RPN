import json
import os
import sys


def make_agent(env, submission_dir):
    sys.path.insert(0, submission_dir)
    from beu_l2rpn.agent import BeUAgent
    model_path = os.path.join(submission_dir, "saved_model")
    with open(os.path.join(model_path, 'config.json')) as json_file:
        config = json.load(json_file)
    agent = BeUAgent(env, config, training=False)
    agent.load_model(os.path.join(model_path, "model.pth"))
    return agent
