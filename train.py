import json

from beu_l2rpn.agent import BeUAgent
from beu_l2rpn.utils import create_env


def train(env, conf):
    agent = BeUAgent(env, conf)
    agent.train()


if __name__ == "__main__":
    with open('data/config.json') as json_file:
        config = json.load(json_file)

    environment = create_env(config["env"], config["seed"])

    train(environment, config)
