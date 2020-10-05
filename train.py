import json

from beu_l2rpn.agent import BeUAgent


def train():
    with open('config.json') as json_file:
        config = json.load(json_file)

    agent = BeUAgent(config=config)
    agent.train()


if __name__ == "__main__":
    train()
