import json
import os

import torch.multiprocessing as mp
import torch
from beu_l2rpn.agent import BeUAgent

os.environ["OMP_NUM_THREADS"] = "1"


def train():
    with open('config.json') as json_file:
        config = json.load(json_file)
    if config["use_gpu"] and torch.cuda.is_available():
        mp.set_start_method('spawn')
    agent = BeUAgent(config=config)
    agent.train()


if __name__ == "__main__":
    train()
