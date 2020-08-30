import torch
import grid2op
from training_params import TrainingParam
from lightsim2grid import LightSimBackend

from agent import BeUAgent
from utils import args_parser
from reward import BeUReward


def main():
    backend = LightSimBackend()
    args = args_parser().parse_args()

    env = grid2op.make(args.env_name,
                       reward_class=BeUReward,
                       backend=backend)

    training_params = init_training_params()

    agent = BeUAgent(env, args, training_params)

    try:
        agent.train()
    finally:
        env.close()


def init_training_params():
    training_params = TrainingParam()
    training_params.buffer_size = 1000000
    return training_params


if __name__ == "__main__":
    main()
