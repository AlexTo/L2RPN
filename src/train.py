import json

import grid2op
from grid2op.Reward import LinesReconnectedReward, CloseToOverflowReward, RedispReward, CombinedScaledReward, \
    L2RPNReward

from rewards.game_play_reward import GameplayReward
from src.agent import BeUAgent


def train(env, conf):
    agent = BeUAgent(env, conf)
    agent.train(num_episodes=10000)


if __name__ == "__main__":
    with open('config.json') as json_file:
        config = json.load(json_file)

    environment = grid2op.make(config["env"], reward_class=CombinedScaledReward)

    # Register custom reward for training
    cr = environment.reward_helper.template_reward
    cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 1.0)
    cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    cr.addReward("l2rpn", L2RPNReward(), 2.0 / float(environment.n_line))
    # Initialize custom rewards
    cr.set_range(-1.0, 1.0)
    cr.initialize(environment)

    train(environment, config)
