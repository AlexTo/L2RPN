import json

import grid2op
from grid2op.Environment import MultiMixEnvironment
from grid2op.Reward import CombinedScaledReward, CloseToOverflowReward, LinesReconnectedReward, \
    L2RPNReward, RedispReward

from beu_l2rpn.agent import BeUAgent
from beu_l2rpn.rewards.game_play_reward import GameplayReward
from beu_l2rpn.utilities.utility_functions import shuffle


def train(env, conf):
    agent = BeUAgent(env, conf)
    agent.train()


if __name__ == "__main__":
    with open('config.json') as json_file:
        config = json.load(json_file)

    environment = grid2op.make(config["env"], reward_class=CombinedScaledReward)

    if isinstance(environment, MultiMixEnvironment):
        for mix in environment:
            mix.chronics_handler.shuffle(shuffler=shuffle)
    else:
        environment.chronics_handler.shuffle(shuffler=shuffle)

    # Register custom reward for training
    cr = environment.reward_helper.template_reward
    cr.addReward("redisp", RedispReward(), 1.0)
    cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 1.0)
    cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    cr.addReward("l2rpn", L2RPNReward(), 2.0 / float(environment.n_line))
    # Initialize custom rewards
    cr.set_range(-50.0, 1.0)
    cr.initialize(environment)

    train(environment, config)
