import grid2op
from grid2op.Reward import CombinedReward, RedispReward, EconomicReward, CloseToOverflowReward, \
    GameplayReward, LinesReconnectedReward, L2RPNReward
from lightsim2grid.LightSimBackend import LightSimBackend

def test():
    env = grid2op.make("l2rpn_neurips_2020_track2_small", reward_class=CombinedReward,
                       backend=LightSimBackend())
    cr = env.get_reward_instance()
    cr.addReward("redisp", RedispReward(), 0.1)
    cr.addReward("economic", EconomicReward(), 0.1)
    cr.addReward("overflow", CloseToOverflowReward(), 0.1)
    cr.addReward("gameplay", GameplayReward(), 0.1)
    cr.addReward("recolines", LinesReconnectedReward(), 0.1)
    cr.addReward("l2rpn", L2RPNReward(), .6 / float(env.n_line))
    cr.initialize(env)
    env.seed(2020)
    print(f"Rewards num: {len(env.get_reward_instance().rewards)}")
    env.reset(random=True)
    print(f"Rewards num: {len(env.get_reward_instance().rewards)}")
if __name__ == '__main__':
    test()