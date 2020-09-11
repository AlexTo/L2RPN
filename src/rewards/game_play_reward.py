from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float


class GameplayReward(BaseReward):
    """
    This rewards is strictly computed based on the Game status.
    It yields a negative reward in case of game over.
    A half negative reward on rules infringment.
    Otherwise the reward is positive.
    """

    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(-50.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error:
            return self.reward_min
        elif is_illegal or is_ambiguous:
            # Did not respect the rules
            return self.reward_min / dt_float(2.0)
        else:
            # Keep playing or finished episode
            return self.reward_max
