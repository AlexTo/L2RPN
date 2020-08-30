import numpy as np

from grid2op.Reward import L2RPNReward


class BeUReward(L2RPNReward):
    def initialize(self, env):
        self.reward_min = 0.0
        self.reward_max = 1.0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            # previous action was bad
            res = self.reward_min
        elif is_done:
            # really strong reward if an episode is over without game over
            res = self.reward_max
        else:
            res = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
            res /= env.n_line
            if not np.isfinite(res):
                res = self.reward_min
        return res
