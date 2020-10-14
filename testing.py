import grid2op
from grid2op.Converter import IdToAct

from lightsim2grid.LightSimBackend import LightSimBackend
from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float


class MyReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(-1.)
        self.reward_max = dt_float(1.)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        print(action)
        print(f"has_error | is_done | is_illegal | is_ambiguous : {has_error} {is_done} {is_illegal} {is_ambiguous}")
        return self.reward_max


def test():
    e = grid2op.make("l2rpn_case14_sandbox", reward_class=MyReward, backend=LightSimBackend())
    e.reset()

    action_space = IdToAct(e.action_space)
    action_space.init_converter()

    print("===========================================")
    print("The above two actions are printed by the environment, ignore them")
    print("===========================================")

    disc_line_0 = action_space.disconnect_powerline(line_id=0)  # disconnect powerline 0
    obs, _, done, info = e.step(disc_line_0)

    print("===========================================")

    recon_line_0 = action_space.reconnect_powerline(bus_or=1, bus_ex=1, line_id=1)
    obs, _, done, info = e.step(disc_line_0)


if __name__ == '__main__':
    test()
