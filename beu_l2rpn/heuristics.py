import numpy as np


def try_reconnect_power_line(action_space, s, episode_broken_lines):
    act = None

    zero_rhos = np.where(s.rho <= 0)[0]

    for line_id in zero_rhos:
        if line_id in episode_broken_lines:
            episode_broken_lines[line_id] += 1
            if episode_broken_lines[line_id] > 10:
                episode_broken_lines[line_id] = 10
        else:
            episode_broken_lines[line_id] = 1

    for line_id in episode_broken_lines:
        if line_id not in zero_rhos:
            episode_broken_lines[line_id] = 0

    for line in episode_broken_lines:
        timesteps_after_broken = episode_broken_lines[line]
        if timesteps_after_broken == 10 and s.time_before_cooldown_line[line] == 0:
            for o, e in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                propose_act = action_space.reconnect_powerline(line_id=line, bus_or=o, bus_ex=e)
                sim_obs, sim_reward, sim_done, info = s.simulate(propose_act)
                if not sim_done:
                    act = propose_act
                    break
            episode_broken_lines[line] = 0
            break
    return act
