import numpy as np


def has_overflow(obs):
    return np.any(obs.rho > 1.0)


def reconnect_lines(action_space, obs):
    line_status = obs.line_status

    if not (False in line_status):
        return None

    disconnected_lines = np.where(obs.line_status == 0)[0]
    for line_id in disconnected_lines:
        if obs.time_before_cooldown_line[line_id] == 0:
            return action_space({"set_line_status": [(line_id, 1)]})

    return None
