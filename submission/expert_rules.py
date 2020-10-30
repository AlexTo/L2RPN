import numpy as np


def reconnect_maintained_lines(maintenance_list, ep_step, action_space, obs):
    act = None
    lines = np.argwhere((maintenance_list <= ep_step)
                        & (maintenance_list > 0))
    if len(lines) == 0:
        return None, None

    line_id = lines[0][0]
    if not obs.line_status[line_id]:
        act = action_space({
            "set_line_status": [(line_id, 1)],
            "set_bus": {"lines_or_id": [(line_id, 1)], "lines_ex_id": [(line_id, 1)]}
        })  # always reconnect to bus 1 for now. We can't omit the set_bus part because then we can't find the act id from all_actions
        maintenance_list[line_id] = -1
        #print(f"{name} - expert act: reconnecting line {line_id}")
        print(f"Reconnecting line {line_id} with status {obs.line_status[line_id]}")
    return act, line_id


def expert_rules(maintenance_list, ep_step, action_space, obs):
    act = None

    act = reconnect_maintained_lines(
        maintenance_list, ep_step, action_space, obs)

    if act is not None:
        return act

    return act
