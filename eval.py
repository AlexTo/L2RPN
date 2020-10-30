import os
import json
import torch
import grid2op
import numpy as np
from grid2op.Converter import IdToAct
from lightsim2grid import LightSimBackend

from grid2op.Runner import Runner
from submission.agent import Agent

def evaluate(env,
             model_name=".",
             save_path=None,
             logs_path=None,
             nb_episode=1,
             nb_process=1,
             max_steps=-1,
             verbose=False,
             save_gif=False,
             **kwargs):

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    with open(os.path.join("data", "config.json"), 'r') as f:
        config = json.load(f)

    env_name = config["env"]

    with open(os.path.join("data", f"{env_name}_action_mappings.npz"), 'rb') as f:
        archive = np.load(f)
        action_mappings = np.float32(archive[archive.files[0]])

    with open(os.path.join("data", f"{env_name}_action_line_mappings.npz"), 'rb') as f:
        archive = np.load(f)
        action_line_mappings = np.float32(archive[archive.files[0]])

    action_space = IdToAct(env.action_space)

    with open(os.path.join("data", f"{env_name}_action_space.npz"), 'rb') as f:
        archive = np.load(f)
        action_space.init_converter(all_actions=archive[archive.files[0]])

    agent = Agent(env, config, action_space,
                  action_mappings, action_line_mappings)
    agent.load(os.path.join("submission", "data", "model.pth"))

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # you can do stuff with your model here

    # start the runner
    res = runner.run(path_save=save_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=False)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(
            nb_time_step, max_ts)
        print(msg_tmp)


if __name__ == "__main__":

    env = grid2op.make("data/input_data_local", backend=LightSimBackend())

    evaluate(env,
             model_name="beu",
             save_path="eval_results",
             logs_path="logs-train",
             nb_episode=24,
             nb_process=1,
             max_steps=2016,
             verbose=True,
             save_gif="eval_results")
