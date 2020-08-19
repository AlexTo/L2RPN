import os
from .DeepQ_NNParam import DeepQ_NNParam
from .DeepQSimple import DeepQSimple
from .MyReward import MyReward

name = "DeepQSimple"


class reward(MyReward):
    def __init__(self):
        MyReward.__init__(self)

    def __call__(self, *args, **kwargs):
        return MyReward.__call__(self, *args, **kwargs)


def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your sudmission directory and return a valid agent.
    """
    path_model, path_target_model = DeepQ_NNParam.get_path_model(submission_dir, name)
    nn_archi = DeepQ_NNParam.from_json(os.path.join(path_model, "nn_architecture.json"))
    res = DeepQSimple(env.action_space,
                      name=name,
                      nn_archi=nn_archi,
                      observation_space=env.observation_space)
    res.load(submission_dir)
    return res
