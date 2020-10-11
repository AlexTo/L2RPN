import json

import grid2op
from grid2op.Converter import IdToAct
from lightsim2grid.LightSimBackend import LightSimBackend

if __name__ == '__main__':
    with open("data/config.json", 'r') as f:
        config = json.load(f)

    env = grid2op.make(config["env"], backend=LightSimBackend())
    env.seed(config["seed"])

    converter = IdToAct(env.action_space)
    converter.init_converter()

    converter.save("data", f"{config['env']}_action_space.npy")
