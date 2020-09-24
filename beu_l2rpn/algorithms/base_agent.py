import os
import random

import numpy as np
import torch
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import MultiMixEnvironment

from beu_l2rpn.nn_builder.pytorch.nn import NN


class BaseAgent(AgentWithConverter):

    def __init__(self, env, config):
        AgentWithConverter.__init__(self, env.action_space, action_space_converter=IdToAct)

        self.action_space.filter_action(self.filter_action)

        self.config = config
        self.env = env
        self.observation_space = env.observation_space
        self.set_random_seeds(config["seed"])

        self.action_size = int(self.action_space.size())
        self.state_size = int(env.observation_space.size())
        self.hyper_parameters = config["hyper_parameters"]

        self.frames = []  # last n-1 frames, including current frame
        self.frames2 = []  # last n-2 frames, current frame and next frame
        self.episode_number = 0
        self.resume_episode = -1
        self.device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
        self.global_step_number = 0
        self.turn_off_exploration = False

    def run_episode(self):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise NotImplementedError("run_episode needs to be implemented by the agent")

    def filter_action(self, action):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise NotImplementedError("filter_action needs to be implemented by the agent")

    def my_act(self, transformed_observation, reward, done=False):
        raise NotImplementedError("my_act needs to be implemented by the agent")

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.env.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.env.seed(self.config["seed"])
        if isinstance(self.env, MultiMixEnvironment):
            self.state = self.env.reset(random=True)
        else:
            self.state = self.env.reset()

        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.info = None
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []

        if "exploration_strategy" in self.__dict__.keys():
            self.exploration_strategy.reset()

    def conduct_action(self, act):
        """Conducts an action in the environment"""
        obs, self.reward, self.done, self.info = self.env.step(act)
        self.next_state = obs
        self.total_episode_score_so_far += self.reward
        if self.hyper_parameters["clip_rewards"]:
            self.reward = max(min(self.reward, 1.0), -1.0)

    def random_action(self):
        return np.random.randint(0, self.action_size)

    def update_learning_rate(self, starting_lr, optimizer):
        pass

    def enough_experiences_to_learn(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyper_parameters["batch_size"]

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    @staticmethod
    def take_optimisation_step(optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward(retain_graph=retain_graph)  # this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               clipping_norm)  # clip gradients to help stabilise training
        optimizer.step()  # this applies the gradients

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def create_nn(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyper_parameters=None):
        """Creates a neural network for the agents to use"""
        if hyper_parameters is None:
            hyper_parameters = self.hyper_parameters
        if key_to_use:
            hyper_parameters = hyper_parameters[key_to_use]
        if override_seed:
            seed = override_seed
        else:
            seed = self.config["seed"]

        default_hyper_parameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                           "initialiser": "default", "batch_norm": False,
                                           "columns_of_data_to_be_embedded": [],
                                           "embedding_dimensions": [], "y_range": ()}

        for key in default_hyper_parameter_choices:
            if key not in hyper_parameters.keys():
                hyper_parameters[key] = default_hyper_parameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyper_parameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyper_parameters["final_layer_activation"],
                  batch_norm=hyper_parameters["batch_norm"], dropout=hyper_parameters["dropout"],
                  hidden_activations=hyper_parameters["hidden_activations"],
                  initialiser=hyper_parameters["initialiser"],
                  columns_of_data_to_be_embedded=hyper_parameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyper_parameters["embedding_dimensions"], y_range=hyper_parameters["y_range"],
                  random_seed=seed).to(self.device)

    @staticmethod
    def copy_model(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    def save_model(self):
        raise NotImplementedError("save_model needs to be implemented by the agent")

    def load_model(self, path):
        raise NotImplementedError("load_model needs to be implemented by the agent")
