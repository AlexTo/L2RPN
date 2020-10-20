import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import MultiMixEnvironment
from torch.distributions import Categorical
from torch.optim import Adam

from beu_l2rpn.actor import Actor
from beu_l2rpn.critic import Critic
from beu_l2rpn.data_structures import ReplayBuffer
from beu_l2rpn.utils import create_action_mappings, init_obs_extraction, convert_obs


class BeUAgent(AgentWithConverter):

    def __init__(self, env, config, training=True, action_mappings_matrix=None):

        self.env = env
        self.config = config
        self.device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")

        self.selected_action_types = self.config["selected_action_types"]

        AgentWithConverter.__init__(self, env.action_space, action_space_converter=IdToAct)

        # self.action_space.filter_action(self.filter_action)
        self.all_actions = np.array(self.action_space.all_actions)

        self.load_action_mappings(action_mappings_matrix)

        self.action_mappings = torch.tensor(self.action_mappings, requires_grad=False).float().to(self.device)

        self.observation_space = env.observation_space
        self.set_random_seeds(config["seed"])

        obs_idx, obs_size = init_obs_extraction(self.observation_space, self.config['selected_attributes'])
        self.obs_idx = obs_idx

        self.action_size = int(self.action_space.size())
        self.state_size = int(obs_size)

        self.config["action_size"] = self.action_size
        self.config["state_size"] = self.state_size

        self.scalers = config["attribute_scalers"]

        self.episode_number = 0
        self.resume_episode = -1
        self.expected_return = 0
        self.completed_episodes = 0
        self.failed_episodes = 0
        self.global_step_number = 0

        if training:
            self.create_replay_buffer()

        self.do_evaluation_iterations = self.config["do_evaluation_iterations"]
        self.training_episodes_per_eval_episode = self.config["training_episodes_per_eval_episode"]
        self.num_stack_frames = self.config["num_stack_frames"]

        self.episode_broken_lines = {}
        self.frames = []
        self.next_frames = []

        self.create_networks()

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

    def load_action_mappings(self, action_mappings_matrix):
        config = self.config
        action_mapping_file = config['action_mappings_matrix']
        if action_mappings_matrix is not None:
            self.action_mappings = action_mappings_matrix
        elif os.path.exists(action_mapping_file):
            with open(config['action_mappings_matrix'], 'rb') as f:
                self.action_mappings = np.load(f)
        else:
            self.action_mappings = create_action_mappings(self.env, self.all_actions, config["selected_action_types"])
            np.save(action_mapping_file, self.action_mappings)

    def create_replay_buffer(self):
        self.memory = ReplayBuffer(self.config["buffer_size"], self.config["batch_size"],
                                   self.config["seed"])
        if os.path.exists(self.config["replay_buffer_file"]):
            self.memory.load(self.config["replay_buffer_file"])
        self.global_step_number = len(self.memory)

    def create_networks(self):
        self.critic_1 = nn.DataParallel(Critic(input_dim=self.state_size * self.num_stack_frames,
                                               action_mappings=self.action_mappings,
                                               config=self.config["Critic"])).to(self.device)

        self.critic_2 = nn.DataParallel(Critic(input_dim=self.state_size * self.num_stack_frames,
                                               action_mappings=self.action_mappings,
                                               config=self.config["Critic"])).to(self.device)

        self.critic_target_1 = nn.DataParallel(Critic(input_dim=self.state_size * self.num_stack_frames,
                                                      action_mappings=self.action_mappings,
                                                      config=self.config["Critic"])).to(self.device)

        self.critic_target_2 = nn.DataParallel(Critic(input_dim=self.state_size * self.num_stack_frames,
                                                      action_mappings=self.action_mappings,
                                                      config=self.config["Critic"])).to(self.device)

        self.actor = nn.DataParallel(Actor(input_dim=self.state_size * self.num_stack_frames,
                                           action_mappings=self.action_mappings,
                                           config=self.config["Actor"])).to(self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                 lr=self.config["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=self.config["Critic"]["learning_rate"], eps=1e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.config["Actor"]["learning_rate"], eps=1e-4)

        self.copy_model(self.critic_1, self.critic_target_1)
        self.copy_model(self.critic_2, self.critic_target_2)

        self.automatic_entropy_tuning = self.config["automatically_tune_entropy_hyper_parameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log(1.0 / self.action_size) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.config["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.config["entropy_term_weight"]

    def reset_game(self):
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
        self.frames = []
        self.next_frames = []
        self.episode_broken_lines = {}

    def stack_frame(self, state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_stack_frames:
            self.frames.pop(0)

    def stack_next_frame(self, next_state):
        self.next_frames.append(next_state.copy())
        if len(self.next_frames) > self.num_stack_frames:
            self.next_frames.pop(0)

    def convert_obs(self, observation):
        return convert_obs(self.observation_space, observation, self.config["selected_attributes"],
                           self.config["attribute_scalers"])

    def conduct_action(self, encoded_act):
        act = self.convert_act(encoded_act)
        """Conducts an action in the environment"""
        obs, self.reward, self.done, self.info = self.env.step(act)
        self.next_state = obs
        self.total_episode_score_so_far += self.reward

    def random_action(self):
        return np.random.randint(0, self.action_size)

    def train(self):
        if self.config["neptune_enabled"]:
            import neptune
            neptune.init(project_qualified_name=self.config["neptune_project_name"],
                         api_token=self.config["neptune_api_token"])
            neptune.create_experiment(name="L2RPN", params=self.config)
            self.neptune = neptune

        while self.episode_number < self.config["train_num_episodes"]:
            self.reset_game()
            self.run_episode()
            if self.episode_number % self.config["check_point_episodes"] == 0 \
                    and self.global_step_number > self.config["min_steps_before_learning"]:
                self.save_model()

            if self.global_step_number > self.config["min_steps_before_learning"] and not os.path.exists(
                    self.config["replay_buffer_file"]):
                self.memory.save(self.config["replay_buffer_file"])

    def run_episode(self):

        eval_ep = self.global_step_number > self.config["min_steps_before_learning"] and \
                  self.episode_number % self.training_episodes_per_eval_episode == 0 and \
                  self.do_evaluation_iterations

        print(f"Env {self.env.name} | Chronic {self.env.chronics_handler.get_name()} | Episode: {self.episode_number} "
              f"| {'Train mode' if not eval_ep else 'Eval mode'}")

        self.episode_step_number_val = 0

        while not self.done:
            self.episode_step_number_val += 1

            if eval_ep:
                self.action = self.act(self.state, self.reward, self.done)
            else:
                self.action = self.act_train()

            self.conduct_action(self.action)

            if self.time_to_learn():
                for _ in range(self.config["learning_updates_per_learning_session"]):
                    self.learn()

            self.stack_frame(self.state)
            self.stack_next_frame(self.next_state)
            if self.done:
                if len(self.info['exception']) > 0:
                    self.reward = -1000
                else:
                    self.reward = 1000

            if not eval_ep and len(self.frames) == self.num_stack_frames:
                self.save_exp(experience=(self.frames, self.action, self.reward, self.next_frames, self.done))

            self.state = self.next_state
            self.global_step_number += 1

        self.episode_number += 1
        if eval_ep:
            self.summarize_eval_episodes()

    def act_train(self):
        frames = self.frames

        if len(frames) < self.num_stack_frames:
            return 0

        if self.global_step_number < self.config["min_steps_before_learning"]:
            encoded_act = self.random_action()
            print("Picking random action ", encoded_act)
            # added by Sonvx
            self.log_metric('random action', encoded_act)
        else:
            encoded_act = self.actor_pick_action()
            # added by Sonvx
            self.log_metric('sampled action', encoded_act)
            print("Picking model sampled action ", encoded_act)

        return encoded_act

    def act(self, state, reward, done=False):
        hyper_params = self.config
        if not any(state.rho > int(hyper_params["danger_threshold"]["rho"])):
            return 0

        frames = self.frames

        if len(frames) < self.num_stack_frames:
            return 0

        encoded_act = self.actor_pick_action(eval_ep=True)

        return encoded_act

    def my_act(self, transformed_observation, reward, done=False):
        pass

    def try_reconnect_power_line(self):

        state = self.state
        act = None

        zero_rhos = np.where(state.rho <= 0)[0]

        for line_id in zero_rhos:
            if line_id in self.episode_broken_lines:
                self.episode_broken_lines[line_id] += 1
                if self.episode_broken_lines[line_id] > 10:
                    self.episode_broken_lines[line_id] = 10
            else:
                self.episode_broken_lines[line_id] = 1

        for line_id in self.episode_broken_lines:
            if line_id not in zero_rhos:
                self.episode_broken_lines[line_id] = 0

        for line in self.episode_broken_lines:
            timesteps_after_broken = self.episode_broken_lines[line]
            if timesteps_after_broken == 10 and state.time_before_cooldown_line[line] == 0:
                for o, e in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                    propose_act = self.action_space.reconnect_powerline(line_id=line, bus_or=o, bus_ex=e)
                    sim_obs, sim_reward, sim_done, info = state.simulate(propose_act)
                    if not sim_done:
                        act = propose_act
                        break
                self.episode_broken_lines[line] = 0
                break
        return act

    def actor_pick_action(self, eval_ep=False):
        """ Uses actor to pick an action in one of two ways:
        1) If eval = False and we aren't in eval mode then it picks an action that has partly been randomly sampled
        2) If eval = True then we pick the action with max probability"""

        frames = self.frames

        frames = np.concatenate([self.convert_obs(frame) for frame in frames])
        frames = torch.FloatTensor(frames).to(self.device)

        if len(frames.shape) == 1:
            frames = frames.unsqueeze(0)
        if not eval_ep:
            action, _, _ = self.produce_action_and_action_info(frames)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(frames)
        action = action.detach().cpu().numpy()
        return action[0]

    def summarize_eval_episodes(self):
        self.expected_return += (self.total_episode_score_so_far - self.expected_return) / (
                self.episode_number / self.training_episodes_per_eval_episode)
        if len(self.info["exception"]) > 0:
            self.failed_episodes += 1
        else:
            self.completed_episodes += 1
        self.log_metric('expected return', self.expected_return)
        self.log_metric('episode reward', self.total_episode_score_so_far)
        self.log_metric("number of steps completed", self.episode_step_number_val)
        self.log_metric('number of episodes completed', self.completed_episodes)

    def time_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.config["min_steps_before_learning"] \
               and self.enough_exp_to_learn() \
               and self.global_step_number % self.config["update_every_n_steps"] == 0

    def enough_exp_to_learn(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.config["batch_size"]

    def sample_experiences(self):
        return self.memory.sample()

    def save_exp(self, experience):

        frames, action, reward, next_frames, done = experience

        frames = np.concatenate([self.convert_obs(frame) for frame in frames])
        next_frames = np.concatenate([self.convert_obs(frame) for frame in next_frames])

        self.memory.add_experience(frames, action, reward, next_frames, done)

    def init_graph(self):
        # TODO: create graph neural net (GNN) from environment observation space
        pass

    def produce_action_and_action_info(self, frames):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probs = self.actor(frames)
        max_prob_action = torch.argmax(action_probs, dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return action, (action_probs, log_action_probs), max_prob_action

    def calc_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (
                action_probs, log_action_probs), _ = self.produce_action_and_action_info(
                next_state_batch)
            qf1_next_target = self.critic_target_1(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probs)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.config[
                "discount_rate"] * min_qf_next_target

        qf1 = self.critic_1(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calc_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probs, log_action_probs), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_1(state_batch)
        qf2_pi = self.critic_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probs - min_qf_pi
        policy_loss = (action_probs * inside_term).sum(dim=1).mean()
        log_action_probs = torch.sum(log_action_probs * action_probs, dim=1)
        return policy_loss, log_action_probs

    def calc_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calc_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                     mask_batch)
        self.update_critics(qf1_loss, qf2_loss)

        policy_loss, log_pi = self.calc_actor_loss(state_batch)

        if self.automatic_entropy_tuning:
            alpha_loss = self.calc_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None

        self.update_actor(policy_loss, alpha_loss)

        self.log_metric('qf1_loss', qf1_loss.item())
        self.log_metric('qf2_loss', qf2_loss.item())
        self.log_metric('policy_loss', policy_loss.item())
        self.log_metric('alpha_loss', alpha_loss.item())

    def log_metric(self, metric_name, metric):
        if self.config["neptune_enabled"]:
            self.neptune.log_metric(metric_name, metric)

    def update_critics(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""
        self.take_optim_step(self.critic_optimizer, self.critic_1, critic_loss_1,
                             self.config["Critic"]["gradient_clipping_norm"])
        self.take_optim_step(self.critic_optimizer_2, self.critic_2, critic_loss_2,
                             self.config["Critic"]["gradient_clipping_norm"])

        self.soft_update(self.critic_1, self.critic_target_1,
                         self.config["Critic"]["tau"])
        self.soft_update(self.critic_2, self.critic_target_2,
                         self.config["Critic"]["tau"])

    def update_actor(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optim_step(self.actor_optimizer, self.actor, actor_loss,
                             self.config["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optim_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def save_model(self):
        cfg = self.config
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_dir = os.path.join(cfg["check_point_folder"], f"{timestamp}_episode_{self.episode_number}")
        os.makedirs(save_dir)
        torch.save({
            "resume_episode": self.episode_number,
            "global_step_number": self.global_step_number,
            "critic_local": self.critic_1.state_dict(),
            "critic_local_2": self.critic_2.state_dict(),
            "critic_target": self.critic_target_1.state_dict(),
            "critic_target_2": self.critic_target_2.state_dict(),
            "actor_local": self.actor.state_dict(),
        }, os.path.join(save_dir, "model.pth"))

        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=4)

        # TODO: to resume, we also have to save the experience replay

    def load_model(self, path):
        chk_point = torch.load(path, map_location=self.device)
        self.episode_number = chk_point["resume_episode"]
        self.global_step_number = chk_point["global_step_number"]
        self.critic_1.load_state_dict(chk_point["critic_local"])
        self.critic_2.load_state_dict(chk_point["critic_local_2"])
        self.critic_target_1.load_state_dict(chk_point["critic_target"])
        self.critic_target_2.load_state_dict(chk_point["critic_target_2"])
        self.actor.load_state_dict(chk_point["actor_local"])

    @staticmethod
    def take_optim_step(optimizer, network, loss, clipping_norm=None, retain_graph=False):
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

    @staticmethod
    def copy_model(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
