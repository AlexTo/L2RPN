import json
import os
from datetime import datetime

import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from beu_l2rpn.algorithms.base_agent import BaseAgent
from beu_l2rpn.utilities.data_structures.replay_buffer import ReplayBuffer
from beu_l2rpn.utilities.utility_functions import create_actor_distribution, init_obs_extraction


class SACDiscrete(BaseAgent):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""

    def __init__(self, env, config):
        BaseAgent.__init__(self, env, config)

        assert self.config["hyper_parameters"]["Actor"][
                   "final_layer_activation"] == "Softmax", "Final actor layer must be softmax"

        self.hyper_parameters = config["hyper_parameters"]

        obs_idx, obs_size = init_obs_extraction(self.observation_space, self.hyper_parameters["selected_attributes"])

        self.state_size = int(obs_size)
        self.obs_idx = obs_idx

        self.critic_local = self.create_nn(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
        self.critic_local_2 = self.create_nn(input_dim=self.state_size, output_dim=self.action_size,
                                             key_to_use="Critic", override_seed=self.config["seed"] + 1)

        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyper_parameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyper_parameters["Critic"]["learning_rate"], eps=1e-4)

        self.critic_target = self.create_nn(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic")
        self.critic_target_2 = self.create_nn(input_dim=self.state_size, output_dim=self.action_size,
                                              key_to_use="Critic")

        BaseAgent.copy_model(self.critic_local, self.critic_target)
        BaseAgent.copy_model(self.critic_local_2, self.critic_target_2)

        self.memory = ReplayBuffer(self.hyper_parameters["Critic"]["buffer_size"], self.hyper_parameters["batch_size"],
                                   self.config["seed"])

        self.actor_local = self.create_nn(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyper_parameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyper_parameters["automatically_tune_entropy_hyper_parameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log(1.0 / self.action_size) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyper_parameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyper_parameters["entropy_term_weight"]

        self.do_evaluation_iterations = self.hyper_parameters["do_evaluation_iterations"]

    def filter_action(self, action):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise NotImplementedError("filter_action needs to be implemented by the agent")

    def my_act(self, transformed_observation, reward, done=False):
        raise NotImplementedError("my_act needs to be implemented by the agent")

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probs = self.actor_local(state)
        max_prob_action = torch.argmax(action_probs, dim=-1)
        action_distribution = create_actor_distribution("DISCRETE", action_probs, self.action_size)
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
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probs)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyper_parameters[
                "discount_rate"] * min_qf_next_target

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calc_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probs, log_action_probs), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
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

    def pick_action(self, eval_ep, state=None):
        raise NotImplementedError("pick_action needs to be implemented by the agent")

    def actor_pick_action(self, state=None, eval=False):
        """ Uses actor to pick an action in one of two ways:
        1) If eval = False and we aren't in eval mode then it picks an action that has partly been randomly sampled
        2) If eval = True then we pick the action with max probability"""
        if state is None:
            state = self.state
        state = self.convert_obs(state)
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if not eval:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyper_parameters["min_steps_before_learning"] \
               and self.enough_experiences_to_learn() \
               and self.global_step_number % self.hyper_parameters["update_every_n_steps"] == 0

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
        raise NotImplementedError("log_metric needs to be implemented by the agent")

    def sample_experiences(self):
        return self.memory.sample()

    def update_critics(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyper_parameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyper_parameters["Critic"]["gradient_clipping_norm"])

        self.soft_update(self.critic_local, self.critic_target,
                         self.hyper_parameters["Critic"]["tau"])
        self.soft_update(self.critic_local_2, self.critic_target_2,
                         self.hyper_parameters["Critic"]["tau"])

    def update_actor(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyper_parameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def save_model(self):
        cfg = self.config
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_dir = os.path.join(cfg["check_point_folder"], f"{timestamp}_episode_{self.episode_number}")
        os.makedirs(save_dir)
        torch.save({
            "resume_episode": self.episode_number,
            "global_step_number": self.global_step_number,
            "critic_local": self.critic_local.state_dict(),
            "critic_local_2": self.critic_local_2.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_target_2": self.critic_target_2.state_dict(),
            "actor_local": self.actor_local.state_dict(),
        }, os.path.join(save_dir, "model.pth"))

        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=4)

        # TODO: to resume, we also have to save the experience replay

    def load_model(self, path):
        chk_point = torch.load(path, map_location=self.device)
        self.episode_number = chk_point["resume_episode"]
        self.global_step_number = chk_point["global_step_number"]
        self.critic_local.load_state_dict(chk_point["critic_local"])
        self.critic_local_2.load_state_dict(chk_point["critic_local_2"])
        self.critic_target.load_state_dict(chk_point["critic_target"])
        self.critic_target_2.load_state_dict(chk_point["critic_target_2"])
        self.actor_local.load_state_dict(chk_point["actor_local"])
