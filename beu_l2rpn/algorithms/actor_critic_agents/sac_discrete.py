import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from beu_l2rpn.algorithms.base_agent import BaseAgent
from beu_l2rpn.utilities.data_structures.replay_buffer import ReplayBuffer
from beu_l2rpn.algorithms.actor_critic_agents.sac import SAC
from beu_l2rpn.utilities.utility_functions import create_actor_distribution, init_obs_extraction


class SACDiscrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""

    agent_name = "SAC"

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

        BaseAgent.copy_model_over(self.critic_local, self.critic_target)
        BaseAgent.copy_model_over(self.critic_local_2, self.critic_target_2)

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
        assert not self.hyper_parameters[
            "add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"

        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyper_parameters["do_evaluation_iterations"]

    def filter_action(self, action):
        raise NotImplementedError("filter_action needs to be implemented by the agent")

    def my_act(self, transformed_observation, reward, done=False):
        raise NotImplementedError("my_act needs to be implemented by the agent")

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = create_actor_distribution("DISCRETE", action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (
                action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(
                next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyper_parameters["discount_rate"] * (
                min_qf_next_target)

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        with torch.no_grad():
            qf1_pi = self.critic_local(state_batch)
            qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = action_probabilities * inside_term
        policy_loss = policy_loss.sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities
