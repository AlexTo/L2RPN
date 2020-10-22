import json
import os
import neptune
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import MultiMixEnvironment
from torch.distributions import Categorical

from beu_l2rpn.utils import convert_obs, set_random_seeds, take_optim_step, soft_update, create_networks


class BeUAgent(AgentWithConverter):

    def __init__(self, env, config, action_mappings_matrix, replay_buffer):

        if config["neptune_enabled"]:
            neptune.init(project_qualified_name=self.config["neptune_project_name"],
                         api_token=self.config["neptune_api_token"])
            neptune.create_experiment(name="L2RPN", params=self.config)
            self.neptune = neptune

        self.env = env
        self.config = config
        self.device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
        self.memory = replay_buffer
        self.selected_action_types = self.config["selected_action_types"]

        AgentWithConverter.__init__(self, env.action_space, action_space_converter=IdToAct)

        # self.action_space.filter_action(self.filter_action)
        self.all_actions = np.array(self.action_space.all_actions)

        self.action_mappings = torch.tensor(action_mappings_matrix, dtype=torch.float).to(self.device)

        self.observation_space = env.observation_space
        set_random_seeds(env, config["seed"])

        self.action_size = int(self.action_space.size())
        self.state_size = int(config["state_size"])

        self.config["action_size"] = self.action_size

        self.scalers = config["feature_scalers"]

        self.do_eval = self.config["do_eval"]
        self.eval_freq = self.config["eval_freq"]

        self.episode_broken_lines = {}
        self.auto_entropy_tuning = config["auto_entropy_tuning"]

        self.actor, self.critic_1, self.critic_2, self.critic_target_1, self.critic_target_2, self.actor_optimizer, \
        self.critic_optimizer, self.critic_optimizer_2, self.alpha, self.log_alpha, self.alpha_optim, \
        self.target_entropy = create_networks(config, self.action_size, self.state_size, self.action_mappings,
                                              self.device)

        self.expected_return = 0
        self.failed_episodes = 0
        self.completed_episodes = 0
        self.eps_num = 0
        self.global_step_number = 0

    def t(self, tensor, dtype=torch.float):
        if type(tensor) is np.ndarray:
            tensor = torch.from_numpy(tensor).to(dtype).to(self.device)
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def convert_obs(self, s, obs_is_vect=False):
        return convert_obs(self.observation_space, s, self.config["selected_attributes"],
                           self.config["feature_scalers"], obs_is_vect)

    def time_to_learn(self):
        return len(self.memory) > self.config["batch_size"] \
               and self.global_step_number % self.config["update_every_n_steps"] == 0

    def train(self):
        while self.eps_num < self.config["train_num_episodes"]:
            eval_ep = self.eps_num > 0 and self.eps_num % self.eval_freq == 0 and self.do_eval

            print(f"Env {self.env.name} | Chronic {self.env.chronics_handler.get_name()} | Episode: {self.eps_num} "
                  f"| {'Train mode' if not eval_ep else 'Eval mode'}")

            if isinstance(self.env, MultiMixEnvironment):
                s = self.env.reset(random=True)
            else:
                s = self.env.reset()

            eps_r = 0
            done = False
            eps_step = 0
            info = None

            while not done:
                eps_step += 1
                s_vect = self.convert_obs(s)
                encoded_act = self.actor_pick_act(s_vect, eval_ep)
                act = self.convert_act(encoded_act)
                s2, r, done, info = self.env.step(act)
                s2_vect = self.convert_obs(s2)

                eps_r += r
                if self.time_to_learn():
                    for _ in range(self.config["updates_per_learning_session"]):
                        self.learn()
                if done:
                    if len(info['exception']) > 0:
                        r = -10
                    else:
                        r = 10
                if not eval_ep:
                    self.save_exp(experience=(s_vect, encoded_act, r, s2_vect, done))
                s = s2
                self.global_step_number += 1

            self.eps_num += 1
            if eval_ep:
                self.log(eps_r, eps_step, info)

            if self.eps_num % self.config["check_point_episodes"] == 0 and self.eps_num > 0:
                self.save_model()

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        bs, ba, br, bs2, bdone = self.sample_experiences()
        qf1_loss, qf2_loss = self.calc_critic_losses(bs, ba, br, bs2, bdone)
        self.update_critics(qf1_loss, qf2_loss)

        policy_loss, log_pi = self.calc_actor_loss(bs)

        if self.auto_entropy_tuning:
            alpha_loss = self.calc_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None

        self.update_actor(policy_loss, alpha_loss)

        self.log_metric('qf1_loss', qf1_loss.item())
        self.log_metric('qf2_loss', qf2_loss.item())
        self.log_metric('policy_loss', policy_loss.item())
        self.log_metric('alpha_loss', alpha_loss.item())

    def my_act(self, transformed_observation, reward, done=False):
        pass

    def actor_pick_act(self, s, eval_ep=False):
        if not eval_ep:
            act, _, _ = self.forward(s)
        else:
            with torch.no_grad():
                _, z, act = self.forward(s)
        act = act.detach().cpu().numpy()
        return act[0]

    def sample_experiences(self):
        return self.memory.sample()

    def save_exp(self, experience):
        s, a, r, s2, done = experience
        self.memory.add_experience(s, a, r, s2, done)

    def forward(self, s):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        s = self.t(s)

        act_probs = self.actor(s)
        max_prob_act = torch.argmax(act_probs, dim=-1)
        act_dist = Categorical(act_probs)
        act = act_dist.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = act_probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(act_probs + z)
        return act, (act_probs, log_probs), max_prob_act

    def calc_critic_losses(self, bs, ba, br, bs2, bdone):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            act, (act_probs, log_probs), _ = self.forward(bs2)
            qf1_next_target = self.critic_target_1(bs2)
            qf2_next_target = self.critic_target_2(bs2)
            min_qf_next_target = act_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_probs)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = br + (1.0 - bdone) * self.config["discount_rate"] * min_qf_next_target

        qf1 = self.critic_1(bs).gather(1, ba.long())
        qf2 = self.critic_2(bs).gather(1, ba.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calc_actor_loss(self, bs):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        act, (act_probs, log_probs), _ = self.forward(bs)
        qf1_pi = self.critic_1(bs)
        qf2_pi = self.critic_2(bs)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_probs - min_qf_pi
        policy_loss = (act_probs * inside_term).sum(dim=1).mean()
        log_probs = torch.sum(log_probs * act_probs, dim=1)
        return policy_loss, log_probs

    def calc_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critics(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""
        take_optim_step(self.critic_optimizer, self.critic_1, critic_loss_1,
                        self.config["Critic"]["gradient_clipping_norm"])
        take_optim_step(self.critic_optimizer_2, self.critic_2, critic_loss_2,
                        self.config["Critic"]["gradient_clipping_norm"])

        soft_update(self.critic_1, self.critic_target_1,
                    self.config["Critic"]["tau"])
        soft_update(self.critic_2, self.critic_target_2,
                    self.config["Critic"]["tau"])

    def update_actor(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        take_optim_step(self.actor_optimizer, self.actor, actor_loss,
                        self.config["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            take_optim_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def save_model(self):
        cfg = self.config
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_dir = os.path.join(cfg["check_point_folder"], f"{timestamp}_episode_{self.eps_num}")
        os.makedirs(save_dir)
        torch.save({
            "resume_episode": self.eps_num,
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
        self.eps_num = chk_point["resume_episode"]
        self.global_step_number = chk_point["global_step_number"]
        self.critic_1.load_state_dict(chk_point["critic_local"])
        self.critic_2.load_state_dict(chk_point["critic_local_2"])
        self.critic_target_1.load_state_dict(chk_point["critic_target"])
        self.critic_target_2.load_state_dict(chk_point["critic_target_2"])
        self.actor.load_state_dict(chk_point["actor_local"])

    def log_metric(self, metric_name, metric):
        if self.config["neptune_enabled"]:
            self.neptune.log_metric(metric_name, metric)

    def log(self, eps_r, eps_step, info):
        self.expected_return += (eps_r - self.expected_return) / (
                self.eps_num / self.eval_freq)
        if len(info["exception"]) > 0:
            self.failed_episodes += 1
        else:
            self.completed_episodes += 1
        self.log_metric('expected return', self.expected_return)
        self.log_metric('episode reward', eps_r)
        self.log_metric("number of steps completed", eps_step)
        self.log_metric('number of episodes completed', self.completed_episodes)
