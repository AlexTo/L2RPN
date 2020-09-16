import numpy as np
from beu_l2rpn.algorithms.actor_critic_agents.sac_discrete import SACDiscrete
from beu_l2rpn.utilities.utility_functions import normalize


class BeUAgent(SACDiscrete):

    def __init__(self,
                 env,
                 config):
        SACDiscrete.__init__(self, env, config)

        self.training_episodes_per_eval_episode = config["hyper_parameters"]["training_episodes_per_eval_episode"]
        self.expected_return = 0

        # TODO: Override the normal memory with Prioritised Replay Buffer
        # self.memory = ...

    def train(self):
        if self.config["neptune_enabled"]:
            import neptune
            neptune.init(project_qualified_name=self.config["neptune_project_name"],
                         api_token=self.config["neptune_api_token"])
            neptune.create_experiment(name="L2RPN", params=self.hyper_parameters)
            self.neptune = neptune

        while self.episode_number < self.hyper_parameters["train_num_episodes"]:
            self.reset_game()
            if self.episode_number > self.resume_episode:
                self.run_episode()
                if self.episode_number % self.config["check_point_episodes"] == 0 \
                        and self.global_step_number > self.hyper_parameters["min_steps_before_learning"]:
                    self.save_model()
            else:
                self.episode_number += 1

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                          mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)

    def run_episode(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % self.training_episodes_per_eval_episode == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyper_parameters["learning_updates_per_learning_session"]):
                    self.learn()

            if not eval_ep:
                self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, self.done))
            self.state = self.next_state
            self.global_step_number += 1

        self.episode_number += 1

        self.summarize_of_latest_evaluation_episode()

    def pick_action(self, eval_ep, state=None):
        if state is None:
            state = self.state

        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)

        elif self.global_step_number < self.hyper_parameters["min_steps_before_learning"]:
            action = self.random_action()
            print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def summarize_of_latest_evaluation_episode(self):
        self.expected_return += (self.total_episode_score_so_far - self.expected_return) / self.episode_number
        if self.config["neptune_enabled"]:
            self.neptune.log_metric('expected return', self.expected_return)

    def sample_experiences(self):
        # TODO: Override this to sample experiences from PER
        return self.memory.sample()

    def save_experience(self, memory=None, experience=None):
        # TODO: Override this to save experience with TD error to re-prioritize samples
        return SACDiscrete.save_experience(self, memory, experience)

    def my_act(self, transformed_observation, reward, done=False):
        a = 0
        # try some rules
        # try predictions from our model
        return a

    def init_graph(self):
        # TODO: create graph neural net (GNN) from environment observation space
        pass

    def convert_obs(self, observation):
        # TODO: transform observation from the environment to graph features using the agent's GNN
        vect = observation.to_vect()
        return normalize(vect[self.obs_idx])

    def evaluate(self):
        pass

    def filter_action(self, action):
        # TODO: Filter action here, first try to follow KAIST method by using only topology actions
        return True
