import hashlib

import numpy as np

from beu_l2rpn.algorithms.actor_critic_agents.sac_discrete import SACDiscrete
from beu_l2rpn.utilities.utility_functions import get_scalers


class BeUAgent(SACDiscrete):

    def __init__(self,
                 env,
                 config):
        SACDiscrete.__init__(self, env, config)

        self.training_episodes_per_eval_episode = config["hyper_parameters"]["training_episodes_per_eval_episode"]
        self.expected_return = 0
        self.hyper_parameters["action_size"] = self.action_size
        self.hyper_parameters["state_size"] = self.state_size

        self.actions_to_ids = {hashlib.sha256(a.to_vect().data.tobytes()).hexdigest(): idx for idx, a in
                               enumerate(self.action_space.all_actions)}

        self.scalers = get_scalers()

        # TODO: Override the normal memory with Prioritised Replay Buffer
        # self.memory = ...

        # Stateful objects to keep track of various things about the current episode
        self.episode_broken_lines = {}

        self.completed_episodes = 0
        self.failed_episodes = 0

    def filter_action(self, action):

        action_impacts = action.impact_on_objects()

        if not action_impacts['has_impact']:
            return True

        if action_impacts['switch_line']['changed'] or \
                action_impacts['redispatch']['changed']:
            return False

        if action_impacts['force_line']['changed'] and action_impacts['force_line']['disconnections']['count'] > 0:
            return False

        return True

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        SACDiscrete.reset_game(self)
        self.episode_broken_lines = {}

    def train(self):
        if self.config["neptune_enabled"]:
            import neptune
            neptune.init(project_qualified_name=self.config["neptune_project_name"],
                         api_token=self.config["neptune_api_token"])
            neptune.create_experiment(name="L2RPN", params=self.hyper_parameters)
            self.neptune = neptune

        while self.episode_number < self.hyper_parameters["train_num_episodes"]:
            self.reset_game()
            self.run_episode()
            if self.episode_number % self.config["check_point_episodes"] == 0 \
                    and self.global_step_number > self.hyper_parameters["min_steps_before_learning"]:
                self.save_model()

    def run_episode(self):

        eval_ep = self.episode_number > 0 and \
                  self.episode_number % self.training_episodes_per_eval_episode == 0 and \
                  self.do_evaluation_iterations

        print(f"Env {self.env.name} | Chronic {self.env.chronics_handler.get_name()} | Episode: {self.episode_number} "
              f"| {'Train mode' if not eval_ep else 'Eval mode'}")

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
        if eval_ep:
            self.summarize_eval_episodes()

    def pick_action(self, eval_ep, state=None):
        if state is None:
            state = self.state

        if eval_ep:
            encoded_act = self.actor_pick_action(state=state, eval=True)

        elif self.global_step_number < self.hyper_parameters["min_steps_before_learning"]:
            encoded_act = self.random_action()
            print("Picking random action ", encoded_act)
            # added by Sonvx
            if self.config["neptune_enabled"]:
                self.neptune.log_metric('random action', encoded_act)
        else:
            encoded_act = self.actor_pick_action(state=state)
            # added by Sonvx
            if self.config["neptune_enabled"]:
                self.neptune.log_metric('sampled action', encoded_act)
            print("Picking model sampled action ", encoded_act)

        action = self.convert_act(encoded_act)
        return action

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

    def try_do_nothing(self):
        pass

    def pick_heuristic_action(self):

        # This function defines some heuristic actions that we can try before seeking prediction from the neural net.

        act = self.try_reconnect_power_line()

        if act is not None:
            return act

        return act

    def summarize_eval_episodes(self):
        self.expected_return += (self.total_episode_score_so_far - self.expected_return) / (
                self.episode_number / self.training_episodes_per_eval_episode)
        if len(self.info["exception"]) > 0:
            self.failed_episodes += 1
        else:
            self.completed_episodes += 1

        if self.config["neptune_enabled"]:
            self.neptune.log_metric('expected return', self.expected_return)
            self.neptune.log_metric('episode reward', self.total_episode_score_so_far)
            self.neptune.log_metric('successful episode completion rate',
                                    self.completed_episodes / (self.completed_episodes + self.failed_episodes))

    def sample_experiences(self):
        # TODO: Override this to sample experiences from PER
        return self.memory.sample()

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = (self.state, self.action, self.reward, self.next_state, self.done)
        state, action, reward, next_state, done = experience

        state = self.convert_obs(self.state)
        next_state = self.convert_obs(self.next_state)
        action = self.to_encoded_act(self.action)
        memory.add_experience(state, action, reward, next_state, done)

    def act(self, observation, reward, done=False):
        act = self.action_space({})
        heuristic_act = self.pick_heuristic_action(observation)
        if heuristic_act is not None:
            act = heuristic_act
        return act

    def init_graph(self):
        # TODO: create graph neural net (GNN) from environment observation space
        pass

    def convert_obs(self, observation):
        # TODO: transform observation from the environment to graph features using the agent's GNN
        for attr in self.hyper_parameters["selected_attributes"]:
            setattr(observation, attr, getattr(observation, attr) / self.scalers[attr])
        vect = observation.to_vect()
        return vect[self.obs_idx]

    def to_encoded_act(self, act):
        return self.actions_to_ids[hashlib.sha256(act.to_vect().data.tobytes()).hexdigest()]
