from beu_l2rpn.algorithms.actor_critic_agents.sac_discrete import SACDiscrete


class BeUAgent(SACDiscrete):

    def __init__(self,
                 env,
                 config):
        SACDiscrete.__init__(self, env, config)

        self.training_episodes_per_eval_episode = config["hyper_parameters"]["training_episodes_per_eval_episode"]

        # TODO: Override the normal memory with Prioritised Replay Buffer
        # self.memory = ...

    def step(self):
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
                self.save_experience(
                    experience=(self.state, self.action, self.reward, self.next_state, self.done))
            self.state = self.next_state
            self.global_step_number += 1

        self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def sample_experiences(self):
        # TODO: Override this to sample experiences from PER
        return self.memory.sample()

    def save_experience(self, memory=None, experience=None):
        # TODO: Override this to save experience with TD error to re-prioritize samples
        return SACDiscrete.save_experience(self, memory, experience)

    def my_act(self, transformed_observation, reward, done=False):
        self.env.cu


        return a

    def init_graph(self):
        # TODO: create graph neural net (GNN) from environment observation space
        pass

    def convert_obs(self, observation):
        # TODO: transform observation from the environment to graph features using the agent's GNN
        return observation.to_vect()

    def train(self, num_episodes=10000):
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()

    def evaluate(self):
        pass

    def filter_action(self, action):
        # TODO: Filter action here, first try to follow KAIST method by using only topology actions
        return True
