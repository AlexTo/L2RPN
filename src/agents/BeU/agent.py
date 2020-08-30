import torch
import os
from torch import nn, optim
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from l2rpn_baselines.utils import ReplayBuffer
from model import BeUNet
from tqdm import tqdm
import numpy as np

NAME = "BeUAgent"


class BeUAgent(AgentWithConverter):

    def __init__(self, env, args, training_params, **kwargs_converter):
        AgentWithConverter.__init__(self, env.action_space, action_space_converter=IdToAct, **kwargs_converter)
        self.filter_action_space(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.training_params = training_params
        self.env = env
        self.feature_list = [s.strip() for s in args.feature_list.split(',')]
        self.obs_idx = self.extract_obs_index_from_feature_list()
        self.observation_size = len(self.obs_idx)
        self.action_size = self.action_space.size()
        self.replay_buffer = ReplayBuffer(training_params.buffer_size)
        self.criterion = nn.MSELoss()

        self.model = BeUNet(self.observation_size, self.action_size).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=training_params.lr,
                                   weight_decay=training_params.weight_decay,
                                   momentum=training_params.momentum)

        self.target_model = BeUNet(self.observation_size, self.action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

    def filter_action_space(self, args):
        if args.load_path is not None:
            load_path = os.path.join(args.load_path, NAME)
            if os.path.exists(os.path.join(load_path, "actions.npy")):
                self.action_space.init_converter(all_actions=os.path.join(load_path, "actions.npy"))
            else:
                self.action_space.filter_action(self.filter_action)

        if args.save_path is not None:
            save_path = os.path.join(args.save_path, NAME)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.action_space.save(path=save_path, name="actions.npy")

    def my_act(self, transformed_observation, reward, done=False):
        pass

    def convert_obs(self, observation):
        obs_vect = observation.to_vect()
        return obs_vect[self.obs_idx]

    def train(self):
        env = self.env
        args = self.args
        training_params = self.training_params
        training_step = training_params.last_step
        iterations = args.num_train_steps

        s_next = None
        done = False
        pbar = tqdm(total=iterations - training_step)

        while training_step < iterations:
            epsilon = self.training_params.get_next_epsilon(current_step=training_step)
            s_curr = self.get_current_state(done, s_next)

            act, encoded_act = self.predict_act(s_curr, epsilon, training_step)
            obs, reward, done, info = env.step(act)

            s_next = self.convert_obs(obs)

            self.store_new_state(s_curr, encoded_act, reward, done, s_next)
            self.train_model(training_step, pbar)

            training_step += 1
            pbar.update(1)

        pbar.close()

    def train_model(self, training_step, pbar):
        training_params = self.training_params
        model = self.model
        target_model = self.target_model
        device = self.device
        gamma = self.training_params.discount_factor
        model.train()
        target_model.eval()
        if training_step > max(training_params.min_observation,
                               training_params.minibatch_size) and training_params.do_train():
            s_batch, a_batch, r_batch, d_batch, s_next_batch = self.replay_buffer.sample(training_params.minibatch_size)

            s_batch, a_batch, r_batch, s_next_batch = torch.tensor(s_batch).to(device), torch.tensor(a_batch).to(
                device), torch.tensor(r_batch).float().to(device), torch.tensor(s_next_batch).to(device)

            batch_size = s_batch.shape[0]
            q_values = model(s_batch)
            q_values_next_state = model(s_next_batch).detach().cpu().numpy()

            target = q_values.detach().clone()

            not_done_batch = ~d_batch

            next_a = np.argmax(q_values_next_state, axis=-1)
            target_q_values_next_state = target_model(s_next_batch)
            idx = np.arange(batch_size)
            target[idx, a_batch] = r_batch
            target[not_done_batch, a_batch[not_done_batch]] += gamma * \
                                                               target_q_values_next_state[idx, next_a][not_done_batch]
            loss = self.criterion(q_values, target)
            pbar.set_postfix_str(f"Loss: {loss:.4f}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_target_model_weights()

    def update_target_model_weights(self):
        tau = self.training_params.tau
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))

    def predict_act(self, s_curr, epsilon, training_step):
        model = self.model
        training_params = self.training_params
        model.eval()
        with torch.no_grad():
            s = torch.tensor(s_curr).to(self.device)

            action_values = model(s)
            action_values = action_values.cpu().numpy()

            predicted_act = np.argmax(action_values, axis=-1)

            if epsilon > 0:
                if np.random.random() < epsilon:
                    predicted_act = np.random.randint(0, self.action_size)

            if training_params.min_observe is not None and training_step < training_params.min_observe:
                predicted_act = 0

            return self.convert_act(predicted_act), predicted_act

    def get_current_state(self, done, s_next):
        env = self.env
        if s_next is None or done:
            s_next = env.reset()
            s_next = self.convert_obs(s_next)
        return s_next

    def store_new_state(self, s_curr, predicted_act, reward, done, s_next):
        self.replay_buffer.add(s_curr, predicted_act, reward, done, s_next)

    def get_obs_size(self):
        env = self.env
        feature_list = self.feature_list
        res = 0
        for obs_attr_name in feature_list:
            beg_, end_, _ = env.observation_space.get_indx_extract(obs_attr_name)
            res += end_ - beg_  # no "+1" needed because "end_" is exclude by python convention
        return res

    def extract_obs_index_from_feature_list(self):
        observation_space = self.env.observation_space
        feature_list = self.feature_list
        obs_idx = np.zeros(0, dtype=np.uint)
        for feat in feature_list:
            beg, end, _ = observation_space.get_indx_extract(feat)
            obs_idx = np.concatenate((obs_idx, np.arange(beg, end, dtype=np.uint)))
        return obs_idx

    @staticmethod
    def filter_action(action):
        act_dict = action.impact_on_objects()

        lines_affected = act_dict["force_line"]["reconnections"]["count"] \
                         + act_dict["force_line"]["disconnections"]["count"] \
                         + act_dict["switch_line"]["count"]

        if lines_affected > 1:
            return False

        substations_affected = {s['substation'] for s in act_dict["topology"]["assigned_bus"]} | \
                               {s['substation'] for s in act_dict["topology"]["bus_switch"]} | \
                               {s['substation'] for s in act_dict["topology"]["disconnect_bus"]}

        if len(substations_affected) > 1:
            return False

        return True
