from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from beu_l2rpn.utils import convert_obs


class Agent(object):
    def __init__(self, model, env, config, obs_idx, action_space, state):
        self.model = model
        self.env = env
        self.state = state
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.action_space = action_space
        self.config = config
        self.obs_idx = obs_idx

    def convert_obs(self, s):
        return convert_obs(s, self.obs_idx, self.config["selected_attributes"], self.config["feature_scalers"])

    def convert_act(self, encoded_act):
        return self.action_space.convert_act(encoded_act)

    def action_train(self):
        value, logit = self.model(Variable(self.state.unsqueeze(0)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        encoded_act = action.cpu().numpy()[0][0]
        state, self.reward, self.done, self.info = self.env.step(self.convert_act(encoded_act))
        if self.done:
            if len(self.info["exception"]) > 0:
                self.reward = -100
            else:
                self.reward = 100
        state = self.convert_obs(state)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        # self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return encoded_act

    def action_test(self):
        with torch.no_grad():
            value, logit = self.model(Variable(self.state.unsqueeze(0)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        encoded_act = action[0]
        state, self.reward, self.done, self.info = self.env.step(self.convert_act(encoded_act))
        state = self.convert_obs(state)

        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return encoded_act

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
