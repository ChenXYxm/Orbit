import random
import numpy as np
import gym
from fcn2 import FCNet
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as op
from dataclasses import dataclass
from typing import Any
from random import sample
import torch
import math
from collections import namedtuple, deque
@dataclass
class Sars:
    state:Any
    action: Any
    reward:float
    next_state:Any
    done: bool

class DQNAgent:
    def __init__(self,env,model,batch_size,gamma,eps_start,
                 eps_end,eps_decay,tau,total_steps) -> None:
        self.policy_model = FCNet(1,1).cuda()
        self.target_model = FCNet(1,1).cuda()
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.memory = ReplayBuffer()
        self.total_steps = total_steps
        self.steps_done = 0
        self.env = env
        self.device = self.env.env.device
        pass
    def get_vals(self,obs):
        # observations shape is (N*1*100*100)
        q_vals_push,q_vals_stop = self.model(obs)
        # print(q_vals_push.size())
        # print(q_vals_stop.size())
        return q_vals_push,q_vals_stop
    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())
    def process_q_vals(self,q_vals_push,q_vals_stop):
        sample = random.random()
        eps_threshold = self.eps_end+(self.eps_start-self.eps_end)*math.exp(-1.*self.steps_done/self.eps_decay)
        if sample>eps_threshold:
            with torch.no_grad():
                return np.unravel_index(np.argmax(q_vals_push.cpu().numpy()), q_vals_push.cpu().numpy().shape)
        else:
            return self.env.action_space.sample()
    def train_step(self,state_transitions):
        cur_states = torch.stack([s.state for s in state_transitions])
        rewards = torch.stack([s.reward for s in state_transitions])
        mask = torch.stack([0 if s.done else 1 for s in state_transitions])
        next_states = torch.stack([s.next_state for s in state_transitions])
        actions = torch.stack([s.action for s in state_transitions])
        with torch.no_grad():
            q_vals_push_next, q_vals_stop_next = self.target_model(next_states)
            
class ReplayBuffer:
    def __init__(self,buffer_size=1000) -> None:
        self.buffer_size = buffer_size
        self.buffer = []
        pass
    def insert(self,Sars):
        self.buffer.append(Sars)
        self.buffer = self.buffer[-self.buffer_size:]
    def sample(self,num_samples):
        assert num_samples<= len(self.buffer)
        return sample(self.buffer,num_samples)
