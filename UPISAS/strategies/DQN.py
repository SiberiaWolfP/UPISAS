import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm


import os


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQNet(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQNet, self).__init__()
        self.input = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.output(x)
    
class DQN():

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    def __init__(self, n_actions, n_observations, batch_size=64, gamma=0.99, eps_start=0.95, eps_end=0.01, 
                 eps_decay=1000, tau=0.005, learning_rate=1e-4) -> None:
        super().__init__()

                # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.LR = learning_rate
        self.n_actions = n_actions
        self.n_observations = n_observations

        self.policy_net = DQNet(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQNet(self.n_observations, self.n_actions).to(self.device)

        if os.path.exists('./checkpoint/dingnet_dqn_discrete_policy.pth'):
            self.policy_net.load_state_dict(torch.load('./checkpoint/dingnet_dqn_discrete_policy.pth'))
            self.target_net.load_state_dict(torch.load('./checkpoint/dingnet_dqn_discrete_target.pth'))
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(1000000)

        self.steps_done = 0
        self.eps_threshold = 0

        self.state = None
        self.next_state = None
        self.reward = None
        self.action = None

        self.steps = 0

    def _select_action(self, state):
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randint(0, self.n_actions - 1)]], device=self.device, dtype=torch.long)


    def _optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _soft_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def _save_checkpoint(self):
        torch.save(self.policy_net.state_dict(), './checkpoint/dingnet_dqn_discrete_policy.pth')
        torch.save(self.target_net.state_dict(), './checkpoint/dingnet_dqn_discrete_target.pth')

    def learn(self, next_state, reward):
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # next_state = torch.from_numpy(next_state).float().to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        # reward = torch.from_numpy(reward).float().to(self.device)
        if self.state is not None:
            # # State changed but not caused by the action
            # if self.action.item() != action:
            #     return None
            # Store the transition caused by last adaptation
            self.memory.push(self.state, self.action, next_state, reward)
        # Make new adaptation
        self.state = next_state
        self.action = self._select_action(self.state)
        self._optimize_model()
        self._soft_update()

        # Save checkpoint every 1000 steps
        self.steps += 1
        if self.steps % 1000 == 0:
            self._save_checkpoint()

        return self.action.item() - 50
    
    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.target_net(state).max(1).indices.view(1, 1).item() - 50
