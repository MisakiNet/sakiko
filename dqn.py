import math
import random
from collections import deque, namedtuple

import torch
import torch.nn.functional as F
from torch import nn, optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Memory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        return self.net(state)


class DqnAgent:
    def __init__(self, qnet: nn.Module, tnet: nn.Module, rand_func,
                 lr=1e-3, gamma=0.99, epsilon=0.01, epsilon_decay=300,
                 target_update_batch=1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rand_func = rand_func

        self.Q = qnet
        self.T = tnet
        self.T.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.AdamW(self.Q.parameters(), lr, amsgrad=True)

        self.target_update_batch = target_update_batch
        self.cnt = 0

    def greedy(self):
        return self.epsilon + (1. - self.epsilon) * \
            math.exp(-1. * self.cnt / self.epsilon_decay)

    def act_ex(self, state):
        if random.random() > self.greedy():
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float)
                return self.Q(state).argmax().item()
        else:
            return self.rand_func()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            return self.Q(state).argmax().item()

    def update_target(self):
        self.T.load_state_dict(self.Q.state_dict())

    def learn(self, transitions):
        self.cnt += 1
        states, actions, rewards, next_states, dones = transitions
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)

        q_val = self.Q(states).gather(1, actions)
        next_t_val = self.T(next_states).max(1)[0].unsqueeze(1)
        target = rewards + self.gamma * next_t_val * (1. - dones)

        self.optimizer.zero_grad()
        loss = F.mse_loss(q_val, target)
        loss.backward()
        self.optimizer.step()

        if self.cnt % self.target_update_batch == 0:
            self.update_target()
