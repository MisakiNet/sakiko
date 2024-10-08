import math
import pickle
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class Memory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def save(self, filename: str):
        data = []
        for transition in self.memory:
            state, action, reward, next_state = transition
            state_data = [arr.tobytes() for arr in state]
            next_state_data = [arr.tobytes() for arr in next_state]
            action_data = action.cpu().numpy().tobytes()
            reward_data = reward
            data.append((state_data, action_data, reward_data, next_state_data))
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename: str, capacity: int):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        memory = cls(capacity)
        for state_data, action_data, reward_data, next_state_data in data:
            state = [np.frombuffer(arr, dtype=np.float32) for arr in state_data]
            action = torch.from_numpy(np.frombuffer(action_data, dtype=np.float32))
            reward = reward_data
            next_state = [np.frombuffer(arr, dtype=np.float32) for arr in next_state_data]
            transition = Transition(state, action, reward, next_state)
            memory.push(transition)
        return memory

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
        if self.epsilon_decay == 0:
            return self.epsilon
        return self.epsilon + (1. - self.epsilon) * \
            math.exp(-1. * self.cnt / self.epsilon_decay)

    def act_ex(self, state):
        if random.random() > self.greedy():
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                return self.Q(state)[0]
        else:
            return self.rand_func()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            return self.Q(state)[0]

    def update_target(self):
        self.T.load_state_dict(self.Q.state_dict())

    @staticmethod
    def get_action_val(q_val: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # qval: (batch_size, 22)
        # action: (batch_size,1)
        # 22 => 4 group: 8 + 3 + 8 + 3
        g0 = q_val[:, :8].gather(1, action[:, :8].argmax(dim=1).unsqueeze(1))
        g1 = q_val[:, 8:11].gather(1, action[:, 8:11].argmax(dim=1).unsqueeze(1))
        g2 = q_val[:, 11:19].gather(1, action[:, 11:19].argmax(dim=1).unsqueeze(1))
        g3 = q_val[:, 19:].gather(1, action[:, 19:].argmax(dim=1).unsqueeze(1))
        return (g0 * g1 + g2 * g3).flatten()

    @staticmethod
    def get_policy_val(t_val: torch.Tensor) -> torch.Tensor:
        """

        :param t_val: (batch_size, 22)
        :return: (batch_size,)
        """
        g0 = t_val[:, :8].max(dim=1).values.unsqueeze(1)
        g1 = t_val[:, 8:11].max(dim=1).values.unsqueeze(1)
        g2 = t_val[:, 11:19].max(dim=1).values.unsqueeze(1)
        g3 = t_val[:, 19:].max(dim=1).values.unsqueeze(1)
        return (g0 * g1 + g2 * g3).flatten()

    def learn(self, transitions):
        self.cnt += 1
        states, actions, rewards, next_states = transitions
        states = torch.tensor(np.stack(states), dtype=torch.float)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float)
        rewards = torch.tensor(np.stack(rewards), dtype=torch.float)
        actions = torch.stack(actions)

        q_val = self.get_action_val(self.Q(states), actions)
        next_t_val = self.get_policy_val(self.T(next_states))
        target = rewards + self.gamma * next_t_val

        self.optimizer.zero_grad()
        loss = F.mse_loss(q_val, target)
        loss.backward()
        self.optimizer.step()
        if self.cnt % self.target_update_batch == 0:
            self.update_target()
        return loss
