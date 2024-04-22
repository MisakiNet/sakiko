import numpy as np

from control import *
from state import StateDevice


class BangEnv:
    def __init__(self, device: StateDevice):
        self.device = device

    def reset(self) -> (np.ndarray, int):
        return self.device.state.receive()

    def step(self, action: torch.Tensor):
        l1, l2, r1, r2 = de_control(action)
        self.device.send(l1, l2, r1, r2)
        state, reward = self.device.state.receive()
        if reward > 100:
            reward = 0
            done = True
        else:
            done = False
        if self.device.dirty:
            done = True
            self.device.dirty = False
        return state, reward, done
