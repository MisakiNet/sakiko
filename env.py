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
        # group2: 0: click, 1: down, 2: flick
        match l2:
            case 0:
                self.device.click(l1, False)
            case 1:
                self.device.key_down(l1, False)
            case _:
                self.device.flick(l1, False)
        if l1 != r1:
            match r2:
                case 0:
                    self.device.click(r1, True)
                case 1:
                    self.device.key_down(r1, True)
                case _:
                    self.device.flick(r1, True)
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
