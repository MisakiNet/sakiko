import threading
import time

import cv2
import numpy as np
import torch

from device import AdbDevice
from reward.model import RewardClassifier


class SingleElementQueue:
    def __init__(self):
        self.condition = threading.Condition()
        self.element = None

    def send(self, element):
        with self.condition:
            if self.element is not None:
                self.element = element
            else:
                self.element = element
                self.condition.notify_all()

    def receive(self):
        with self.condition:
            while self.element is None:
                self.condition.wait()
            element = self.element
            self.element = None
            return element


def process(frame: np.ndarray):
    width_margin = 0.18
    height_top_crop = 0.3
    height_bottom_crop = 0.18

    height, width, _ = frame.shape
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (int(width / 5), int(height / 5)))
    height, width = img.shape
    left_margin = int(width_margin * width)
    right_margin = width - left_margin
    top_margin = int(height_top_crop * height)
    bottom_margin = int((1 - height_bottom_crop) * height)
    img = img[top_margin:bottom_margin, left_margin:right_margin]
    return img


def label_reward(label) -> int:
    match label:
        case 0:  # bad
            return -1
        case 1:  # good
            return 1
        case 2:  # great
            return 2
        case 3:  # miss
            return -2
        case 4:  # nokey
            return 0
        case 5:  # perfect
            return 3
        case 6:  # done
            return 1000
        case _:
            raise 'Unreachable'


class StateDevice(AdbDevice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dirty = False
        self.rwc = RewardClassifier()
        self.rwc.load_state_dict(torch.load('reward/save/model_96.pt'))
        self.rwc.eval()
        self.state = SingleElementQueue()

    def ready(self):
        # state: (STATE_FRAMES, 113, 299)
        state = np.stack(list(map(process, self.frames))) / 255.
        # crop [:, 82:93, 121:177] to get the reward
        # (3, 11, 56) -> (3, 1, 11, 56)
        crops = state[:, 82:93, 121:177].reshape(-1, 1, 11, 56)
        crops = torch.tensor(crops, dtype=torch.float)
        reward = sum(map(label_reward, self.rwc(crops).argmax(1)))
        # print(f'[StateDevice] Reward: {reward}')
        self.state.send((state, reward))


def test_crops(images, rewards):
    from utils import show_image
    for i, (img, reward) in enumerate(zip(images, rewards)):
        if input(f'{i}: {reward}') != '?':
            show_image(img)
        else:
            break


if __name__ == '__main__':
    sd = StateDevice()
    sd.start(threaded=True)
    time.sleep(5000)
