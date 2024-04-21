import cv2
import numpy as np
import torch

from device import AdbDevice
from reward.model import RewardClassifier

rwc = RewardClassifier()
rwc.load_state_dict(torch.load('reward/checkpoint/model_0.000038.pt'))
rwc.eval()


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
            return 100
        case _:
            raise 'Unreachable'


def ready_func(frames):
    # images: (113, 299, 5)
    images = np.stack(list(map(process, frames)), axis=2) / 255.
    # crop [82:93, 121:177] to get the reward
    # (11, 56, 3) -> (3, 1, 11, 56)
    crops = np.transpose(images[82:93, 121:177], (2, 0, 1)).reshape(3, 1, 11, 56)
    crops = torch.tensor(crops, dtype=torch.float)

    # state: torch.Size([113, 299, 5])
    state = torch.tensor(images, dtype=torch.float)
    reward = sum(map(label_reward, rwc(crops).argmax(1)))
    if reward > 100:
        print('Done')
    else:
        print(reward)


def test_crops(images, rewards):
    from utils import show_image
    for i, (img, reward) in enumerate(zip(images, rewards)):
        if input(f'{i}: {reward}') != '?':
            show_image(img)
        else:
            break


if __name__ == '__main__':
    device = AdbDevice(host='127.0.0.1', port=21503)
    device.ready = ready_func
    device.start()
