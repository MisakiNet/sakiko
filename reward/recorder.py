import os

from PIL import Image

from device import AdbDevice
from state import process

cnt = 0


def ready_func(frames):
    global cnt
    frames = list(map(process, frames))
    # state: torch.Size([113, 299, 3])
    for f in frames:
        Image.fromarray(f[82:93, 121:177]).save(f'temp/{cnt}.png')
        cnt += 1
    if cnt > 1:
        exit()


if __name__ == '__main__':
    # os.mkdir('temp')
    device = AdbDevice()
    device.start()
