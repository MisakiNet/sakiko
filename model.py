import torch
from torch import nn

from config import *


class SakikoNetwork(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        # input: torch.Size([STATE_FRAMES, height, width])
        # output: torch.Size([22])
        self.net = nn.Sequential(
            nn.Conv2d(STATE_FRAMES, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten(),

            nn.Linear(121472, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 22)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = SakikoNetwork(113, 299)
    print(model)
    print(model(torch.rand(STATE_FRAMES, 113, 299)).shape)
