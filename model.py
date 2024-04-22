from torch import nn

from config import *


class SakikoNetwork(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        # input: torch.Size([height, width, STATE_FRAMES])
        # output: torch.Size([22])
        self.net = nn.Sequential(
            nn.Conv2d(STATE_FRAMES, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (width - 6) * (height - 6), 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 22)
        )

    def forward(self, x):
        return self.net(x)
