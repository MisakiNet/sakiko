import torch
from torch import nn

from config import *


class SakikoNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # input: torch.Size([STATE_FRAMES, height, width])
        # output: torch.Size([22])
        self.net = nn.Sequential(
            nn.Conv2d(STATE_FRAMES, 20, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(20, 40, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten(),

            nn.Linear(75920, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 22)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = SakikoNetwork()
    print(model)
    print(model(torch.rand(1, STATE_FRAMES, 113, 299)).shape)
    torch.save(model.state_dict(), 'checkpoint/model_size.pth')
