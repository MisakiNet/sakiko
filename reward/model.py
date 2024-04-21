from torch import nn


class RewardClassifier(nn.Module):
    def __init__(self):
        super(RewardClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten(),
            nn.Linear(in_features=1792, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=7)
        )

    def forward(self, x):
        return self.net(x)
