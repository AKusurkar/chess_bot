from torch import nn
from utils.utils_file import NUM_MOVES, NUM_PLANES 


class ResidualBlock(nn.Module):

    def __init__(self, channels: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity        
        return self.relu(out)


class MarchHare(nn.Module):
    def __init__(self, channels=96, num_blocks=8,
                 num_planes=NUM_PLANES, num_moves=NUM_MOVES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.spine = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, num_moves),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 128),        
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),                        
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.spine(x)
        return self.policy_head(x), self.value_head(x)