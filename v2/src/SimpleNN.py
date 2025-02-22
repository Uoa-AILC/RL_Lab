import torch.nn as nn

class MiniNN(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, state):
        return self.fc(state)