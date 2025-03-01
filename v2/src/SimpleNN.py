import torch.nn as nn

class MiniNN(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, output_size),
        )

    def forward(self, state):
        # print(state.shape)
        return self.fc(state)