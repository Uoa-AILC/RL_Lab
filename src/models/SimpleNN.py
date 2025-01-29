import numpy as np
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces

class SimpleNN(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, state):
        return self.fc(state)
    
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 32):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_features = int(np.prod(observation_space.shape))
        self.cnn = SimpleNN(feature_size=n_input_features, output_size=features_dim)

    def forward(self, observations):
        return self.cnn(observations)
    
    