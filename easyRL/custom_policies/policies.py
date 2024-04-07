import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from functools import reduce

class BasicCustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(BasicCustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)

        C = observation_space.shape[0]
        H = observation_space.shape[1]
        W = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # take auto number of pixels after cnn
        out_shape = self._get_output_shape_cnn(C, H, W)

        # Define fully connected layers
        self.fc1 = nn.Linear(out_shape, features_dim)

        self._initialize_weights()

    def _get_output_shape_cnn(self, C, H, W):
        random_observation = torch.rand(1, C, H, W)
        with torch.no_grad():
            x = self.cnn(random_observation)
            x = x.reshape(x.size(0), -1)

        return x.size()[1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, observations):
        # Forward pass through the cnn
        x = self.cnn(observations)

        # Flatten the feature map
        x = x.reshape(x.size(0), -1)

        # Go to fc net
        x = torch.relu(self.fc1(x))

        return x


class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomMLP, self).__init__(observation_space, features_dim)

        # Here should be dim of observation_space
        input_dim = observation_space.shape[0]

        # Create the MLP with fixed hidden layer sizes
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Forward pass through the MLP
        return self.mlp(observations)
