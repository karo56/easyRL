import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BasicCustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        """ Remember that number_of_pixels is based on output image shape"""
        super(BasicCustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)

        C = observation_space.shape[0]

        # TODO: create automatic calculation
        H = observation_space.shape[1]
        W = observation_space.shape[2]


        # Define the CNN layers
        self.conv1 = nn.Conv2d(
            in_channels=C, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # Define max-pooling layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        number_of_pixels = 33280
        # Define fully connected layers
        self.fc1 = nn.Linear(number_of_pixels, features_dim)

        self._initialize_weights()

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
        x = observations

        # Forward pass through the network
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)

        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)

        x = torch.relu(self.conv3(x))
        x = self.maxpool(x)

        x = x.reshape(x.size(0), -1)  # Flatten the feature map

        x = torch.relu(self.fc1(x))

        return x


class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomMLP, self).__init__(observation_space, features_dim)

        # Calculate input dimension based on observation space
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
            nn.ReLU()
        )

    def forward(self, observations):
        # Forward pass through the MLP
        return self.mlp(observations)
