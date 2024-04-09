# Import the necessary libraries
import numpy as np

# Import torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

import math

# Import deque
from collections import deque

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features).to("cpu"))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features).to("cpu"))
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, inp):
        if self.training:
            return F.linear(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inp, self.weight_mu, self.bias_mu)

class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()

        state_dim = (4, 128, 128)
        action_dim = 12
        frame_skipping = 4

        # Use the convolutional neural network
        self.cnn = nn.Sequential(
            nn.Conv2d(frame_skipping, 8, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(),

            # Max pooling
            nn.MaxPool2d(2, 2)
        )

        # Calculate the output size
        self.conv_out_size = self._get_conv_out(state_dim)

        self.advantage = nn.Sequential(
            NoisyLinear(self.conv_out_size, 256),
            nn.LeakyReLU(),
            NoisyLinear(256, action_dim)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(self.conv_out_size, 256),
            nn.LeakyReLU(),
            NoisyLinear(256, 1)
        )

        # Directly load the module
        self.load_state_dict(torch.load("110060018_hw2_data.py")["model_state_dict"])

        self.frame_stacking_queue = deque(maxlen=4)
        self.frame_skipping = 0

        self.prev_action = 0

        self.first_remove = True
        self.first_count = 0

        # Move self to cpu
        self.to(torch.device("cpu"))

    def reset(self):
        self.frame_stacking_queue.clear()
        self.frame_skipping = 0
        self.prev_action = 0
        self.first_remove = True
        self.first_count = 0

    def _get_conv_out(self, shape):
        o = self.cnn(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def sample_noise(self):
        # for layer in self.linear:
        #     if isinstance(layer, NoisyLinear):
        #         layer.sample_noise()
        
        for layer in self.advantage:
            if isinstance(layer, NoisyLinear):
                layer.sample_noise()
        for layer in self.value:
            if isinstance(layer, NoisyLinear):
                layer.sample_noise()

    def init_weights(self):

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):

        # Make x to cpu
        # x = x.to(torch.device("cpu"))

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        cnn_out = self.cnn(x)
        x = cnn_out.view(cnn_out.size(0), -1)
        
        advantage = self.advantage(x)
        value = self.value(x)
        
        q = value + advantage - advantage.mean()
        return q

    def resize_and_gray(self, observation):

        # Gray scale with cv2
        obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        
        # Resize
        obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_AREA)

        return obs

    def act(self, observation):

        if self.first_count >= 3441:
            self.reset()

        self.first_count += 1

        if self.first_count < 5:
            return 0

        # Do frame skipping
        if self.frame_skipping % 4 == 1:

            # self.frame_skipping = 0
            observation = self.resize_and_gray(observation)

            self.frame_stacking_queue.append(observation)
            while len(self.frame_stacking_queue) < 4:
                self.frame_stacking_queue.append(observation)

            # Pick action
            state = np.stack(self.frame_stacking_queue, axis=0)

            # Make the state to tensor
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(torch.device("cpu"))

            # Get the action
            with torch.no_grad():
                action = self(state).max(1).indices.view(1, 1).item()

            self.prev_action = action

        self.frame_skipping += 1

        return self.prev_action