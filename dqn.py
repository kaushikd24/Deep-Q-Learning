#model goes here

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random


class DQN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size = 8, stride =4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        #pass the output of conv2 to a linear layer
        self.linear1 = nn.Linear(32*9*9, 256)
        self.output = nn.Linear(256, 4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        #flatten the output
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.output(x)
        return x
        
