import numpy as np
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):

        super().__init__()

        self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, 
                          out_channels = 64,
                          kernel_size = 3,
                          stride = 1,
                          padding = 0),
                nn.BatchNorm2d(num_features = 64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3,
                             stride = 2,
                             padding = 0)
                )

        self.conv_block2 = nn.Sequential(
                nn.Conv2d(in_channels = 64,
                          out_channels = 128,
                          kernel_size = 3,
                          stride = 1,
                          padding = 0),
                nn.BatchNorm2d(num_features = 128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2,
                             stride = 2,
                             padding = 0)
                )

        self.conv_block3 = nn.Sequential(
                nn.Conv2d(in_channels = 128,
                          out_channels = 256,
                          kernel_size = 3,
                          stride = 1,
                          padding = 0),
                nn.BatchNorm2d(num_features = 256),
                nn.ReLU()
                )

        self.flatten = nn.Flatten()

        self.MLP = nn.Sequential(
                nn.LazyLinear(out_features = 128),
                nn.ReLU(),
                nn.Linear(in_features = 128,
                          out_features = out_channels),
                #nn.Softmax()
                )

    def forward(self, x):

        return self.MLP(self.flatten(self.conv_block3(self.conv_block2(self.conv_block1(x)))))


