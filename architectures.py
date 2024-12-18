"""
Neural architecture class definitions.

Used by the main train.py and run_with_lightning.py.
"""

import torch
from torch import nn
from typing import Tuple, List, Union

class MLPBasic(nn.Module):
    """Basic MLP network with fixed architecture."""
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(), # rectified linear unit function
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class MLP(nn.Module):
    """MLP class with minimally configurable layers.
    
    All hidden layers have the same size.
    Args:
        n_hidden:
            number of hidden layers
        size_hidden:
            size of the hidden layers
    """
    
    def __init__(self, n_hidden: int, size_hidden: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential()
        self.linear_relu_stack.append(nn.Linear(28*28, size_hidden))
        self.linear_relu_stack.append(nn.ReLU())
        for i in range(n_hidden):
            self.linear_relu_stack.append(nn.Linear(size_hidden, size_hidden))
            self.linear_relu_stack.append(nn.ReLU())
        self.linear_relu_stack.append(nn.Linear(size_hidden, 10))
        

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CNN(nn.Module):
    """Basic 2D CNN network with three convolutional layers.
    
    Args:
        out_channels:
            how many output channels/ conv filters
        kernel_size:
            kernel size for convolution and pooling
        padding:
            padding for the convolution operation:
            'same' or 'valid' or int or Tuple[int, int]
        dilation:
            dilation for the convolution operation
        act_function:
            pytorch activation function to use
        dropout_rate:
            if nonzero, dropout rate for the final dense layer
    """
    
    def __init__(self, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int], str],
                 dilation: Union[int, Tuple[int, int]],
                 act_function, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        if isinstance(kernel_size, int):
            pool_stride = (kernel_size - 1, kernel_size - 1)
        else:
            pool_stride = (kernel_size[0] - 1, kernel_size[1] -1)
        conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                          kernel_size=kernel_size, padding=padding)
        pooling = nn.MaxPool2d(kernel_size=kernel_size,
                               stride=pool_stride, padding=0)
        conv2 = nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size, padding=padding)
        self.conv_stack1 = nn.Sequential(conv1, act_function, pooling)
        self.conv_stack2 = nn.Sequential(conv2, act_function, pooling)
        self.conv_stack3 = nn.Sequential(conv2, act_function, pooling)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(32, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_stack1(x)
        output = self.conv_stack2(output)
        output = self.conv_stack3(output)
        output = output.view(-1, 32)
        if self.dropout_rate > 0:
            output = self.dropout(output)
        logits = self.dense(output)
        return logits