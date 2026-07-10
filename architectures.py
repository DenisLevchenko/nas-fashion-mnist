"""
PyTorch Neural architecture class definitions.

Used by the main LitModule in lightning_definitions.py.
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
            nn.Linear(28*28, 128),
            nn.ReLU(), # rectified linear unit function
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
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

# TODO: fix repeated use of conv2 etc. need to create from functions

class CNNBasic(nn.Module):
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
        dropout_rate:
            if nonzero, dropout rate for the final dense layer
    """
    
    def __init__(self):
        super().__init__()

        conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                          kernel_size=3, padding='same')
        max_pool = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=3, padding='same')
        self.conv_stack1 = nn.Sequential(conv1, nn.ReLU(), max_pool)
        self.conv_stack2 = nn.Sequential(conv2, nn.ReLU(), max_pool)
        self.conv_stack3 = nn.Sequential(conv2, nn.ReLU(), nn.AdaptiveAvgPool2d(3))
        self.dense = nn.Linear(64 * 3 * 3, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_stack1(x)
        output = self.conv_stack2(output)
        output = self.conv_stack3(output)
        output = output.view(-1, 64 * 3 * 3)
        logits = self.dense(output)
        return logits


class CNN(nn.Module):
    """Basic 2D CNN network with configurable parameters.
    
    Args:
        out_channels:
            how many output channels/ conv filters
        kernel_size:
            kernel size for convolution
        padding:
            padding for the convolution operation:
            'same' or 'valid' or int or Tuple[int, int]
        dilation:
            dilation for the convolution operation
        dropout_rate:
            if nonzero, dropout rate for the final dense layer
    """
    
    def __init__(self, out_channels: int = 8, n_intermediate: int = 1,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: Union[int, Tuple[int, int], str] = 'same',
                 stride: int = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 dropout_rate: float = 0):
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        conv1 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                          kernel_size=kernel_size, padding=padding)
        max_pool = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(in_channels=self.out_channels,
                          out_channels=self.out_channels,
                          kernel_size=kernel_size, padding=padding)
        conv_stack1 = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU(), max_pool)
        conv_stack2 = nn.Sequential(conv2, nn.ReLU(), conv2, nn.ReLU(), max_pool)
        conv_stack3 = nn.Sequential(conv2, nn.ReLU(), conv2, nn.ReLU(), nn.AdaptiveAvgPool2d(3))
        self.conv_net = nn.Sequential()
        self.conv_net.append(conv_stack1)
        for i in range(n_intermediate):
            self.conv_net.append(conv_stack2)
        self.conv_net.append(conv_stack3)
        self.flatten = nn.Flatten()
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(self.out_channels * 3 * 3, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_net(x)
        output = self.flatten(output)
        if self.dropout_rate > 0:
            output = self.dropout(output)
        logits = self.dense(output)
        return logits


class CNNBatchNorm(nn.Module):
    """Basic 2D CNN network with batch normalization.
    
    Args:
        out_channels:
            how many output channels/ conv filters
        kernel_size:
            kernel size for convolution
        padding:
            padding for the convolution operation:
            'same' or 'valid' or int or Tuple[int, int]
        dilation:
            dilation for the convolution operation
        dropout_rate:
            if nonzero, dropout rate for the final dense layer
    """
    
    def __init__(self, out_channels: int = 8, n_intermediate: int = 1,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: Union[int, Tuple[int, int], str] = 'same',
                 stride: int = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 dropout_rate: float = 0):
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        def conv(in_channels, out_channels):
            return nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels, bias=False,
                             kernel_size=kernel_size, padding=padding)
        def batch_norm():
            return nn.BatchNorm2d(out_channels)
        max_pool = nn.MaxPool2d(2)

        conv_stack1 = nn.Sequential(conv(1, self.out_channels), batch_norm(), nn.ReLU(), max_pool)
        conv_stack2 = nn.Sequential(conv(self.out_channels, self.out_channels), batch_norm(), nn.ReLU(), max_pool)
        conv_stack3 = nn.Sequential(conv(self.out_channels, self.out_channels), batch_norm(), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.conv_net = nn.Sequential()
        self.conv_net.append(conv_stack1)
        for i in range(n_intermediate):
            self.conv_net.append(conv_stack2)
        self.conv_net.append(conv_stack3)
        self.flatten = nn.Flatten()
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(self.out_channels * 1 * 1, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_net(x)
        output = self.flatten(output)
        if self.dropout_rate > 0:
            output = self.dropout(output)
        logits = self.dense(output)
        return logits


class CNNRich(nn.Module):
    """Basic 2D CNN network with configurable parameters and batch normalization.
    
    Args:
        out_channels:
            how many output channels/ conv filters
        kernel_size:
            kernel size for convolution
        padding:
            padding for the convolution operation:
            'same' or 'valid' or int or Tuple[int, int]
        dilation:
            dilation for the convolution operation
        dropout_rate:
            if nonzero, dropout rate for the final dense layer
    """
    
    def __init__(self, out_channels: int = 8, n_intermediate: int = 1,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: Union[int, Tuple[int, int], str] = 'same',
                 stride: int = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 dropout_rate: float = 0):
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        max_pool = nn.MaxPool2d(2)
        def conv(in_channels, out_channels):
            return nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels, bias=False,
                             kernel_size=kernel_size, padding=padding)
        def batch_norm(out_channels):
            return nn.BatchNorm2d(out_channels)
        # two convolutions, with increasing number of filters before first pooling
        conv_stack1 = nn.Sequential(conv(1, self.out_channels), batch_norm(self.out_channels), nn.ReLU(), conv(self.out_channels, 2* self.out_channels), batch_norm(2 * self.out_channels), nn.ReLU(), max_pool)
        def conv_stack2(in_channels, out_channels):
            return nn.Sequential(conv(in_channels, out_channels), batch_norm(out_channels), nn.ReLU(), conv(out_channels, out_channels), batch_norm(out_channels), nn.ReLU(), max_pool)
        def conv_stack_last(in_channels):
            return nn.Sequential(conv(in_channels, in_channels), batch_norm(in_channels), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.conv_net = nn.Sequential()
        self.conv_net.append(conv_stack1)
        n = 1
        for i in range(n_intermediate):
            self.conv_net.append(conv_stack2(self.out_channels * (2 ** n), self.out_channels * (2 ** (n+1))))
            n += 1
        self.conv_net.append(conv_stack_last(self.out_channels * (2 ** (n_intermediate + 1))))
        self.flatten = nn.Flatten()
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(self.out_channels * (2 ** (n_intermediate + 1)), 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_net(x)
        output = self.flatten(output)
        if self.dropout_rate > 0:
            output = self.dropout(output)
        logits = self.dense(output)
        return logits


class CNN2(nn.Module):
    """2D CNN network with two conv operations before each pooling. Configurable parameters.
    
    Args:
        out_channels:
            how many output channels/ conv filters
        kernel_size:
            kernel size for convolution
        padding:
            padding for the convolution operation:
            'same' or 'valid' or int or Tuple[int, int]
        dilation:
            dilation for the convolution operation
        dropout_rate:
            if nonzero, dropout rate for the final dense layer
    """
    
    def __init__(self, out_channels: int = 8, n_intermediate: int = 1,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: Union[int, Tuple[int, int], str] = 'same',
                 stride: int = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 dropout_rate: float = 0):
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        max_pool = nn.MaxPool2d(2)
        def conv(in_channels, out_channels):
            return nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels, bias=False,
                             kernel_size=kernel_size, padding=padding)
        def batch_norm(out_channels):
            return nn.BatchNorm2d(out_channels)
        # two convolutions, with increasing number of filters before first pooling
        conv_stack1 = nn.Sequential(conv(1, self.out_channels), batch_norm(self.out_channels), nn.ReLU(), conv(self.out_channels, self.out_channels), batch_norm(self.out_channels), nn.ReLU(), max_pool)
        def conv_stack2(in_channels, out_channels):
            return nn.Sequential(conv(in_channels, out_channels), batch_norm(out_channels), nn.ReLU(), conv(out_channels, out_channels), batch_norm(out_channels), nn.ReLU(), max_pool)
        def conv_stack_last(in_channels):
            return nn.Sequential(conv(in_channels, in_channels), batch_norm(in_channels), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.conv_net = nn.Sequential()
        self.conv_net.append(conv_stack1)
        n = 0
        for i in range(n_intermediate):
            self.conv_net.append(conv_stack2(self.out_channels * (2 ** n), self.out_channels * (2 ** (n+1))))
            n += 1
        self.conv_net.append(conv_stack_last(self.out_channels * (2 ** n_intermediate)))
        self.flatten = nn.Flatten()
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(self.out_channels * (2 ** n_intermediate), 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_net(x)
        output = self.flatten(output)
        if self.dropout_rate > 0:
            output = self.dropout(output)
        logits = self.dense(output)
        return logits


class CNNExpand(nn.Module):
    """Rapidly expanding CNN.
    
    Args:
        out_channels:
            how many output channels/ conv filters for first conv layer, then doubles with each conv stack
        kernel_size:
            kernel size for convolution
        padding:
            padding for the convolution operation:
            'same' or 'valid' or int or Tuple[int, int]
        dilation:
            dilation for the convolution operation
        dropout_rate:
            if nonzero, dropout rate for the final dense layer
    """
    
    def __init__(self, out_channels: int = 8, n_intermediate: int = 1,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: Union[int, Tuple[int, int], str] = 'same',
                 stride: int = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 dropout_rate: float = 0):
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        max_pool = nn.MaxPool2d(2)
        def conv(in_channels, out_channels):
            return nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels, bias=False,
                             kernel_size=kernel_size, padding=padding)
        def batch_norm(out_channels):
            return nn.BatchNorm2d(out_channels)
        conv_stack1 = nn.Sequential(conv(1, self.out_channels), batch_norm(self.out_channels), nn.ReLU(), max_pool)
        def conv_stack2(in_channels):
            return nn.Sequential(conv(in_channels, 2 * in_channels), batch_norm(2 * in_channels), nn.ReLU(), max_pool)
        def conv_stack_last(in_channels):
            return nn.Sequential(conv(in_channels, 2 * in_channels), batch_norm(2 * in_channels), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.conv_net = nn.Sequential()
        self.conv_net.append(conv_stack1)
        n = 0
        for i in range(n_intermediate):
            self.conv_net.append(conv_stack2(self.out_channels * (2 ** n)))
            n += 1
        self.conv_net.append(conv_stack_last(self.out_channels * (2 ** (n_intermediate))))
        self.flatten = nn.Flatten()
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(self.out_channels * (2 ** (n_intermediate + 1)), 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_net(x)
        output = self.flatten(output)
        if self.dropout_rate > 0:
            output = self.dropout(output)
        logits = self.dense(output)
        return logits