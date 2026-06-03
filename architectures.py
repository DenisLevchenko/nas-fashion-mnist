"""
Neural architecture class definitions.

Used by the main train.py and run_with_lightning.py.
"""

import torch
from torch import nn
from typing import Tuple, List, Union
import lightning as L
from torchmetrics.classification import MulticlassAccuracy

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


# define the LightningModule
class LitMLP(L.LightningModule):
    def __init__(self, n_hidden: int, size_hidden: int, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = MLP(n_hidden=n_hidden, size_hidden=size_hidden)
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=10)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.net(x)
        loss = self.loss(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.net(x)
        val_loss = self.loss(z, y)
        val_acc = self.accuracy(z, y.argmax(dim=1))
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", val_acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.net(x)
        test_loss = self.loss(z, y)
        test_acc = self.accuracy(z, y.argmax(dim=1))
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_acc)
        # self.test_step_outputs.append(test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


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
        dropout_rate:
            if nonzero, dropout rate for the final dense layer
    """
    
    def __init__(self, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int], str],
                 dilation: Union[int, Tuple[int, int]],
                 dropout_rate: float):
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
        self.conv_stack1 = nn.Sequential(conv1, nn.ReLU(), pooling)
        self.conv_stack2 = nn.Sequential(conv2, nn.ReLU(), pooling)
        self.conv_stack3 = nn.Sequential(conv2, nn.ReLU(), pooling)
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


class LitCNN(L.LightningModule):
    def __init__(self, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int], str],
                 dilation: Union[int, Tuple[int, int]],
                 dropout_rate: float, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = CNN(out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                       dilation=dilation, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=10)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.net(x)
        loss = self.loss(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.net(x)
        val_loss = self.loss(z, y)
        val_acc = self.accuracy(z, y.argmax(dim=1))
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", val_acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.net(x)
        test_loss = self.loss(z, y)
        test_acc = self.accuracy(z, y.argmax(dim=1))
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_acc)
        # self.test_step_outputs.append(test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer