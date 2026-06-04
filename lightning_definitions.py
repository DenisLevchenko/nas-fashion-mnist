"""
LightningModule definitions: model modules, DataModules.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import lightning as L
from torchmetrics.classification import MulticlassAccuracy
from typing import Tuple, List, Union
from architectures import MLP, CNN

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


def one_hot(y):
    return torch.zeros(10, dtype=torch.float).scatter_(
        0,
        torch.tensor(y),
        value=1
    )


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "~/Coding/torch_tutorial", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def prepare_data(self):
        # download
        datasets.FashionMNIST(root=self.data_dir,train=True, download=True)
        datasets.FashionMNIST(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # load data and split into train and val
        training_data = datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=ToTensor(),
            target_transform=one_hot
        )
        
        test_data = datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=ToTensor(),
            target_transform=one_hot
        )
        
        # split the train set into two
        train_set_size = int(len(training_data) * 0.8)
        valid_set_size = len(training_data) - train_set_size
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.val_set = random_split(
            training_data, [train_set_size, valid_set_size], generator=seed)
        
        self.test_set = test_data
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          num_workers=8, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          num_workers=8, pin_memory=True, persistent_workers=True)