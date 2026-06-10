"""
LightningModule definitions: model modules, DataModules.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets
import torchvision.transforms as T
import lightning as L
from torchmetrics.classification import MulticlassAccuracy
from typing import Tuple, List, Union
from architectures import MLP, CNN, CNNExpand, CNNRich

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
    def __init__(self, out_channels: int, n_intermediate: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int], str],
                 dilation: Union[int, Tuple[int, int]],
                 dropout_rate: float, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = CNN(out_channels=out_channels, n_intermediate=n_intermediate, kernel_size=kernel_size, padding=padding,
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
        acc = self.accuracy(z, y.argmax(dim=1))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
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


class LitCNNRich(L.LightningModule):
    def __init__(self, out_channels: int, n_intermediate: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int], str],
                 dilation: Union[int, Tuple[int, int]],
                 dropout_rate: float,
                 learning_rate: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = CNNRich(out_channels=out_channels, n_intermediate=n_intermediate, kernel_size=kernel_size, padding=padding,
                       dilation=dilation, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=10)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.net(x)
        loss = self.loss(z, y)
        acc = self.accuracy(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.net(x)
        val_loss = self.loss(z, y)
        val_acc = self.accuracy(z, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", val_acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.net(x)
        test_loss = self.loss(z, y)
        test_acc = self.accuracy(z, y)
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_acc)
        # self.test_step_outputs.append(test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class LitCNNExpand(L.LightningModule):
    def __init__(self, out_channels: int, n_intermediate: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int], str],
                 dilation: Union[int, Tuple[int, int]],
                 dropout_rate: float,
                 learning_rate: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = CNNExpand(out_channels=out_channels, n_intermediate=n_intermediate, kernel_size=kernel_size, padding=padding,
                       dilation=dilation, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=10)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.net(x)
        loss = self.loss(z, y)
        acc = self.accuracy(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.net(x)
        val_loss = self.loss(z, y)
        val_acc = self.accuracy(z, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", val_acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.net(x)
        test_loss = self.loss(z, y)
        test_acc = self.accuracy(z, y)
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_acc)
        # self.test_step_outputs.append(test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
        transform = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST stats
        ])
        # load data and split into train and val
        training_data = datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_data = datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
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


class FashionMNISTGPUDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir="~/Coding/torch_tutorial",
        batch_size=512,
        pin_to_gpu=True,
        device="cuda"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pin_to_gpu = pin_to_gpu
        self.device = device

    def prepare_data(self):
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        def load_split(train: bool):
            ds = datasets.FashionMNIST(
                root=self.data_dir,
                train=train,
                download=False
            )

            xs = []
            ys = []

            for x, y in ds:
                xs.append(ToTensor()(x))
                ys.append(one_hot(y))

            x = torch.stack(xs)          # [N, 1, 28, 28]
            y = torch.stack(ys)          # [N, 10]

            return x, y

        # ---- load full datasets into memory ----
        x_train, y_train = load_split(train=True)
        x_test, y_test = load_split(train=False)

        # ---- split train/val ----
        n = len(x_train)
        n_train = int(0.8 * n)
        n_val = n - n_train

        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=g)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        train_x, train_y = x_train[train_idx], y_train[train_idx]
        val_x, val_y = x_train[val_idx], y_train[val_idx]

        # ---- optionally move everything to GPU once ----
        if self.pin_to_gpu:
            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)
            val_x = val_x.to(self.device)
            val_y = val_y.to(self.device)
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)

        self.train_set = TensorDataset(train_x, train_y)
        self.val_set = TensorDataset(val_x, val_y)
        self.test_set = TensorDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size
        )