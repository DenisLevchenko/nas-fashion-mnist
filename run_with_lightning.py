"""
Main script using PyTorch Lightning.

Only uses MLP for now.
Also does validation and testing, logs everything.
The logs are accessible through TensorBoard.
Uses early stopping and saves the best model.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from architectures import MLP, CNN
from dataclasses import dataclass


# define the LightningModule
class LitModule(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = model
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


@dataclass
class MLPConfig:
    n_hidden: int = 3
    size_hidden: int = 64


@dataclass
class CNNConfig:
    out_channels: int = 8
    kernel_size: int = 3
    padding: str = 'same'
    dilation: int = 0
    act_function = nn.ReLU()
    dropout_rate: float = 0.2


@dataclass
class ModelConfig:
    name: str = "mlp"
    cfg: object = MLPConfig()


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 32


def build_model(model_config):
    if model_config.name == "mlp":
        return MLP(**model_config.cfg.__dict__)
    elif model_config.name == "cnn":
        return CNN(**model_config.cfg.__dict__)
    else:
        raise ValueError(f"Unknown model: {model_config.name}")


def main():
    # init the PyTorch model
    model = build_model(ModelConfig())
    # init the LightningModule
    lit_module = LitModule(model=model, learning_rate=TrainConfig().lr)
    
    # setup data
    training_data = datasets.FashionMNIST(
        root='~/Coding/torch_tutorial',
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float)
            .scatter_(0, torch.tensor(y), value=1)
        )
    )
    
    test_data = datasets.FashionMNIST(
        root='~/Coding/torch_tutorial',
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float)
            .scatter_(0, torch.tensor(y), value=1)
        )
    )
    
    # use 20% of training data for validation
    train_set_size = int(len(training_data) * 0.8)
    valid_set_size = len(training_data) - train_set_size
    
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(
        training_data, [train_set_size, valid_set_size], generator=seed)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # setup trainer and fit the model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',  # Metric to monitor
        save_top_k=1,  # Save only the best model
        mode='max',  # Mode for monitoring metric ('min' for minimizing, 'max' for maximizing)
        verbose=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.00,
        patience=5,
        mode="max",
        verbose=False
    )
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=50)
    trainer.fit(lit_module, train_loader, val_loader)
    
    # Load and test the best model from checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = LitModule.load_from_checkpoint(best_model_path)
    trainer.test(best_model, dataloaders=test_loader)


if __name__ == '__main__':
    main()