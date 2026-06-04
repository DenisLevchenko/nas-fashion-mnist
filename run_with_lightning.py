"""
Main script using PyTorch Lightning.

Uses MLP and CNN architectures are defined in the `architectures.py` file.
Also does validation and testing, logs everything.
The logs are accessible through TensorBoard.
Uses early stopping and saves the best model.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
# import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from architectures import MLP, CNN
from dataclasses import dataclass
from lightning_definitions import LitMLP, LitCNN, FashionMNISTDataModule


@dataclass
class MLPConfig:
    n_hidden: int = 3
    size_hidden: int = 16


@dataclass
class CNNConfig:
    out_channels: int = 8
    kernel_size: int = 3
    padding: str = 'same'
    dilation: int = 0
    dropout_rate: float = 0.2


@dataclass
class TrainConfig:
    lr: float = 16e-3
    batch_size: int = 1024


def build_lit_module(architecture_type: str) -> L.LightningModule:
    if architecture_type == "mlp":
        return LitMLP(learning_rate=TrainConfig().lr, **MLPConfig().__dict__)
    elif architecture_type == "cnn":
        return LitCNN(learning_rate=TrainConfig().lr, **CNNConfig().__dict__)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")

architecture_type = "mlp"
# architecture_type = "cnn"

def main():
    # init the LightningModule
    lit_module = build_lit_module(architecture_type)
    
    # init the datamodule
    dm = FashionMNISTDataModule(batch_size=TrainConfig().batch_size)
    
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
    trainer = L.Trainer(profiler='simple', callbacks=[checkpoint_callback, early_stop_callback], max_epochs=50)
    trainer.fit(lit_module, datamodule=dm)
    
    # Load and test the best model from checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if architecture_type == "mlp":
        best_model = LitMLP.load_from_checkpoint(best_model_path)
    elif architecture_type == "cnn":
        best_model = LitCNN.load_from_checkpoint(best_model_path)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    trainer.test(best_model, datamodule=dm)
    print(f"Best model hparams: {best_model.hparams}")


if __name__ == '__main__':
    main()