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
from lightning_definitions import FashionMNISTGPUDataModule, LitMLP, LitCNN, LitCNNRich, FashionMNISTDataModule

torch.set_float32_matmul_precision("high")

@dataclass
class MLPConfig:
    n_hidden: int = 3
    size_hidden: int = 16


@dataclass
class CNNConfig:
    out_channels: int = 32
    n_intermediate: int = 1
    kernel_size: int = 3
    padding: str = 'same'
    dilation: int = 1
    dropout_rate: float = 0


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 512
    profiler: str = None # 'simple' or 'advanced' or 'pytorch' or None


def build_lit_module(architecture_type: str) -> L.LightningModule:
    if architecture_type == "mlp":
        return LitMLP(learning_rate=TrainConfig().lr, **MLPConfig().__dict__)
    elif architecture_type == "cnn":
        return LitCNN(learning_rate=TrainConfig().lr, **CNNConfig().__dict__)
    elif architecture_type == "cnn_rich":
        return LitCNNRich(learning_rate=TrainConfig().lr, **CNNConfig().__dict__)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")

# architecture_type = "mlp"
# architecture_type = "cnn"
architecture_type = "cnn_rich"


# set if want to load whole datamodule on GPU once (instead of loading batches on GPU one by one)
# dm_full_gpu = True
dm_full_gpu = False

def main():
    # init the LightningModule
    lit_module = build_lit_module(architecture_type)
    
    # init the datamodule
    if dm_full_gpu:
        dm = FashionMNISTGPUDataModule(batch_size=TrainConfig().batch_size)
    else:
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
        patience=10,
        mode="max",
        verbose=False
    )
    print(lit_module)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100)
    trainer.fit(lit_module, datamodule=dm)
    
    # Load and test the best model from checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if architecture_type == "mlp":
        best_model = LitMLP.load_from_checkpoint(best_model_path)
    elif architecture_type == "cnn":
        best_model = LitCNN.load_from_checkpoint(best_model_path)
    elif architecture_type == "cnn_rich":
        best_model = LitCNNRich.load_from_checkpoint(best_model_path)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    trainer.test(best_model, datamodule=dm)
    print(f"Best model hparams: {best_model.hparams}")


if __name__ == '__main__':
    main()