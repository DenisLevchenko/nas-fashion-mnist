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
from dataclasses import dataclass, asdict
from lightning_definitions import LitMLP, LitCNN, LitCNNRich, LitCNNExpand, FashionMNISTDataModule, FashionMNISTNoAugment
from config import MLPConfig, CNNConfig, TrainConfig, SetupConfig


torch.set_float32_matmul_precision("high")


def build_lit_module(architecture_type: str) -> L.LightningModule:
    if architecture_type == "mlp":
        return LitMLP(learning_rate=TrainConfig().learning_rate, **MLPConfig().__dict__)
    elif architecture_type == "cnn":
        return LitCNN(learning_rate=TrainConfig().learning_rate, **CNNConfig().__dict__)
    elif architecture_type == "cnn_rich":
        return LitCNNRich(learning_rate=TrainConfig().learning_rate, weight_decay=TrainConfig().weight_decay, **CNNConfig().__dict__)
    elif architecture_type == "cnn_expand":
        return LitCNNExpand(learning_rate=TrainConfig().learning_rate, weight_decay=TrainConfig().weight_decay, **CNNConfig().__dict__)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")


def load_lit_from_checkpoint(architecture_type: str, model_path: str):
    if architecture_type == "mlp":
        return LitMLP.load_from_checkpoint(model_path)
    elif architecture_type == "cnn":
        return LitCNN.load_from_checkpoint(model_path)
    elif architecture_type == "cnn_rich":
        return LitCNNRich.load_from_checkpoint(model_path)
    elif architecture_type == "cnn_expand":
        return LitCNNExpand.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")


def main():
    # init the LightningModule
    lit_module = build_lit_module(SetupConfig().architecture_type)
    
    # init the datamodule
    if SetupConfig.augment:
        dm = FashionMNISTDataModule(batch_size=TrainConfig().batch_size, affine_scale=None)
    else:
        dm = FashionMNISTNoAugment(batch_size=TrainConfig().batch_size)
    
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
    best_model = load_lit_from_checkpoint(SetupConfig.architecture_type, best_model_path)
    trainer.test(best_model, datamodule=dm)
    print(f"Model hparams: {best_model.hparams}")


if __name__ == '__main__':
    main()