"""
Main script using PyTorch Lightning.

Uses MLP and CNN architectures are defined in the `architectures.py` file.
Also does validation and testing, logs everything.
The logs are accessible through TensorBoard.
Uses early stopping and saves the best model.
"""

from pathlib import Path
import yaml
import torch
from torchvision.transforms import ToTensor, Lambda
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
# import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from dataclasses import dataclass, asdict
from lightning_definitions import LitModule, FashionMNISTDataModule, FashionMNISTNoAugment
from config import *


torch.set_float32_matmul_precision("high")
haparams_file_name = 'cnn_rich_am_best_params.yaml'
hparams_path = Path(haparams_file_name)
def main():
    # init the LightningModule
    with open(hparams_path, 'r') as file:
            full_config = yaml.safe_load(file)
    lit_module = LitModule(**full_config)
    print(lit_module)
    # init the datamodule
    if data_config['augment']:
        dm = FashionMNISTDataModule(batch_size=data_config['batch_size'], affine_scale=None)
    else:
        dm = FashionMNISTNoAugment(batch_size=data_config['batch_size'])
    
   
    checkpoint_callback = ModelCheckpoint(**checkpoint_config)
    early_stop_callback = EarlyStopping(**early_stop_config)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100)
    trainer.fit(lit_module, datamodule=dm)
    
    # Load and test the best model from checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = LitModule.load_from_checkpoint(best_model_path)
    trainer.test(best_model, datamodule=dm)
    print(f"Model hparams: {best_model.hparams}")


if __name__ == '__main__':
    main()