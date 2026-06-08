import optuna
import torch
import sys
import logging
import plotly
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dataclasses import dataclass
from lightning_definitions import FashionMNISTGPUDataModule, LitMLP, LitCNN, FashionMNISTDataModule
from run_with_lightning import build_lit_module

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
    lr: float = 1e-2
    batch_size: int = 512

dm_full_gpu = False
architecture_type = "mlp"
# architecture_type = "cnn"

if dm_full_gpu:
    dm = FashionMNISTGPUDataModule(batch_size=TrainConfig().batch_size)
else:
    dm = FashionMNISTDataModule(batch_size=TrainConfig().batch_size)


def objective(trial):
    config = MLPConfig(
        n_hidden=trial.suggest_int("n_hidden", 1, 5),
        size_hidden=trial.suggest_int("size_hidden", 8, 256, log=True),
    )
    if architecture_type == "mlp":
        lit_module = LitMLP(learning_rate=TrainConfig().lr, **config.__dict__)
    elif architecture_type == "cnn":
        lit_module = LitCNN(learning_rate=TrainConfig().lr, **CNNConfig().__dict__)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    
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
    trainer.fit(lit_module, datamodule=dm)
    
    best_val_accuracy = checkpoint_callback.best_model_score.item()
    return best_val_accuracy


def main():
    # init the datamodule
    # setup optuna study
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{architecture_type}_optimization_hidden_n_and_size"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True)
    study.optimize(objective, n_trials=50)

if __name__ == "__main__":
    main()