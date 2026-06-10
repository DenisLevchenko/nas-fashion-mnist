import optuna
import torch
import sys
import logging
import plotly
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dataclasses import dataclass
from lightning_definitions import FashionMNISTGPUDataModule, LitMLP, LitCNN, LitCNNRich, FashionMNISTDataModule
from run_with_lightning import build_lit_module

torch.set_float32_matmul_precision("high")

@dataclass
class MLPConfig:
    n_hidden: int = 3
    size_hidden: int = 16

@dataclass
class CNNConfig:
    out_channels: int = 16
    n_intermediate: int = 1
    kernel_size: int = 3
    padding: str = 'same'
    dilation: int = 1
    dropout_rate: float = 0

@dataclass
class TrainConfig:
    lr: float = 1e-2
    batch_size: int = 128

dm_full_gpu = False
# architecture_type = "mlp"
# architecture_type = "cnn"
architecture_type = "cnn_rich"

if dm_full_gpu:
    dm = FashionMNISTGPUDataModule(batch_size=TrainConfig().batch_size)
else:
    dm = FashionMNISTDataModule(batch_size=TrainConfig().batch_size)


def objective_mlp(trial):
    config = MLPConfig(
        n_hidden=trial.suggest_int("n_hidden", 1, 5),
        size_hidden=trial.suggest_int("size_hidden", 8, 256, log=True),
    )
    lit_module = LitMLP(learning_rate=TrainConfig().lr, **config.__dict__)
    
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


def objective_cnn(trial):
    # dropout = trial.suggest_categorical("dropout", [True, False])
    # if dropout:
    #     dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    # else:
    #     dropout_rate = 0
    config = CNNConfig(
        out_channels=trial.suggest_int("out_channels", 4, 32, log=True),
        n_intermediate=trial.suggest_int("n_intermediate", 0, 2),
        kernel_size=trial.suggest_int("kernel_size", 3, 7, 2),
        padding='same',
        dilation=1,
        dropout_rate=0
    )
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    lit_module = LitCNN(learning_rate=lr, **config.__dict__)
    
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
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100)
    trainer.fit(lit_module, datamodule=dm)
    
    best_val_accuracy = checkpoint_callback.best_model_score.item()
    return best_val_accuracy


def objective_cnn_rich(trial):
    config = CNNConfig()
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    lit_module = LitCNNRich(learning_rate=lr, weight_decay=wd, **config.__dict__)
    
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
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100)
    trainer.fit(lit_module, datamodule=dm)
    
    best_val_accuracy = checkpoint_callback.best_model_score.item()
    return best_val_accuracy


def main():
    # init the datamodule
    # setup optuna study
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{architecture_type}_batch128"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True)
    if architecture_type == "mlp":
        study.optimize(objective_mlp, n_trials=50)
    elif architecture_type == "cnn":
        study.optimize(objective_cnn, n_trials=100)
    elif architecture_type == "cnn_rich":
        study.optimize(objective_cnn_rich, n_trials=30)

if __name__ == "__main__":
    main()