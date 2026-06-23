import optuna
import torch
import sys
import logging
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning_definitions import FashionMNISTNoAugment, FashionMNISTDataModule, LitModule
from config import *

torch.set_float32_matmul_precision("high")

# architecture_type = "mlp"
# architecture_type = "cnn"
architecture_type = "cnn_rich"
# architecture_type = "cnn_expand"


# set if want to augment the training data with affine transforms and flips
augment = True

if augment:
    dm = FashionMNISTDataModule(batch_size=data_config['batch_size'], affine_scale=None)
else:
    dm = FashionMNISTNoAugment(batch_size=data_config['batch_size'])


def objective_mlp(trial):
    # config = MLPConfig(
    #     n_hidden=trial.suggest_int("n_hidden", 1, 5),
    #     size_hidden=trial.suggest_int("size_hidden", 8, 256, log=True),
    # )
    lit_module = LitModule(learning_rate=optimizer_config['lr'])
    
    # setup trainer and fit the model
    checkpoint_callback = ModelCheckpoint(**checkpoint_config)
    early_stop_callback = EarlyStopping(**early_stop_config)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=50)
    trainer.fit(lit_module, datamodule=dm)
    
    best_val_accuracy = checkpoint_callback.best_model_score.item()
    return best_val_accuracy


def objective_cnn(trial):
    # dropout = trial.suggest_categorical("dropout", [True, False])
    dropout = True
    if dropout:
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3)
    else:
        trial.set_user_attr("dropout_rate", 0)
        dropout_rate = trial.user_attrs["dropout_rate"]
    trial.set_user_attr("kernel_size", 3)
    trial.set_user_attr("dilation", 1)
    trial.set_user_attr("padding", "same")
    trial.set_user_attr("n_intermediate", 1)
    cnn_config = {
    'out_channels' : trial.suggest_int("out_channels", 16, 32, log=True),
    'n_intermediate' : trial.user_attrs["n_intermediate"],
    'kernel_size' : trial.user_attrs["kernel_size"],
    'padding' : trial.user_attrs["padding"],
    'dilation' : trial.user_attrs["dilation"],
    'dropout_rate' : dropout_rate
    }

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer_config ={
        'lr' : lr,
        'weight_decay' : weight_decay
    }

    lit_module = LitModule(architecture_type=architecture_type, net_params=cnn_config, optimizer_params=optimizer_config)
    
    # setup trainer and fit the model
    checkpoint_callback = ModelCheckpoint(**checkpoint_config)
    early_stop_callback = EarlyStopping(**early_stop_config)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100)
    trainer.fit(lit_module, datamodule=dm)
    
    best_val_accuracy = checkpoint_callback.best_model_score.item()
    return best_val_accuracy


def main():
    # init the datamodule
    # setup optuna study
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    database_dir = "optuna_databases"
    study_name = f"{architecture_type}_am"
    storage_name = f"sqlite:///{database_dir}/{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=False)
    if architecture_type == "mlp":
        study.optimize(objective_mlp, n_trials=50)
    else:
        study.optimize(objective_cnn, n_trials=5)
    variable_params = study.best_params.copy()
    set_params = study.best_trial.user_attrs.copy()
    all_params = variable_params | set_params
    optimizer_params = {k: all_params[k] for k in optimizer_config}
    net_params = {k: all_params[k] for k in all_params if k not in optimizer_config}
    full_config = {'architecture_type': architecture_type,
                   'net_params': net_params,
                   'optimizer_params': optimizer_params}
    with open(f"{study_name}_best_params.yaml", "w") as f:
        yaml.safe_dump(full_config, f, sort_keys=False)

if __name__ == "__main__":
    main()