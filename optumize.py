import optuna
import torch
import sys
import logging
import yaml
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from lightning_definitions import FashionMNISTNoAugment, FashionMNISTDataModule, LitModule


architecture_type = "mlp"
# architecture_type = "cnn"
# architecture_type = "cnn_rich"
# architecture_type = "cnn_expand"

# set if want to augment the training data with affine transforms and flips
optimizer_keys = ["lr", "weight_decay"]


# init the datamodule

def objective_mlp(trial):
    
    n_hidden = trial.suggest_int("n_hidden", 2, 4)
    size_hidden = trial.suggest_int("size_hidden", 64, 512, log=True)
    mlp_config = {
        'n_hidden': n_hidden,
        'size_hidden': size_hidden
    }
    
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    trial.set_user_attr("lr", 1e-3)
    # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    trial.set_user_attr("weight_decay", 1e-5)
    optimizer_config ={
        'lr' : trial.user_attrs['lr'],
        'weight_decay' : trial.user_attrs['weight_decay']
    }
    
    lit_module = LitModule(architecture_type=architecture_type, net_params=mlp_config, optimizer_params=optimizer_config)

    # setup trainer and fit the model
    # pruning callback allows using Optuna pruners as a lightning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_accuracy')
    early_stopping_config = {
        'monitor' : "val_accuracy",
        'min_delta' : 0.00,
        'patience' : 10,
        'mode' : "max",
        'verbose' : False
    }
    early_stop_callback = EarlyStopping(**early_stopping_config)
    trainer = L.Trainer(callbacks=[pruning_callback, early_stop_callback], max_epochs=50)
    # init the datamodule. We don't use augmentation for MLP
    dm = FashionMNISTNoAugment(batch_size=128)
    trainer.fit(lit_module, datamodule=dm)
    
    best_val_accuracy = early_stop_callback.best_score.item()
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
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        save_top_k=0, # don't save the weights for each trial
        mode='max',
        verbose=False
    )
    early_stop_callback = EarlyStopping(**early_stop_config)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100)
    # init the datamodule. For CNNs, we use data augmentation
    dm = FashionMNISTDataModule(batch_size=128, affine_scale=None)
    trainer.fit(lit_module, datamodule=dm)
    
    best_val_accuracy = checkpoint_callback.best_model_score.item()
    return best_val_accuracy


def main():
    torch.set_float32_matmul_precision("high")
    # setup optuna study
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    database_dir = "optuna_databases"
    study_name = f"{architecture_type}3"
    storage_name = f"sqlite:///{database_dir}/{study_name}.db"
    # use Optuna median pruner. Can also use Hyperband, etc...
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=10,
        interval_steps=1, n_min_trials=2
    )
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize',
                                pruner=pruner, load_if_exists=False)
    if architecture_type == "mlp":
        study.optimize(objective_mlp, n_trials=100)
    else:
        study.optimize(objective_cnn, n_trials=5)
    variable_params = study.best_params.copy()
    set_params = study.best_trial.user_attrs.copy()
    all_params = variable_params | set_params
    optimizer_params = {k: all_params[k] for k in optimizer_keys}
    net_params = {k: all_params[k] for k in all_params if k not in optimizer_keys}
    lit_module_config = {'architecture_type': architecture_type,
                         'net_params': net_params,
                         'optimizer_params': optimizer_params}
    full_config = {'lit_module_config': lit_module_config}
    with open(f"{study_name}_best_params.yaml", "w") as f:
        yaml.safe_dump(full_config, f, sort_keys=False)

if __name__ == "__main__":
    main()