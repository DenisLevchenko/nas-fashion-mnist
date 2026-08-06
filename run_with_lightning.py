"""
Main script using PyTorch Lightning.

Uses MLP and CNN architectures defined in the `architectures.py` file.
Also does validation and testing, logs everything.
The logs are accessible through TensorBoard.
Uses early stopping and saves the best model.
"""

from pathlib import Path
import argparse
import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning_definitions import LitModule, FashionMNISTDataModule, FashionMNISTNoAugment


def train_and_test(lit_module_config: dict, data_config: dict,
                   callbacks: list | None = None, seed: int | None = None):
    """
    Train and test the lightning module with the given configuration.
    Args:
        lit_module_config (dict): Configuration for the LitModule.
        data_config (dict): Configuration for the data module.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)  # for reproducibility
    
    # init the datamodule
    if data_config['augment']:
        dm = FashionMNISTDataModule(batch_size=data_config['batch_size'])
    else:
        dm = FashionMNISTNoAugment(batch_size=data_config['batch_size'])
    
    if callbacks is None:
        checkpoint_config = {
            'monitor' : 'val_accuracy',
            'save_top_k' : 1, # save only the best model
            'mode' : 'max', # max for maximizing, min for minimizaing
            'verbose' : False
        }
        early_stopping_config = {
            'monitor' : "val_accuracy",
            'min_delta' : 0.00,
            'patience' : 10,
            'mode' : "max",
            'verbose' : False
        }
        checkpoint_callback = ModelCheckpoint(**checkpoint_config)
        early_stopping_callback = EarlyStopping(**early_stopping_config)
        callbacks = [checkpoint_callback, early_stopping_callback]
    
    trainer = L.Trainer(deterministic=True, # reproducibility
                        callbacks=callbacks, max_epochs=100)
    torch.use_deterministic_algorithms(True, warn_only=True) # reproducible whenever possible
    lit_module = LitModule(**lit_module_config)
    print(lit_module)
    print(f"Model hparams: {lit_module.hparams}")
    trainer.fit(lit_module, datamodule=dm)
    # Load and test the best model from checkpoint.
    # Due to early stopping, the best model might not be the last one trained.
    results = trainer.test(ckpt_path="best", datamodule=dm) # test the best version before EarlyStopping stopped the training
    return results
    

def train_and_test_from_yaml(config_path: str, data_config: dict | None = None,
                             augment: bool | None = False, bs: int | None = 128,
                             seed: int | None = None):
    """
    Train and test the lightning module with configuration given in a .yaml file.
    Args:
        config_path (str): Path to the configuration file.
        data_config (dict): Configuration for the data module if separate.
        augment (bool): Whether to use data augmentation. Recommended for CNNs.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    full_config_path = Path(config_path)
    with open(full_config_path, 'r') as file:
        full_config = yaml.safe_load(file)
    lit_module_config = full_config['lit_module_config']
    
    if data_config is None:
        if 'data_config' in full_config:
            data_config = full_config['data_config']
        else:
            data_config = {'augment': augment, 'batch_size': bs}
    if 'callbacks_config' in full_config:
        callbacks_config = full_config['callbacks_config']
        checkpoint_config = callbacks_config['checkpoint_config']
        early_stopping_config = callbacks_config['early_stopping_config']
        checkpoint_callback = ModelCheckpoint(**checkpoint_config)
        early_stopping_callback = EarlyStopping(**early_stopping_config)
        callbacks = [checkpoint_callback, early_stopping_callback]
    else:
        callbacks = None
    return train_and_test(lit_module_config=lit_module_config, data_config=data_config,
                          callbacks = callbacks, seed=seed)


def load_and_test(model_checkpoint_path: str, data_config: dict, seed: int | None = None):
    """
    Load a model from a checkpoint and test it.
    Args:
        model_checkpoint_path (str): Path to the model checkpoint.
        data_config (dict): Configuration for the data module.
        seed (int, optional): Seed for reproducibility. Defaults to None.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)  # for reproducibility
    
    # init the datamodule
    if data_config['augment']:
        dm = FashionMNISTDataModule(batch_size=data_config['batch_size'])
    else:
        dm = FashionMNISTNoAugment(batch_size=data_config['batch_size'])
    
    model_path = Path(model_checkpoint_path)
    lit_module = LitModule.load_from_checkpoint(model_path)
    print(lit_module)
    print(f"Model hparams: {lit_module.hparams}")
    
    trainer = L.Trainer(deterministic=True)  # for reproducibility
    results = trainer.test(lit_module, datamodule=dm)
    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str,
                       help="Path to checkpoint to load")
    group.add_argument("--config", type=str,
                       help="Path to configuration file")
    
    parser.add_argument("--augment", action="store_true",
                        help="Whether to use data augmentation")
    parser.add_argument("--bs", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if args.config is not None:
        train_and_test_from_yaml(config_path=args.config, augment=args.augment,
                                 bs=args.bs, seed=args.seed)
    else:
        data_config = {'augment': args.augment, 'batch_size': args.bs}
        load_and_test(model_checkpoint_path=args.model,
                      data_config=data_config, seed=args.seed)
    

if __name__ == '__main__':
    main()