"""
Main script using PyTorch Lightning.

Uses MLP and CNN architectures are defined in the `architectures.py` file.
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


def train_and_test(full_config_path: str = None, augment: bool = True,
                   model_checkpoint_path: str = None, seed: int = None):
    """
    Train and test the lightning module with the given configuration.
    Args:
        lit_module_config (dict): Configuration for the LitModule.
        augment (bool): Whether to use data augmentation.
        model_checkpoint_path (str, optional): Path to a model checkpoint to load. Defaults to None.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)  # for reproducibility
    # init the datamodule
    if full_config_path is not None:
        with open(full_config_path, 'r') as file:
            full_config = yaml.safe_load(file)
        lit_module_config = full_config['lit_module_config']
        if 'data_config' in full_config:
            data_config = full_config['data_config']
        else:
            data_config = None
        if 'callbacks_config' in full_config:
            callbacks_config = full_config['callbacks_config']
        else:
            callbacks_config = None

    if data_config is not None:
        if data_config['augment']:
            dm = FashionMNISTDataModule(batch_size=data_config['batch_size'], affine_scale=data_config['affine_scale'])
        else:
            dm = FashionMNISTNoAugment(batch_size=data_config['batch_size'])
    elif augment:
        dm = FashionMNISTDataModule(batch_size=128, affine_scale=None)
    else:
        dm = FashionMNISTNoAugment(batch_size=128)
    
    if callbacks_config is not None:
        checkpoint_config = callbacks_config['checkpoint_config']
        early_stopping_config = callbacks_config['early_stopping_config']
    else:
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
    trainer = L.Trainer(deterministic=True, # reproducibility
                        callbacks=[checkpoint_callback, early_stopping_callback], max_epochs=100)
    
    if model_checkpoint_path is not None:
        model_path = Path(model_checkpoint_path)
        lit_module =LitModule.load_from_checkpoint(model_path)
        print(lit_module)
        print(f"Model hparams: {lit_module.hparams}")
        trainer.test(lit_module, datamodule=dm)
    else:
        lit_module = LitModule(**lit_module_config)
        print(lit_module)
        print(f"Model hparams: {lit_module.hparams}")
        trainer.fit(lit_module, datamodule=dm)
        # Load and test the best model from checkpoint.
        # Due to early stopping, the best model might not be the last one trained.
        trainer.test(ckpt_path="best", datamodule=dm) # test the best version before EarlyStopping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Path to checkpoint to load")
    parser.add_argument("--config", type=str,
                        help="Path to configuration file")
    parser.add_argument("--augment", action="store_true",
                        help="Whether to use data augmentation")
    
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    train_and_test(full_config_path=args.config, augment=args.augment,
                   model_checkpoint_path=args.model)
    

if __name__ == '__main__':
    main()