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
from config import *


# haparams_file_name = 'cnn_rich_am_best_params.yaml'
# hparams_path = Path(haparams_file_name)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true",
                        help="Load saved model")
    parser.add_argument("--model", type=str,
                        help="Path to checkpoint to load")
    parser.add_argument("--config", type=str,
                        help="Path to configuration file")
    
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")

    # init the datamodule
    if data_config['augment']:
        dm = FashionMNISTDataModule(batch_size=data_config['batch_size'], affine_scale=None)
    else:
        dm = FashionMNISTNoAugment(batch_size=data_config['batch_size'])
    
   
    checkpoint_callback = ModelCheckpoint(**checkpoint_config)
    early_stop_callback = EarlyStopping(**early_stop_config)
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100)
    
    # init the LightningModule
    if args.load:
        model_file_name = args.model
        model_path = Path(model_file_name)
        lit_module =LitModule.load_from_checkpoint(model_path)
        print(lit_module)
        trainer.test(lit_module, datamodule=dm)
        print(f"Model hparams: {lit_module.hparams}")
    else:
        config_file_name = args.config
        config_path = Path(config_file_name)
        with open(config_path, 'r') as file:
            full_config = yaml.safe_load(file)
        lit_module = LitModule(**full_config)
        print(lit_module)
        trainer.fit(lit_module, datamodule=dm)
        # Load and test the best model from checkpoint
        # best_model_path = checkpoint_callback.best_model_path
        # best_model = LitModule.load_from_checkpoint(best_model_path)
        # trainer.test(best_model, datamodule=dm)
        trainer.test(ckpt_path="best", datamodule=dm) # test the best version before EarlyStopping
        print(f"Model hparams: {lit_module.hparams}")


if __name__ == '__main__':
    main()