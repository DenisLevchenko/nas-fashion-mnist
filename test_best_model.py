from pathlib import Path
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from config import *
from lightning_definitions import *


# Check for a file in the current directory
model_file_name = 'best_model.pth'
model_path = Path(model_file_name)
haparams_file_name = 'hparams.yaml'
hparams_path = Path(haparams_file_name)

dm = FashionMNISTDataModule(batch_size=data_config['batch_size'], affine_scale=None) 

checkpoint_callback = ModelCheckpoint(**checkpoint_config)
early_stop_callback = EarlyStopping(**early_stop_config)
trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=100)

if model_path.is_file():
    print(f"Model '{model_file_name}' exists in the current directory. Loading model.")
    model = LitModule.load_from_checkpoint(model_path)
else:
    print(f"No best model with name '{model_file_name}' exist in the current directory.")
    if hparams_path.is_file():
        print(f"Loading hyperparameters from file '{haparams_file_name}'")
        with open(hparams_path, 'r') as file:
            model_config = yaml.safe_load(file)
        # init lightning module
        model = LitModule(model_config)
        # train the model
        trainer.fit(model, datamodule=dm)
    else:
        raise ValueError('No saved model or hyperparameters')

print(f"Model hparams: {model.hparams}")
trainer.test(model, datamodule=dm)

