# Neural net architecture optimization on Fashion-MNIST
Using `Optuna`, `PyTorch`, and `Lightning`.
Monitoring on `Tensorboard`.

Recreate python environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Project Structure
### `architectures.py`
defines the broad neural net `PyTorch` architectures:  
1. `MLPBasic`, a basic MLP with fixed number of layers and nodes. Used as a baseline.
2. `MLP`, a configurable MLP implementation, number of layers and nodes per layer are set by the user.
3. `CNNBasic`, a basic fixed CNN architecture with 3 convolutional layers. Used as a baseline for CNN, to compare with the MLP architecture and to improve on with optimization.
4. `CNNRich`, a more involved, fully configurable CNN implementation.

### `lightning_definitions.py`
defines the main `LitModule` and the `FashionMNISTDataModule`, inherited from standard `lightning` classes.  
The `LitModule` encapsulates any of the neural net architectures above with logic for training, validating, logging, testing, including the optimizer setup... It inherits all the standard boilerplate from the `LightningModule`.  
The `FashionMNISTDataModule` handles all the standard procedures for downloading and preprocessing the `Fashion-MNIST` dataset, does the normalization and augmentation, wraps the dataset in dataloaders used throughout training, validating, and testing.

### `optumize.py`
does the hyperparameter optimization of the `LitModule` on the `FashionMNISTDataModulee` using `Optuna`, saving results in a `.db` file that can be explored and visualized later to understand the importance and best values for the hyperparameters, whether another optimization is needed, etc. It saves the best hyperparameters found in a `.yaml` file.
By default uses Optuna's TPE (tree Parzen estimators) sampler and median pruner to efficiently explore the hyperparameter space.

### `test_best_model.py`
will attempt to load the best model weights saved in a `.pth` file into a `LitModule` and test it on the test data not seen during training and optimization.  
If no such file is found, it will load the best hyperparameters from a `.yaml` file, then train and test it. Model layout and basic statistics are displayed in the terminal. The full training, validation, and testing is logged, logs kept in the `lightning_logs` folder. One can run `tensorboard --logdir lightning_logs` to start `tensorboard` and explore the full training dynamic, compare between runs, etc.


