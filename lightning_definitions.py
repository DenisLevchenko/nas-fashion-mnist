"""
LightningModule definitions: model modules, DataModules.
"""

from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets
import torchvision.transforms as T
import lightning as L
from torchmetrics.classification import MulticlassAccuracy
from typing import Tuple, List, Union
from architectures import MLPBasic, MLP, CNN, CNNBasic, CNNExpand, CNNRich, CNN2


# Fashion-MNIST channel statistics (single channel, pre-computed)
_MEAN = (0.2860,)
_STD  = (0.3530,)


architectures = {
    "mlp_basic": MLPBasic,
    "mlp": MLP,
    "cnn_basic": CNNBasic,
    "cnn_rich": CNNRich,
    "cnn_expand": CNNExpand,
    "cnn2": CNN2
}


class LitModule(L.LightningModule):
    def __init__(self, architecture_type: str,
                 net_params: dict, optimizer_params: dict
                 ):
        super().__init__()
        self.save_hyperparameters()
        architecture_class = architectures[architecture_type]
        self.net = architecture_class(**net_params)
        # self.learning_rate = learning_rate
        # self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=10)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.net(x)
        loss = self.loss(z, y)
        acc = self.accuracy(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.net(x)
        val_loss = self.loss(z, y)
        val_acc = self.accuracy(z, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", val_acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.net(x)
        test_loss = self.loss(z, y)
        test_acc = self.accuracy(z, y)
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_acc)
        # self.test_step_outputs.append(test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_params)
        return optimizer


def one_hot(y):
    return torch.zeros(10, dtype=torch.float).scatter_(
        0,
        torch.tensor(y),
        value=1
    )


class FashionMNISTNoAugment(L.LightningDataModule):
    def __init__(self, data_dir: str = "~/Coding/torch_tutorial", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def prepare_data(self):
        # download
        datasets.FashionMNIST(root=self.data_dir,train=True, download=True)
        datasets.FashionMNIST(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        transform = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST stats
        ])
        # load data and split into train and val
        training_data = datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_data = datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # split the train set into two
        train_set_size = int(len(training_data) * 0.8)
        valid_set_size = len(training_data) - train_set_size
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.val_set = random_split(
            training_data, [train_set_size, valid_set_size], generator=seed)
        
        self.test_set = test_data
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          num_workers=8, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          num_workers=8, pin_memory=True, persistent_workers=True)


class FashionMNISTDataModule(L.LightningDataModule):
    """
    LightningDataModule for the Fashion-MNIST dataset.

    - Train split : random horizontal flip + random affine + normalization
    - Val / Test  : normalization only
    - The official test set is used as the test split.
      A configurable fraction of the training set is held out as validation.

    Args:
        data_dir:       Root directory for dataset download / cache.
        val_fraction:   Fraction of the training set used for validation.
        batch_size:     Mini-batch size for all dataloaders.
        num_workers:    Number of workers for DataLoader.
        pin_memory:     Whether to pin memory (recommended when using a GPU).
        seed:           Seed for the train/val split RNG. Set to an int for
                        a fully reproducible split; None gives a random split.
        flip_p:         Probability of a random horizontal flip.
        affine_degrees: Rotation range (±degrees) for RandomAffine.
                        Fashion-MNIST images are 28×28 and orientation carries
                        class information, so keep this small (default 5°).
        affine_translate: Max absolute fraction of total width/height for
                          translation, e.g. (0.1, 0.1).
        affine_scale:   Scale range for RandomAffine. At 28×28 a ±5% zoom
                        moves ~1–2 px — subtle but non-trivial. The transform
                        always outputs the original 28×28 size via crop/pad.
                        Pass None to disable scaling entirely.
    """

    def __init__(
        self,
        data_dir: str = "~/Coding/torch_tutorial",
        val_fraction: float = 0.2,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int | None = 42,
        # --- augmentation hyper-parameters ---
        flip_p: float = 0.5,
        affine_degrees: float = 0,   
        affine_translate: tuple[float, float] = (0.1, 0.1),
        affine_scale: tuple[float, float] | None = (0.95, 1.05),  # ±5%; pass None to disable
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Transforms shared by val and test (no augmentation)
        self._base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])

        # Transforms for the training split (augmentation + normalization)
        self._train_transform = T.Compose([
            T.RandomHorizontalFlip(p=flip_p),
            T.RandomAffine(
                degrees=affine_degrees,
                translate=affine_translate,
                scale=affine_scale,      # None disables scaling
            ),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])

        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download the dataset (called on a single process in DDP)."""
        datasets.FashionMNIST(self.hparams.data_dir, train=True,  download=True)
        datasets.FashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """
        Instantiate and split datasets.
        Called on every process in DDP, after prepare_data.

        Stages
        ------
        'fit'       -> sets up train + val
        'validate'  -> sets up val only
        'test'      -> sets up test only
        None        -> sets up all three
        """
        if stage in ("fit", "validate", None):
            # Load the full training split twice so each subset gets
            # its own transform without any data leakage.
            full_train_aug  = datasets.FashionMNIST(
                self.hparams.data_dir, train=True, transform=self._train_transform
            )
            full_train_base = datasets.FashionMNIST(
                self.hparams.data_dir, train=True, transform=self._base_transform
            )

            n_total = len(full_train_aug)
            n_val   = int(n_total * self.hparams.val_fraction)
            n_train = n_total - n_val

            # Build a seeded generator so the split is reproducible.
            # seed=None falls back to the global RNG (non-deterministic).
            generator = None
            if self.hparams.seed is not None:
                generator = torch.Generator().manual_seed(self.hparams.seed)

            train_indices, val_indices = random_split(
                range(n_total),
                [n_train, n_val],
                generator=generator,
            )

            # Apply augmented transform to train, base transform to val
            self.train_dataset = Subset(full_train_aug,  train_indices.indices)
            self.val_dataset   = Subset(full_train_base, val_indices.indices)

        if stage in ("test", None):
            self.test_dataset = datasets.FashionMNIST(
                self.hparams.data_dir, train=False, transform=self._base_transform
            )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )
