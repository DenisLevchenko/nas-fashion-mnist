"""
Main script using PyTorch Lightning.

Only uses a basic MLP for now.
Also does validation and testing, logs everything.
The logs are accessible through TensorBoard.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from architectures import MLP, CNN

# define the LightningModule
class MLPLight(L.LightningModule):
    def __init__(self, net:nn.Module, loss):
        super().__init__()
        self.net = net
        self.loss = loss
        # self.val_accuracy = Accuracy(task='multiclass', num_classes=10)
        # self.test_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.val_accuracy = MulticlassAccuracy(num_classes=10)
        self.test_accuracy = MulticlassAccuracy(num_classes=10)
        
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.net(x)
        loss = self.loss(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.net(x)
        val_loss = self.loss(z, y)
        val_acc = self.val_accuracy(z, y.argmax(dim=1))
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", val_acc, prog_bar=True)
        # self.validation_step_outputs.append(val_acc)
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.net(x)
        test_loss = self.loss(z, y)
        test_acc = self.test_accuracy(z, y.argmax(dim=1))
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_acc)
        # self.test_step_outputs.append(test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    # don't think we need these below
    # def on_validation_epoch_end(self):
    #     self.val_accuracy.reset()
    #     self.validation_step_outputs.clear()

    # def on_test_epoch_end(self):
    #     self.test_accuracy.reset()
    #     self.test_step_outputs.clear()


loss_fn = nn.CrossEntropyLoss()
# init the model
model = MLPLight(MLP(), loss_fn)

# setup data
training_data = datasets.FashionMNIST(
    root='~/Coding/torch_tutorial',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float)
        .scatter_(0, torch.tensor(y), value=1)
    )
)

test_data = datasets.FashionMNIST(
    root='~/Coding/torch_tutorial',
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float)
        .scatter_(0, torch.tensor(y), value=1)
    )
)

# use 20% of training data for validation
train_set_size = int(len(training_data) * 0.8)
valid_set_size = len(training_data) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(training_data, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# train the model
early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=5, verbose=False, mode="max")
trainer = L.Trainer(callbacks=[early_stop_callback], max_epochs=50)
# trainer = L.Trainer(limit_train_batches=100, max_epochs=15)
trainer.fit(model, train_loader, val_loader)

trainer.test(model, dataloaders=test_loader)
