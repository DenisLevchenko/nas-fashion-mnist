"""
Main training script using only raw PyTorch.

Will load the FashionMNIST dataset if it can't find it locally.
Choose model 'mlp' or 'cnn', this will load the architectures
from architectures.py.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from architectures import MLP, CNN

# Load training and test data
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

# Define the device to be used. Use GPU if available, otherwise CPU
if torch.cuda.is_available():
    device = 'cuda' # nvidia gpu
elif torch.backends.mps.is_available():
    device = 'mps' # mac silicon gpu
else:
    device = 'cpu'

# Uncomment below to force cpu
# device = 'cpu'
print(f'Using device: {device}')

# Choose a model architecture: MLP or CNN, set appropriate image shapes.
model_name = 'mlp'
# model_name = 'cnn'
if model_name == 'mlp':
    model = MLP().to(device, dtype=torch.float32)
    train_images = training_data.data.to(device=device, dtype=torch.float32)
    test_images = test_data.data.to(device=device, dtype=torch.float32)
elif model_name == 'cnn':
    model = CNN(out_channels=8, kernel_size=3, padding='same',
                dilation=0, act_function=nn.ReLU(),
                dropout_rate=0.2).to(device, dtype=torch.float32)
    train_images = training_data.data.to(
        device=device, dtype=torch.float32).unsqueeze(1) # unsqueeze for cnns
    test_images = test_data.data.to(
        device=device, dtype=torch.float32).unsqueeze(1) # unsqueeze for cnns
else:
    raise ValueError('Unsupported model')

# Get targets too, combine with images in TensorDataset
train_targets = training_data.targets.to(device=device)
test_targets = test_data.targets.to(device=device)
trainset = TensorDataset(train_images, train_targets)
testset = TensorDataset(test_images, test_targets)

# Build dataloaders
train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=64, shuffle=True)

# classic way using original datasets. Can't move the whole thing onto a GPU.
# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Define train and test loops

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test_loop(dataloader, model, loss_fn) -> float:
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n'
          f'Accuracy: {(100*correct):>0.1f}%, '
          f'Avg loss: {test_loss:>8f} \n')
    return correct

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
# and Optimizer with its parameters 
learning_rate = 2e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Set the number of epochs and parameters for early stopping
epochs = 50
best_test_acc = 0
patience = 5
no_improvement_counter = 0

for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    current_test_acc = test_loop(test_dataloader, model, loss_fn)
    if current_test_acc > best_test_acc:
        no_improvement_counter = 0
        best_test_acc = current_test_acc
        torch.save(model, f'{model_name}.pth')
        print(f'Saved the model with accuracy {(100*best_test_acc):>0.1f}% \n')
    if current_test_acc < best_test_acc:
        no_improvement_counter += 1
    if no_improvement_counter > patience:
        print(f'Early stopping early stopped at Epoch {t}!')
        break
print(f'Done! Best model had {(100*best_test_acc):>0.1f}% accuracy')