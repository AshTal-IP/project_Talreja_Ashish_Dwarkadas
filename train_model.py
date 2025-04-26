from model import RecyclingResNet as TheModel

# Trainer
from train import train_model as the_trainer

# Dataset and Dataloader
from dataset import WasteDataset as TheDataset
from dataset import create_dataloader as the_dataloader

from predict import predict_images as the_predictor


from config import batch_size as the_batch_size
from config import epochs as total_epochs

from config import learning_rate, train_data_path, val_data_path, weight_decay

import torch
from torch import nn, optim

model = TheModel()
train_loader = the_dataloader(train_data_path, batch_size=the_batch_size)
val_loader = the_dataloader(val_data_path, batch_size=the_batch_size, shuffle=False)

# Loss and Optimizer with Weight Decay
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Start training
the_trainer(
    model=model,
    num_epochs=total_epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer
)