# Model
from model import RecyclingResNet as TheModel

# Trainer
from train import train_model as the_trainer

# Dataset and Dataloader
from dataset import WasteDataset as TheDataset
from dataset import create_dataloader as the_dataloader

from predict import predict_images as the_predictor

# # Config
# from config import batch_size as the_batch_size
# from config import epochs as total_epochs