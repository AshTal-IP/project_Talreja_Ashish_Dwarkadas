import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
from config import resize_x, resize_y

class WasteDataset(Dataset):
    def __init__(self, data_dir=None, filepaths=None, labels=None, transform=None):
        """
        Supports both directory-based (training) and direct filepath-based (inference) initialization.
        """
        if data_dir is not None:
            # Training/Validation Mode: Load from directory
            self.data_dir = data_dir
            # Assuming classes are in subdirectories named 'non_recyclable' and 'recyclable'
            self.classes = ['non_recyclable', 'recyclable']
            self.filepaths = []
            self.labels = []
            for idx, cls in enumerate(self.classes):
                cls_dir = os.path.join(data_dir, cls)
                for img in os.listdir(cls_dir):
                    self.filepaths.append(os.path.join(cls_dir, img))
                    self.labels.append(idx)
        elif filepaths is not None:
            # Inference Mode: Direct filepaths
            self.filepaths = filepaths
            self.labels = labels if labels is not None else [0] * len(filepaths)
        else:
            raise ValueError("Either `data_dir` or `filepaths` must be provided.")
        
        # Default transform if none provided
        # Note: The transform for inference should not include augmentations
        # that would change the image content.
        # For training, we can include augmentations
        self.transform =  transform or transforms.Compose([
            transforms.Resize((resize_x, resize_y)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        label = self.labels[idx]
        img = self.transform(img)
        return img, label

def create_dataloader(data_dir, batch_size, shuffle=True, transform=None):
    """For backward compatibility with existing `train.py`."""
    
    dataset = WasteDataset(data_dir=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader