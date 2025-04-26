import torch
from torchvision import transforms
from model import RecyclingResNet as TheModel
from dataset import WasteDataset as TheDataset
from config import resize_x, resize_y
from torch.utils.data import DataLoader  # Import DataLoader directly
import os

def predict_images(list_of_image_paths):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TheModel().to(device)
    model.load_state_dict(
        torch.load("checkpoints/final_weights.pth", map_location=device)
    )
    model.eval()

    # Inference transforms (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor()
    ])

    # Create dataset directly with file paths
    dataset = TheDataset(filepaths=list_of_image_paths, transform=transform)

    # Manually create DataLoader (can't use `the_dataloader` since it assumes data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Run prediction
    predictions = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            # preds = (outputs > 0.5).int().squeeze(1).tolist()  # Squeeze for scalar output
            preds = (outputs > 0.5).int().flatten().tolist()  # Use flatten() instead of squeeze(1)
            predictions.extend(preds)

    return ["recyclable" if p == 1 else "non_recyclable" for p in predictions]

if __name__ == "__main__":
    # Example usage
    # test_images = ["data/recyclable/00677.jpg", "data/non_recyclable/00973.jpg"]
    test_images=["data/recyclable/Image_1.png", "data/non_recyclable/Image_2.png", 
                 "data/recyclable/Image_3.png", "data/non_recyclable/Image_4.png", 
                 "data/recyclable/Image_5.png", "data/non_recyclable/Image_6.png", 
                 "data/recyclable/Image_7.png", "data/non_recyclable/Image_8.png", 
                 "data/recyclable/Image_9.png", "data/non_recyclable/Image_10.png"]
    print(predict_images(test_images))
