from PIL import Image
import os
from pathlib import Path

# Replace this with your actual dataset root
data_dir = "waste_data_png"  # where your .pngs are
output_dir = "waste_data_jpg"  # converted .jpgs will go here

Path(output_dir).mkdir(parents=True, exist_ok=True)

for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".png"):
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert("RGB")  # convert to RGB to save as jpg

            rel_path = os.path.relpath(img_path, data_dir)
            new_path = os.path.join(output_dir, Path(rel_path).with_suffix(".jpg"))

            Path(os.path.dirname(new_path)).mkdir(parents=True, exist_ok=True)
            img.save(new_path, "JPEG")

print("✅ All .png files converted to .jpg.")
