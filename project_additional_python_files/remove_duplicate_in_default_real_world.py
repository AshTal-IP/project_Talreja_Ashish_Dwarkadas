import os
from pathlib import Path
from PIL import Image
import imagehash

# Use this to remove duplicates across all subfolders
def remove_duplicates(img_dir, output_dir):
    seen_hashes = set()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path).convert("RGB")
                    h = imagehash.phash(img)

                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        rel = os.path.relpath(path, img_dir)
                        out_path = os.path.join(output_dir, Path(rel).with_suffix(".jpg"))
                        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
                        img.save(out_path, "JPEG")
                except:
                    print(f"⚠️ Skipped corrupted file: {path}")

    print("✅ Deduplication done.")

# Example usage
# remove_duplicates("waste_data_jpg", "cleaned_waste_data_jpg")
