import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm
import uuid
from PIL import Image


# Input dir (where 'waste_data' is located)
original_dir = 'waste_data'

# Output dir (new augmented dataset)
augmented_base = 'augmented_waste_data'
os.makedirs(augmented_base, exist_ok=True)

# Class folders
classes = ['recyclable', 'non_recyclable']
for cls in classes:
    os.makedirs(os.path.join(augmented_base, cls), exist_ok=True)

# Minimal augmentation: rotate, zoom, horizontal flip
augmentor = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment and save function
def process_class(class_name):
    src_folder = os.path.join(original_dir, class_name)
    tgt_folder = os.path.join(augmented_base, class_name)
    
    all_imgs = sorted(os.listdir(src_folder))
    img_count = 0

    for img_name in tqdm(all_imgs, desc=f"Processing {class_name}"):
        if not img_name.lower().endswith('.jpg'):
            continue
        img_path = os.path.join(src_folder, img_name)
        img = load_img(img_path)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, 0)

        # Save original
        new_name = f"{img_count:05d}.jpg"
        img.save(os.path.join(tgt_folder, new_name))
        img_count += 1

        # Generate 3 augmentations
        aug_iter = augmentor.flow(arr, batch_size=1)
        for _ in range(3):
            aug_img = next(aug_iter)[0].astype(np.uint8)
            # aug_pil = load_img(img_path).fromarray(aug_img)
            aug_pil = Image.fromarray(aug_img.astype('uint8'))
            new_name = f"{img_count:05d}.jpg"
            aug_pil.save(os.path.join(tgt_folder, new_name))
            img_count += 1

# Run augmentation for both classes
for cls in classes:
    process_class(cls)

# === SPLITTING TO TRAIN / VAL / TEST ===

def split_to_sets(src_dir, out_dir='augmented_waste_data', splits=(0.7, 0.2, 0.1)):
    for subset in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(out_dir, subset, cls), exist_ok=True)

    for cls in classes:
        files = sorted(os.listdir(os.path.join(src_dir, cls)))
        files = [f for f in files if f.endswith('.jpg')]
        paths = [os.path.join(src_dir, cls, f) for f in files]

        train, temp = train_test_split(paths, train_size=splits[0], shuffle=True)
        val, test = train_test_split(temp, train_size=splits[1]/(splits[1]+splits[2]))

        def move_files(file_list, subset):
            for f in file_list:
                shutil.move(f, os.path.join(out_dir, subset, cls, os.path.basename(f)))

        move_files(train, 'train')
        move_files(val, 'val')
        move_files(test, 'test')

# Split now
split_to_sets(augmented_base)

# === Log final image count per split to a text file ===
log_path = os.path.join(augmented_base, "dataset_summary.txt")
with open(log_path, 'w') as f:
    f.write("📊 Final image count per class per split:\n")
    for subset in ['train', 'val', 'test']:
        f.write(f"\n[{subset.upper()}]\n")
        for cls in classes:
            folder = os.path.join(augmented_base, subset, cls)
            count = len([img for img in os.listdir(folder) if img.endswith('.jpg')])
            f.write(f"  {cls:17s}: {count} images\n")

print(f"\nDataset summary saved to {log_path}")


print("\n Augmentation & dataset splitting completed!")
