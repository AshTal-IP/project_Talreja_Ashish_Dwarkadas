import os

# Set your dataset root directory here
dataset_root = "/storage/ashishtalreja/cleaned_waste_data_jpg/images/images"

# Output file path
output_file = "waste_categories.txt"

# Get all first-level subdirectories (categories)
category_folders = [
    folder for folder in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, folder))
]

# Save to txt file
with open(output_file, "w") as f:
    for category in sorted(category_folders):
        f.write(category + "\n")

print(f"Saved {len(category_folders)} category names to '{output_file}'")
