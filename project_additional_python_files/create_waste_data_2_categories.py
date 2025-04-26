import os
import shutil

# Define the mapping of categories to recyclability
category_to_label = {
    'aerosol_cans': 'recyclable',
    'aluminum_food_cans': 'recyclable',
    'aluminum_soda_cans': 'recyclable',
    'cardboard_boxes': 'recyclable',
    'cardboard_packaging': 'recyclable',
    'clothing': 'non_recyclable',
    'coffee_grounds': 'non_recyclable',
    'disposable_plastic_cutlery': 'non_recyclable',
    'eggshells': 'non_recyclable',
    'food_waste': 'non_recyclable',
    'glass_beverage_bottles': 'recyclable',
    'glass_cosmetic_containers': 'recyclable',
    'glass_food_jars': 'recyclable',
    'magazines': 'recyclable',
    'newspaper': 'recyclable',
    'office_paper': 'recyclable',
    'paper_cups': 'non_recyclable',
    'plastic_cup_lids': 'non_recyclable',
    'plastic_detergent_bottles': 'recyclable',
    'plastic_food_containers': 'recyclable',
    'plastic_shopping_bags': 'non_recyclable',
    'plastic_soda_bottles': 'recyclable',
    'plastic_straws': 'non_recyclable',
    'plastic_trash_bags': 'non_recyclable',
    'plastic_water_bottles': 'recyclable',
    'shoes': 'non_recyclable',
    'steel_food_cans': 'recyclable',
    'styrofoam_cups': 'non_recyclable',
    'styrofoam_food_containers': 'non_recyclable',
    'tea_bags': 'non_recyclable'
}

# Directory containing the category folders with .jpg files
source_dir = 'cleaned_waste_data_jpg/images/images'

# Directory where the 'waste_data' folder will be created
base_dir = os.getcwd()

# Create the waste_data directory if it doesn't exist
waste_data_dir = os.path.join(base_dir, 'waste_data')
if not os.path.exists(waste_data_dir):
    os.mkdir(waste_data_dir)

# Create the recyclable and non_recyclable subdirectories
recyclable_dir = os.path.join(waste_data_dir, 'recyclable')
non_recyclable_dir = os.path.join(waste_data_dir, 'non_recyclable')

if not os.path.exists(recyclable_dir):
    os.mkdir(recyclable_dir)

if not os.path.exists(non_recyclable_dir):
    os.mkdir(non_recyclable_dir)

# Initialize naming counter
naming_counter = 1

# Loop through the category folders
for category, label in category_to_label.items():
    category_folder = os.path.join(source_dir, category)
    
    # Check if the category folder exists
    if os.path.isdir(category_folder):
        # Check 'default' and 'real_world' subdirectories
        for subfolder in ['default', 'real_world']:
            subfolder_path = os.path.join(category_folder, subfolder)
            
            if os.path.isdir(subfolder_path):
                # Determine target directory based on recyclability
                target_dir = recyclable_dir if label == 'recyclable' else non_recyclable_dir
                
                # Loop through all .jpg files in the subfolder
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.jpg'):
                        source_file = os.path.join(subfolder_path, filename)
                        
                        # Format the new filename with five-digit convention (e.g., 00001.jpg)
                        new_filename = f"{naming_counter:05d}.jpg"
                        target_file = os.path.join(target_dir, new_filename)
                        
                        # Move and rename the file to the appropriate folder
                        shutil.move(source_file, target_file)
                        
                        print(f"Moved and renamed {filename} to {new_filename}")
                        
                        # Increment the naming counter for the next file
                        naming_counter += 1
