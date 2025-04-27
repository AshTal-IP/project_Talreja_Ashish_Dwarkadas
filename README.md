# Waste Classification using Residual CNN

## Project Description

### Problem Formulation
This project tackles the binary classification of waste materials into **recyclable** (class 1) and **non-recyclable** (class 0) categories using deep learning. Accurate automated waste sorting can significantly improve recycling efficiency and reduce environmental pollution.

**Input**: RGB images of waste items (which would be resized to 128x128px by the model)  
**Output**: Binary prediction (0 or 1) with confidence score

### Dataset
- **Source**: "augmented_waste_data" (which is attached in this repo), which is a augmented and preprocessed version of "Recyclable and Household Waste Classification" (find more info in later section)
- **Class Distribution**:
  - Train: 14,638 recyclable + 12,933 non-recyclable
  - Val: 4,182 recyclable + 3,695 non-recyclable
  - Test: 2,092 recyclable + 1,848 non-recyclable


## Model Description

### Architecture Choice
**RecyclingResNet**: A custom Residual CNN with:
- 3 residual blocks with skip connections
- Batch normalization and dropout (p=0.7)
- Adaptive average pooling
- Sigmoid-activated single-neuron output

**Key Features**:
1. **Residual Blocks**: Prevent vanishing gradients in deep networks
   ```python
   class ResidualBlock(nn.Module):
       def __init__(self, in_channels, out_channels, stride=1):
           super().__init__()
           self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                 stride=stride, padding=1, bias=False)
           self.bn1 = nn.BatchNorm2d(out_channels)
           # ... (shortcut connection if stride/channels change)

2. **Regularization**:

     - L2 weight decay (λ=1e-3)

     - Dropout before final layer

3. **Dynamic Learning Rate**: ReduceLROnPlateau scheduler monitors validation accuracy

# Project Pipeline Instructions

> ⚠️ **Important Warning**: Make sure `Pillow` (PIL) and all required `torch` libraries are installed **before** running the project python files, otherwise the  script will not work properly.

---
>
# **Very Important Note for Prediction**
the_predictor function, which would be imported from the predict.py file, will only accept list of images paths which means a list of strings. This is as was asked in the project instructions.
example:

```python
the_predictor(["data/recyclable/Image_1.png", "data/non_recyclable/Image_2.png"])
```

---
## 1. Training Weights Saving
- If you wish to run `train.py` again, note that the final weights will now be saved in the `_checkpoints` directory instead of the `checkpoints` directory.
- This ensures the original `final_weights.pth` present in the `checkpoints` folder is **not overwritten**, as per the instructions.

---

## 2. Downloading the Raw Dataset
The raw dataset is named **"Recyclable and Household Waste Classification"** ([Kaggle Dataset Link](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification))
 and can be downloaded from Kaggle using the following steps:



```bash
pip install kaggle --user
mkdir -p ~/.kaggle
# Download your Kaggle API key (kaggle.json) from your Kaggle account
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d alistairking/recyclable-and-household-waste-classification
unzip recyclable-and-household-waste-classification.zip -d waste_data_png
```
**Note: This was not the dataset that was used to train the model, the dataset that was preprocessed and the attached "augmented_waste_data" dataset was used for training**
---

## 3. Preprocessing the Raw Dataset
After downloading and extracting the dataset into the `waste_data_png` folder, execute the following Python scripts from the `project_additional_python_files` folder **in order**:

### 3.1 Convert PNG to JPG
**Script:** `png_to_jpg_recyc_dataset.py`
- Converts images from `.png` to `.jpg`.
- Creates a new folder: `waste_data_jpg`.

### 3.2 Remove Duplicate Images
**Script:** `remove_duplicate_in_default_real_world.py`
- Removes duplicate images within each waste category using image hashing.
- Maintains the original folder structure.

### 3.3 Create Two Waste Categories
**Script:** `create_waste_data_2_categories.py`
- Splits the dataset into **Recyclable** and **Non-Recyclable** categories based on the following mapping:

| Recyclable Categories         | Non-Recyclable Categories        |
|:-------------------------------|:---------------------------------|
| aerosol_cans                  | clothing                         |
| aluminum_food_cans            | coffee_grounds                   |
| aluminum_soda_cans            | disposable_plastic_cutlery       |
| cardboard_boxes               | eggshells                        |
| cardboard_packaging           | food_waste                       |
| glass_beverage_bottles        | paper_cups                       |
| glass_cosmetic_containers     | plastic_cup_lids                 |
| glass_food_jars               | plastic_shopping_bags            |
| magazines                     | plastic_straws                   |
| newspaper                     | plastic_trash_bags               |
| office_paper                  | shoes                            |
| plastic_detergent_bottles     | styrofoam_cups                   |
| plastic_food_containers       | styrofoam_food_containers        |
| plastic_soda_bottles          | tea_bags                         |
| plastic_water_bottles         |                                  |
| steel_food_cans               |                                  |

### 3.4 Augment and Split Data
**Script:** `augment_and_split_waste_data.py`
- Augments images (adds 3–4 augmented versions per original image).
- Splits data into **train**, **validation**, and **test** sets.
- Creates the final dataset structure in the `augmented_waste_data` folder.

---

## 4. Required Library Versions
Please ensure the following libraries are installed:

| Library         | Version     |
|:----------------|:------------|
| `torch`         | 2.5.1       |
| `torchaudio`    | 2.5.1+cu121 |
| `torchmetrics`  | 1.6.1       |
| `torchvision`   | 0.20.1+cu121 |
| `Pillow`        | Latest      |

Install them on Linux using the following official command from the PyTorch website:

```bash
pip3 install torch torchvision torchaudio pillow
```

---

## 5. Additional Required Checks
Please make sure you have 
1. cuda compilers and runtime libraries installed on you local machine with cuda-12.1 or higher support
2. It helps if cuDNN library is availabe (cudnn-8.2), it offers highly optimized implementations of deep learning operations
3. The model was trained on an Nvidia GPU


---

## Notes
- If you rerun the training script, the **best weights** will be saved inside the `_checkpoints` directory.
- Please do not manually overwrite any weights unless specifically instructed.

---

# 📌 End of Instructions

