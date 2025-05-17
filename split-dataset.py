# Run this script from your project directory
import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Source directory containing all images by class
SOURCE_DIR = 'dataset-raw'  # Update this to your source directory

# Destination directory for train/val/test splits
DEST_DIR = 'dataset'

# Create destination directory if it doesn't exist 
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)


# Create destination directories
for split in ['train', 'valid', 'test']:
    for class_name in ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']:
        os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

# Define image counts for each split by class
class_split_counts = [
    {'class': 'cardboard', 'train': 282, 'val': 60, 'test': 61},
    {'class': 'glass', 'train': 350, 'val': 75, 'test': 76},
    {'class': 'metal', 'train': 287, 'val': 61, 'test': 62},
    {'class': 'paper', 'train': 415, 'val': 89, 'test': 90},
    {'class': 'plastic', 'train': 337, 'val': 72, 'test': 73},
    {'class': 'trash', 'train': 95, 'val': 20, 'test': 22}
]

# Split data for each class
for class_data in class_split_counts:
    class_name = class_data['class']
    class_dir = os.path.join(SOURCE_DIR, class_name)
    
    # Get all image files for this class
    all_files = [f for f in os.listdir(class_dir) 
                if os.path.isfile(os.path.join(class_dir, f)) and 
                f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle files
    random.shuffle(all_files)
    
    # Split into train, validation, and test sets
    train_count = class_data['train']
    val_count = class_data['val']
    
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count+val_count]
    test_files = all_files[train_count+val_count:]
    
    # Copy files to their respective directories
    for file_name in train_files:
        shutil.copy(
            os.path.join(class_dir, file_name),
            os.path.join(DEST_DIR, 'train', class_name, file_name)
        )
    
    for file_name in val_files:
        shutil.copy(
            os.path.join(class_dir, file_name),
            os.path.join(DEST_DIR, 'valid', class_name, file_name)
        )
    
    for file_name in test_files:
        shutil.copy(
            os.path.join(class_dir, file_name),
            os.path.join(DEST_DIR, 'test', class_name, file_name)
        )
    
    print(f"Class {class_name}: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")

print("Data split complete!")