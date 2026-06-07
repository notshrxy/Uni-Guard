#Scripts used to crop, structure, and prepare face images for training or reference registration.

import os
import shutil
import random
from pathlib import Path
import yaml

# Configurations
DATASET_DIR = Path("dataset")
LABELS_DIR = Path("labels")
SPLIT_RATIO = 0.8  # 80% training, 20% validation

# New YOLOv8 Directory Structure
YOLO_DIR = Path("yolo_dataset")
IMAGES_TRAIN = YOLO_DIR / "images" / "train"
IMAGES_VAL = YOLO_DIR / "images" / "val"
LABELS_TRAIN = YOLO_DIR / "labels" / "train"
LABELS_VAL = YOLO_DIR / "labels" / "val"

def setup_directories():
    """Create the YOLOv8 directory structure."""
    print("[Prepare] Setting up directories...")
    for path in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        path.mkdir(parents=True, exist_ok=True)

def prepare_data():
    """Split and move the data."""
    print("[Prepare] Gathering labeled images...")
    
    # Get all labeled txt files (ignoring classes.txt)
    label_files = [f for f in LABELS_DIR.glob("*.txt") if f.name != "classes.txt"]
    
    if not label_files:
        print("[Error] No label files found in 'labels/' directory.")
        return

    # Shuffle the dataset for randomness
    random.seed(42)
    random.shuffle(label_files)
    
    # Split
    split_index = int(len(label_files) * SPLIT_RATIO)
    train_files = label_files[:split_index]
    val_files = label_files[split_index:]
    
    print(f"[Prepare] Found {len(label_files)} labeled images.")
    print(f"[Prepare] Splitting: {len(train_files)} for Training, {len(val_files)} for Validation.")
    
    # Helper to move files
    def move_files(files, split_type):
        img_dest = IMAGES_TRAIN if split_type == "train" else IMAGES_VAL
        lbl_dest = LABELS_TRAIN if split_type == "train" else LABELS_VAL
        
        for lbl_path in files:
            img_path = DATASET_DIR / f"{lbl_path.stem}.jpg"
            
            if img_path.exists():
                shutil.copy(img_path, img_dest / img_path.name)
                shutil.copy(lbl_path, lbl_dest / lbl_path.name)
            else:
                print(f"[Warning] Image not found for label: {lbl_path.name}")

    print("[Prepare] Copying files to YOLO structure...")
    move_files(train_files, "train")
    move_files(val_files, "val")

def create_yaml():
    """Create the data.yaml file required by YOLOv8."""
    print("[Prepare] Generating data.yaml...")
    
    # Read classes from classes.txt
    classes_path = LABELS_DIR / "classes.txt"
    if not classes_path.exists():
        print("[Error] classes.txt not found in labels directory!")
        return
        
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
        
    yaml_data = {
        "path": str(YOLO_DIR.absolute()),  # Absolute path is safer for YOLO
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"[Prepare] Created {yaml_path}")

if __name__ == "__main__":
    setup_directories()
    prepare_data()
    create_yaml()
    print("\n[Done] Dataset is successfully prepared! You are ready to train.")
