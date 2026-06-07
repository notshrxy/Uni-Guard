"""
This script is used to register students in bulk.

It reads student information from a CSV file and a directory of photos.
It then uses the FacePipeline to extract face embeddings and register the students in the FaceDatabase.

Processes a CSV list and a directory of profile photos to register students and generate embeddings in bulk.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from facial_features.facial_pipeline import FacePipeline
from facial_features.facial_database import FaceDatabase

def bulk_register(csv_path, photo_dir):
    """
    Registers students from a CSV and a directory of photos.
    
    Expected CSV: roll_no, name, gender, Year, tag_color
    Expected Photos: roll_no.jpg or roll_no.png in photo_dir
    """
    if not os.path.exists(csv_path):
        print(f"[Error] CSV not found: {csv_path}")
        return

    # Load CSV and ensure roll_no is read as string
    df = pd.read_csv(csv_path, dtype={'roll_no': str})
    # Clean column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Drop completely empty rows
    df = df.dropna(how='all')
    
    # Initialize Pipelines
    face_pipeline = FacePipeline()
    face_db = FaceDatabase()
    
    print(f"[BulkRegister] Starting registration for {len(df)} students...")
    
    success_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Skip row if roll_no is missing
        if pd.isna(row['roll_no']):
            continue
            
        roll_no = str(row['roll_no']).split('.')[0] # Second safety check for .0
        name = row['name']
        gender = row['gender']
        year = row['year']
        tag_color = row['tag_color']
        
        # Find photo
        photo_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            p = Path(photo_dir) / f"{roll_no}{ext}"
            if p.exists():
                photo_path = p
                break
        
        if not photo_path:
            print(f"  [Skip] Photo not found for {name} ({roll_no})")
            continue
            
        # Process image
        import cv2
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"  [Error] Could not read image: {photo_path}")
            continue
            
        # Extract face
        faces = face_pipeline.app.get(img) if face_pipeline.app else []
        if not faces:
            print(f"  [Error] No face detected in photo for {name}")
            continue
            
        # Take the most prominent face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        # Register
        face_db.register_face(
            roll_no=roll_no,
            name=name,
            gender=gender,
            year=year,
            tag_color=tag_color,
            embedding=face.normed_embedding,
            face_image=img
        )
        success_count += 1
        
    print(f"\n[BulkRegister] Done! Successfully registered {success_count}/{len(df)} students.")

if __name__ == '__main__':
    csv = 'student_list.csv'
    photos = 'student_photos'
    bulk_register(csv, photos)
