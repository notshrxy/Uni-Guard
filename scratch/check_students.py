"""
A command-line script to inspect student registration status and verify if their face reference embeddings exist in the database.

"""
import sqlite3
from pathlib import Path
import sys
import cv2
sys.path.append('.')
from facial_features.facial_database import FaceDatabase

def _auto_generate_embedding(roll_no, name, gender, year, tag_color):
    from facial_features.facial_pipeline import FacePipeline
    
    photo_dir = Path("student_photos")
    photo_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
        p = photo_dir / f"{roll_no}{ext}"
        if p.exists():
            photo_path = p
            break
    
    if not photo_path:
        return False, f"{roll_no}.jpg is missing in student_photos"
        
    try:
        img = cv2.imread(str(photo_path))
        if img is None:
            return False, f"Could not read image {photo_path.name} in student_photos"
            
        pipeline = FacePipeline()
        if pipeline.app is None:
            return False, "InsightFace model not initialized"
            
        faces = pipeline.app.get(img)
        if not faces:
            return False, f"No face detected in {photo_path.name}"
            
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        face_db = FaceDatabase()
        face_db.register_face(
            roll_no=roll_no,
            name=name,
            gender=gender,
            year=year,
            tag_color=tag_color,
            embedding=face.normed_embedding,
            face_image=img
        )
        return True, "Success"
    except Exception as e:
        return False, f"Generation error: {str(e)}"

db_path = Path('face_db/uniguard.db')
ref_dir = Path('face_db/reference_faces')
fdb = FaceDatabase()
embeddings_db = fdb.embeddings_db

conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute('SELECT roll_no, name, gender, batch_year, tag_color FROM students')
students = c.fetchall()
conn.close()

print('Students checked:')
for r in students:
    roll_no = r[0]
    name, gender, year, tag_color = r[1], r[2], r[3], r[4]
    has_emb = roll_no in embeddings_db and len(embeddings_db[roll_no]) > 0
    pdir = ref_dir / roll_no
    has_img = False
    if pdir.exists():
        images = list(pdir.glob('*.jpg')) + list(pdir.glob('*.png')) + list(pdir.glob('*.jpeg'))
        if len(images) > 0:
            has_img = True
            
    if not has_emb or not has_img:
        success, reason = _auto_generate_embedding(roll_no, name, gender, year, tag_color)
        print(f'Student: {roll_no}, success: {success}, reason: {reason}')
    else:
        print(f'Student: {roll_no}, has_emb: {has_emb}, has_img: {has_img}')
