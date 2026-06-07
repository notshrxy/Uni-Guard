"""
Face Embedding Database
Maintains a database of known face embeddings for identification.
For example, a database of 1000 people with 512-d face embeddings each.

The database is stored as a set of numpy arrays and a JSON metadata file.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


from facial_features.db_manager import DatabaseManager

class FaceDatabase:
    """Manages a database of face embeddings and links them to SQLite metadata."""

    def __init__(self, db_dir='face_db', similarity_threshold=0.5):
        """
        Args:
            db_dir: Directory to store face embeddings.
            similarity_threshold: Minimum cosine similarity for a match.
        """
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        
        # Initialize SQL Database Manager
        self.sql_db = DatabaseManager(self.db_dir / 'uniguard.db')

        # In-memory database for embeddings: {roll_no: [embeddings]}
        self.embeddings_db = {}
        self._load_embeddings()

    def _embeddings_path(self):
        return self.db_dir / 'embeddings.npz'

    def _load_embeddings(self):
        """Load existing embeddings from disk."""
        emb_path = self._embeddings_path()
        if emb_path.exists():
            data = np.load(str(emb_path), allow_pickle=True)
            for key in data.files:
                roll_no = key.replace('emb_', '')
                self.embeddings_db[roll_no] = list(data[key])
            print(f"[FaceDB] Loaded embeddings for {len(self.embeddings_db)} students.")
        else:
            print(f"[FaceDB] No existing embeddings found at {self.db_dir}")

    def save_embeddings(self):
        """Persist embeddings to disk."""
        emb_dict = {f'emb_{roll_no}': np.array(embs) for roll_no, embs in self.embeddings_db.items()}
        np.savez(str(self._embeddings_path()), **emb_dict)

    def register_face(self, roll_no, name, gender, year, tag_color, embedding, face_image=None):
        """Register a new face embedding and student details."""
        # 1. Update SQLite Metadata
        self.sql_db.add_student(roll_no, name, gender, year, tag_color)

        # 2. Add Embedding
        if roll_no not in self.embeddings_db:
            self.embeddings_db[roll_no] = []
        self.embeddings_db[roll_no].append(np.asarray(embedding, dtype=np.float32))

        # 3. Save Image (Reference)
        if face_image is not None:
            person_dir = self.db_dir / 'reference_faces' / roll_no
            person_dir.mkdir(parents=True, exist_ok=True)
            count = len(list(person_dir.glob('*.jpg')))
            cv2.imwrite(str(person_dir / f'face_{count}.jpg'), face_image)

        self.save_embeddings()
        print(f"[FaceDB] Registered: {name} (Roll: {roll_no})")

    def identify(self, embedding):
        """Match embedding to a student and return SQL details."""
        if not self.embeddings_db:
            return None

        query = np.asarray(embedding, dtype=np.float32).flatten()
        best_roll = None
        best_sim = -1.0

        for roll_no, stored_embs in self.embeddings_db.items():
            for stored in stored_embs:
                dot = np.dot(query, stored.flatten())
                norm = np.linalg.norm(query) * np.linalg.norm(stored)
                if norm == 0: continue
                sim = float(dot / norm)

                if sim > best_sim:
                    best_sim = sim
                    best_roll = roll_no

        if best_roll and best_sim >= self.similarity_threshold:
            print(f"[FaceDB Debug] Identified {best_roll} successfully! Similarity: {best_sim:.4f} (Threshold: {self.similarity_threshold})")
            # Fetch full details from SQLite
            student = self.sql_db.get_student_by_roll(best_roll)
            if student:
                return {
                    'person_id': best_roll,
                    'name': student['name'],
                    'gender': student['gender'],
                    'year': student['batch_year'],
                    'tag_color': student['tag_color'],
                    'similarity': best_sim
                }
        else:
            if best_roll:
                print(f"[FaceDB Debug] Match rejected: {best_roll} (Similarity: {best_sim:.4f} < Threshold: {self.similarity_threshold})")
            else:
                print("[FaceDB Debug] No matching face in database.")
        return None

    def register_from_directory(self, images_dir, face_pipeline):
        """Bulk register faces from a directory structure.

        Expected structure:
            images_dir/
                person_001/
                    photo1.jpg
                    photo2.jpg
                person_002/
                    photo1.jpg

        Args:
            images_dir: Path to the root directory.
            face_pipeline: FacePipeline instance for face detection.
        """
        images_dir = Path(images_dir)
        for person_dir in sorted(images_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            person_id = person_dir.name
            name = person_id.replace('_', ' ').title()

            for img_path in person_dir.glob('*.jpg'):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                faces = face_pipeline.app.get(img) if face_pipeline.app else []
                for face in faces:
                    self.register_face(person_id, name, face.normed_embedding, img)
                    break  # One face per image

        print(f"[FaceDB] Bulk registration complete. {len(self.database)} persons registered.")


"""
Creating a database with registered student visual identities for accurate alert + information mapping

"""