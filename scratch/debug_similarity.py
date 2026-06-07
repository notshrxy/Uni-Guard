"""
A troubleshooting tool that tests face verification similarity scores by comparing a local image against registered embeddings.

"""

import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facial_features.facial_pipeline import FacePipeline

print("--- DIAGNOSTIC START ---")

# 1. Load registered NPZ
npz_path = Path("face_db/embeddings.npz")
if npz_path.exists():
    data = np.load(str(npz_path), allow_pickle=True)
    stored_emb = data.get("emb_43110042")
    if stored_emb is not None:
        print(f"[NPZ] Stored embedding found for 43110042. Type: {type(stored_emb)}, Shape/Length: {len(stored_emb)}")
    else:
        print("[NPZ] emb_43110042 NOT found in embeddings.npz!")
else:
    print("[NPZ] embeddings.npz does not exist!")

# 2. Init FacePipeline
pipeline = FacePipeline()

# 3. Load photo from student_photos
photo_path = Path("student_photos/43110042.jpg")
if photo_path.exists():
    img = cv2.imread(str(photo_path))
    if img is not None:
        print(f"[Photo] Successfully loaded student_photos/43110042.jpg. Shape: {img.shape}")
        faces = pipeline.app.get(img) if pipeline.app else []
        print(f"[Photo] Faces detected in photo: {len(faces)}")
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            query_emb = face.normed_embedding
            
            # Compare with NPZ
            if stored_emb is not None:
                # Note: stored_emb might be a list of embeddings if registered multiple times
                embs_to_check = stored_emb if isinstance(stored_emb, list) or (isinstance(stored_emb, np.ndarray) and stored_emb.ndim > 1) else [stored_emb]
                for idx, single_stored in enumerate(embs_to_check):
                    s_flat = np.asarray(single_stored).flatten()
                    q_flat = np.asarray(query_emb).flatten()
                    dot = np.dot(s_flat, q_flat)
                    norm = np.linalg.norm(s_flat) * np.linalg.norm(q_flat)
                    sim = dot / norm if norm != 0 else 0
                    print(f"[Similarity] Comparison with stored embedding #{idx}: {sim:.4f}")
    else:
        print("[Photo] Failed to decode image student_photos/43110042.jpg!")
else:
    print("[Photo] student_photos/43110042.jpg does not exist!")

print("--- DIAGNOSTIC END ---")
