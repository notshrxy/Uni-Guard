"""
Face mapping and recognition pipeline for face detection and recognition.
Handles face cropping, embedding extraction, and matching
Meant for students detected without ID cards.

Uses the InsightFace model to find faces in a photo 
and convert them into 512-dimensional "fingerprint" vectors.

"""

import os
import numpy as np
import cv2
from pathlib import Path


class FacePipeline:
    """Complete face detection → recognition pipeline using InsightFace."""

    def __init__(self, model_name='buffalo_l', det_size=(640, 640), similarity_threshold=0.5):
        """
        Args:
            model_name: InsightFace model pack name.
            det_size: Face detection input size.
            similarity_threshold: Cosine similarity threshold for matching.
        """
        self.similarity_threshold = similarity_threshold
        self.det_size = det_size
        self.app = None
        self._init_model(model_name)

    def _init_model(self, model_name):
        """Initialize InsightFace application."""
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            print(f"[FacePipeline] InsightFace loaded with model: {model_name}")
        except ImportError:
            print("[FacePipeline] WARNING: insightface not installed. Run: pip install insightface onnxruntime-gpu")
        except Exception as e:
            print(f"[FacePipeline] WARNING: Failed to load InsightFace: {e}")

    def crop_face_region(self, frame, person_box, margin=0.1):
        """Crop the head/face region from a person bounding box.

        Takes the upper 40% of the person box (head area) with optional margin.

        Args:
            frame: Full image (numpy array BGR).
            person_box: [x1, y1, x2, y2] of the detected person.
            margin: Extra margin around the head crop (fraction).

        Returns:
            Cropped face region or None if too small.
        """
        x1, y1, x2, y2 = map(int, person_box)
        h = y2 - y1
        w = x2 - x1

        # Upper 75% of person box = head/face region (generous to support sitting or close-up)
        head_y2 = int(y1 + 0.75 * h)
        margin_x = int(w * margin)
        margin_y = int(h * 0.75 * margin)

        # Clamp to image boundaries
        img_h, img_w = frame.shape[:2]
        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(img_w, x2 + margin_x)
        crop_y2 = min(img_h, head_y2 + margin_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.shape[0] < 30 or crop.shape[1] < 30:
            return None
        return crop

    def detect_and_embed(self, face_crop):
        """Detect face in crop and extract embedding.

        Args:
            face_crop: BGR image of head region.

        Returns:
            tuple: (face_embedding_512d, aligned_face_image, face_bbox) or (None, None, None)
        """
        if self.app is None or face_crop is None:
            return None, None, None

        faces = self.app.get(face_crop)
        if not faces:
            return None, None, None

        # Take the largest/highest-confidence face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        embedding = face.normed_embedding  # 512-d normalized vector
        bbox = face.bbox.astype(int)

        # Get aligned face for display/logging
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(face_crop.shape[1], x2)
        y2 = min(face_crop.shape[0], y2)
        aligned = face_crop[y1:y2, x1:x2]

        return embedding, aligned, bbox

    def process_violator(self, frame, person_box):
        """Full pipeline: crop face from person box → detect → embed.

        Args:
            frame: Full BGR image.
            person_box: [x1, y1, x2, y2] person bounding box.

        Returns:
            dict with keys: embedding, face_image, bbox, or None if no face found.
        """
        face_crop = self.crop_face_region(frame, person_box)
        if face_crop is None:
            return None

        embedding, aligned, bbox = self.detect_and_embed(face_crop)
        if embedding is None:
            return None

        return {
            'embedding': embedding,
            'face_image': aligned,
            'bbox': bbox,
            'head_crop': face_crop,
        }

    @staticmethod
    def cosine_similarity(a, b):
        """Compute cosine similarity between two vectors."""
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)