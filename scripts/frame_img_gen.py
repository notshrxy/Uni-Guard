"""
Run the person cropping py script by modifying the source to "college_gate"video.mp4" i.e. the video file that we record,
infront of Department gate, this way, people in the video are automatically identified, cropped, and saved.

This script extracts frames from a video file, detects people using YOLOv8, and saves cropped images of people.

"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# Folders
VIDEO_DIR = "raw_videos"
OUTPUT_DIR = "dataset"


# Ensure folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

# Counters
image_count = 0
frame_count = 0

# Prevent overwriting by finding the highest existing image number
existing_files = list(Path(OUTPUT_DIR).glob("person_*.jpg"))
if existing_files:
    save_count = max([int(f.stem.split("_")[1]) for f in existing_files]) + 1
    print(f"[Generator] Found existing images. Resuming save count from {save_count}")
else:
    save_count = 0


# Track processed videos to avoid re-processing
PROCESSED_LOG = Path(VIDEO_DIR) / ".processed_videos.txt"
processed_videos = set()
if PROCESSED_LOG.exists():
    with open(PROCESSED_LOG, "r") as f:
        processed_videos = set(f.read().splitlines())

# Finding all mp4 files in the raw_videos folder
videos = list(Path(VIDEO_DIR).glob("*.mp4"))
print(f"[Generator] Found {len(videos)} videos in '{VIDEO_DIR}'")

for v_path in videos:
    if v_path.name in processed_videos:
        print(f"[Skip] '{v_path.name}' has already been processed.")
        continue

    print(f"[Processing] {v_path.name}...")
    cap = cv2.VideoCapture(str(v_path))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. Skip frames (process only every 30th frame - about every 1 second)
        if frame_count % 30 != 0:
            continue

        results = model(frame, classes=[0], verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Add margin to the bounding box
            margin = 30
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)

            person_crop = frame[y1:y2, x1:x2]
            
            # 2. Filter out poor quality crops (smaller than 100px)
            if person_crop.shape[0] < 200 or person_crop.shape[1] < 200:
                continue

            # 3. Save every 5th person detected
            if image_count % 5 == 0:
                filename = f"{OUTPUT_DIR}/person_{save_count}.jpg"
                cv2.imwrite(filename, person_crop)
                save_count += 1   

            image_count += 1

        # Show preview (Commented out to prevent GUI errors)
        # cv2.imshow("UniGuard Dataset Gen", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     exit()

    cap.release()
    
    # Mark video as processed
    with open(PROCESSED_LOG, "a") as f:
        f.write(v_path.name + "\n")
    
print(f"\n[Done] Successfully saved {save_count} images to '{OUTPUT_DIR}/'")
# cv2.destroyAllWindows()