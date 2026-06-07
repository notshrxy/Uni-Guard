"""
A standalone utility script that tests the raw webcam capture and YOLO model object detection logic outside the GUI framework.

"""

import cv2
import time
from ultralytics import YOLO

def main():
    model_path = "runs/detect/train/weights/best.pt"
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("Testing webcam for 5 seconds. Please stand in front of the camera with your ID card visible!")
    start_time = time.time()
    
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        results = model(frame, conf=0.1, verbose=False)[0]
        
        print(f"\n--- Frame at {time.time() - start_time:.1f}s ---")
        if len(results.boxes) == 0:
            print("  No detections.")
        else:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = model.names.get(cls, "unknown")
                print(f"  • {cls_name} (Class {cls}): {conf:.3f} | Box: {[round(x,1) for x in xyxy]}")
                
        time.sleep(0.5)
        
    cap.release()
    print("\nTest finished!")

if __name__ == "__main__":
    main()
