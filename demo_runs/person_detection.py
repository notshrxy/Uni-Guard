import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    # Run YOLO detection
    results = model(frame, classes=[0,24], verbose=False)

    person_id = 0
    for box in results[0].boxes:

        margin = 20

        cls = int(box.cls[0])

        if cls == 0:  # person

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)

            # Crop the detected person
            person_crop = frame[y1:y2, x1:x2]

            cv2.imshow("Person Crop", person_crop)

            #Adding a counter to count number of people
            person_id += 1
      
    # Draw detections
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("UniGuard Detection", annotated_frame)

    key = cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()