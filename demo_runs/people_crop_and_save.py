import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

#Insert video file taken infront of department gate
cap = cv2.VideoCapture("college_gate_video.mp4")

#Counts the number of people cropped out
image_count = 0

#keeps count of number of frames, lesser framers, better performance on low power edge device
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Skip frames (process only every 10th frame)
    if frame_count % 10 != 0:
        continue
    
    #if fps = 30, 1 frame out of every 10 frames is processed, so 30/10 = 3 frames are processed in total

    #Class 0 detects people alone
    results = model(frame, classes=[0])  # detect persons

    for box in results[0].boxes:

        #Cropping the detected person's frame while maintaining slight margin
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        person_crop = frame[y1:y2, x1:x2]

        # save cropped person like person_0.jpg, person_1.jpg and so on
        filename = f"dataset/person_{image_count}.jpg"
        cv2.imwrite(filename, person_crop)

        image_count += 1

    #Shows generated frame with cropped person
    cv2.imshow("Frame", frame)

    #Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()