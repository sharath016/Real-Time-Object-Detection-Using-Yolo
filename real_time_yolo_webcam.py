from ultralytics import YOLO
import cv2
import math

# 1. Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Width
cap.set(4, 480) # Height

# 2. Load Model
# Note: Ensure the path is correct or just use "yolov8n.pt" to auto-download
model = YOLO("yolov8n.pt")

# 3. Object Classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    
    # Check if frame was successfully captured
    if not success:
        print("Ignoring empty camera frame.")
        break

    # Run YOLOv8 on the frame
    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # Class name
            cls = int(box.cls[0])

            # Text details
            org = [x1, y1 - 10] # Moved slightly up so it's not on the line
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, f'{classNames[cls]} {confidence}', org, font, fontScale, color, thickness)

    # RECTIFIED: Use the correct variable 'img' and ensure it's inside the loop
    cv2.imshow('Webcam', img)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# RECTIFIED: cap.release() MUST be outside the while loop
cap.release()
cv2.destroyAllWindows()