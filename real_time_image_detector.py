import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# --- 1. MANUAL FILE SELECTION ---
def select_input_file():
    """Opens a file dialog window to select an image file."""
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    
    file_path = filedialog.askopenfilename(
        title="Select Image File for Detection",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

# Get the directory where your script is saved
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define model file locations
weights_path = os.path.join(base_dir, "weights", "yolov3-tiny.weights")
cfg_path = os.path.join(base_dir, "cfg", "yolov3-tiny.cfg")
names_path = os.path.join(base_dir, "coco.names")

# Manually select the image file
image_path = select_input_file()

# --- 2. HEALTH CHECK ---
if not image_path:
    print("No image selected. Exiting...")
    exit()

for path in [weights_path, cfg_path, names_path]:
    if not os.path.exists(path):
        print(f"ERROR: Model file not found at {path}")
        exit()

# --- 3. LOAD YOLO ---
net = cv2.dnn.readNet(weights_path, cfg_path)

classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- 4. LOAD AND PROCESS THE IMAGE ---
frame = cv2.imread(image_path)
if frame is None:
    print("Error: Could not load image.")
    exit()

height, width, _ = frame.shape

# Detecting objects (Blob conversion)

blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.3:
            # Calculate coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates (top-left)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 5. NON-MAXIMUM SUPPRESSION

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y - 10), 
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

# --- 6. DISPLAY RESULT ---
print(f"Object detection complete for: {os.path.basename(image_path)}")
cv2.imshow("YOLO Image Detection", frame)

# Wait indefinitely until a key is pressed
print("Press any key on the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()