# Object Detection and Tracking using YOLOv8 + OpenCV
# Install dependencies first:
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")  

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

# Tracking IDs
track_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + tracking
    results = model.track(frame, persist=True)

    # Annotated frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Object Detection & Tracking", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# -----------------------------
#  Use video file instead of webcam
# -----------------------------
"""
cap = cv2.VideoCapture("video.mp4")
"""


# -----------------------------
#  Save output video
# -----------------------------
"""
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)
    cv2.imshow("Output", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
"""
