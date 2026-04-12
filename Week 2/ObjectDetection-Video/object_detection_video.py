import os
from ultralytics import YOLO

# Load the latest YOLO26n model (nano version for speed)
model = YOLO("yolo26n.pt")

directory = r"C:\Users\ishit\CV_IIITH\Week 2\ObjectDetection-Video\frames"

for i in range(1800):
    frame_path = os.path.join(directory, f"frame_{i+1:04d}.png")
    if os.path.exists(frame_path):
        results = model(frame_path)
        results[0].save()
        
    else:
        print(f"Frame {frame_path} does not exist.")