import os
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

directory = r"C:\Users\ishit\CV_IIITH\Week 3\Object-Segmentation\frames"
output_dir = r"C:\Users\ishit\CV_IIITH\Week 3\Object-Segmentation\segmented_frames"

os.makedirs(output_dir, exist_ok=True)

for i in range(1800):
    frame_path = os.path.join(directory, f"frame_{i+1:04d}.png")
    if os.path.exists(frame_path):
        results = model(
            frame_path,
            save=True,                          # ✅ tells YOLO to save output
            project=output_dir,                 # ✅ where the runs folder is created
            name="seg",                         # ✅ subfolder name inside project
            exist_ok=True                       # ✅ reuses same folder instead of seg1, seg2...
        )
    else:
        print(f"Frame {frame_path} does not exist.")