from ultralytics import YOLO
import os

model = YOLO(r"C:\Users\ishit\CV_IIITH\runs\detect\train\weights\best.pt")

directory = r"C:\Users\ishit\CV_IIITH\Week 5\frames"
output_dir = r"C:\Users\ishit\CV_IIITH\Week 5\detected_frames"

os.makedirs(output_dir, exist_ok=True)

for i in range(1534):

    frame_path = os.path.join(directory, f"frame_{i+1:05d}.jpg")

    if os.path.exists(frame_path):

        results = model(frame_path)

        save_path = os.path.join(
            output_dir,
            f"detected_{i+1:05d}.jpg"
        )

        results[0].save(filename=save_path)

    else:
        print(f"{frame_path} does not exist")