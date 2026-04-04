from ultralytics import YOLO

# Load the latest YOLO26n model (nano version for speed)
model = YOLO("yolo26n.pt")

url = "https://ultralytics.com/images/bus.jpg"
# Run inference on an image from a URL
results = model(url)

# Display the results with bounding boxes
results[0].show()