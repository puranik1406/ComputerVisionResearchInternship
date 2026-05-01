from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

metrics = model.val(data="coco8-seg.yaml")
print(metrics) 