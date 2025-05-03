from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

print("The Model training is starting.")
model.train(
    data = "/home/berat/Computer_Vision/Road_Segmentation/dataset.yaml",
    imgsz=640,
    epochs = 120,
    batch=16,
    name="road_seg_yolov8",
    project="runs/segment",
    save=True,
)