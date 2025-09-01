from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="../data/data.yaml", 
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=0
)

print("Model :", results.save_dir)
