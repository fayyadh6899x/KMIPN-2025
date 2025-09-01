from ultralytics import YOLO

model = YOLO("runs/detect/train12/weights/best.pt")

results = model.predict(source="../data/test/images", conf=0.25, iou=0.45, device="cpu", save=True)

print("Prediksi tersimpan di :", results[0].save_dir)

