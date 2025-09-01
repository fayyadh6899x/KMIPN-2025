from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from ultralytics import YOLO
from PIL import Image

model = YOLO("models/best.pt")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"error": "No file provided"}
            )
        
        file_extension = file.filename.split(".")[-1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        results = model.predict(source=file_path, save=False, conf=0.25)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0].item())
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1)
                    })
        
        os.remove(file_path)

        return JSONResponse(content={
            "success": True,
            "filename": unique_filename,
            "total_detections": len(detections),
            "detections": detections
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.get("/")
async def root():
    return {"message": "Mantap", "status": "Siap pakai"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)