from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS, GPSTAGS

model = YOLO("models/best.pt")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()


def get_gps(image_path):

    try:
        img = Image.open(image_path)
        exif = img._getexif()

        if exif and 34853 in exif:
            gps_data = exif[34853]

            # Cek ada koordinat atau tidak
            if 2 in gps_data and 4 in gps_data:
                lat = gps_data[2]
                lon = gps_data[4]

                # Konversi ke decimal (convert Fraction ke float)
                lat_decimal = float(lat[0]) + float(lat[1]) / 60 + float(lat[2]) / 3600
                lon_decimal = float(lon[0]) + float(lon[1]) / 60 + float(lon[2]) / 3600

                # Cek arah mata angin
                if 1 in gps_data and gps_data[1] == "S":
                    lat_decimal = -lat_decimal
                if 3 in gps_data and gps_data[3] == "W":
                    lon_decimal = -lon_decimal

                return {
                    "latitude": round(lat_decimal, 6),
                    "longitude": round(lon_decimal, 6),
                }
        return None
    except:
        return None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return JSONResponse(status_code=400, content={"error": str(e)})

        file_extension = file.filename.split(".")[-1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        gps = get_gps(file_path)

        results = model.predict(source=file_path, save=False, conf=0.25)

        img = Image.open(file_path)

        try:
            exif = img._getexif()
            if exif is not None:
                orientation_key = 274
                if orientation_key in exif:
                    orientation = exif[orientation_key]
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
        except:
            pass

        draw = ImageDraw.Draw(img)

        # Loop untuk setiap hasil deteksi
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0].item())

                    # Bounding box
                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)

        bbox_filename = f"bbox_{uuid.uuid4()}.{file_extension}"
        bbox_path = os.path.join(UPLOAD_DIR, bbox_filename)
        img.save(bbox_path)

        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0].item())

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detections.append(
                        {
                            "class": class_name,
                            "confidence": round(confidence, 3),
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                        }
                    )

        return JSONResponse(
            content={
                "success": True,
                "filename": unique_filename,
                "total_detections": len(detections),
                "detections": detections,
                "gps": gps,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@app.get("/")
async def root():
    return {"message": "Berjalan akses /docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
