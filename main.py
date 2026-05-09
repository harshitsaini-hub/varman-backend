from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import uuid
import numpy as np
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DIR = os.path.join(BASE_DIR, "faces")

os.makedirs(FACE_DIR, exist_ok=True)

net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, "models", "deploy.prototxt"),
    os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
)

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    try:

        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image"}
            )

        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)

        detections = net.forward()

        faces = []

        for i in range(detections.shape[2]):

            confidence = float(detections[0, 0, i, 2])

            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                (x1, y1, x2, y2) = box.astype("int")
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                face_crop = img[y1:y2, x1:x2]

                filename = f"{uuid.uuid4()}.jpg"

                filepath = os.path.join(FACE_DIR, filename)

                cv2.imwrite(filepath, face_crop)

                faces.append({
                    "id": i,
                    "confidence": round(confidence, 4),
                    "saved_as": filename,
                    "box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                })

        return {
            "face_count": len(faces),
            "faces": faces
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )