from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, "models", "deploy.prototxt"),
    os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
)

app = FastAPI()

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})
   
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
        
        _, buffer = cv2.imencode(".jpg", img)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})