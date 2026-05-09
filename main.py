from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import uuid
import face_recognition
from services.unknown_service import init_db, save_new_unknown, get_all_unknowns

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, "unknown_faces")

known_face_encodings = []
known_face_names = []

init_db()

for filename in os.listdir(KNOWN_FACES_DIR):
    path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.splitext(filename)[0])

print(f"Loaded {len(known_face_names)} known faces")

@app.post("/recognize-face")
def recognize_face(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        results = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            saved_filename = None

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            if name == "Unknown":
                face_crop = img[top:bottom, left:right]
                saved_filename = f"U-{uuid.uuid4().hex[:8]}.jpg"
                save_path = os.path.join(UNKNOWN_FACES_DIR, saved_filename)
                cv2.imwrite(save_path, face_crop)

                unknown_id = saved_filename.replace('.jpg', '')
                save_new_unknown(unknown_id, face_encoding)

            results.append({
                "name": name,
                "saved_as": saved_filename,
                "location": {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)}
            })

        return {"face_count": len(results), "results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})