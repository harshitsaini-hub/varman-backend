from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from typing import List
import face_recognition
from config import STORAGE_DIR
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from services.db_service import init_db, save_image_metadata
from services.amor_service import apply_adversarial_noise, apply_watermark

app = FastAPI(title="Project AMOR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()
    print("AMOR Database (FAISS + SQLite) initialized.")

def process_image_background(user_id: str, file_path: str):
    try:
        print(f"[WORKER] Starting protection for: {file_path}")

        noised_path = apply_adversarial_noise(file_path)

        watermark_id = f"W-{uuid.uuid4().hex[:8]}"
        final_path = apply_watermark(noised_path, watermark_id)

        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            face_encoding = encodings[0]
            fake_phash = "a1b2c3d4"

            save_image_metadata(user_id, fake_phash, watermark_id, face_encoding)
            print(f"[WORKER] Successfully secured and saved: {file_path}")
        else:
            print(f"[WORKER] ERROR: No face detected in {file_path}")
            
    except Exception as e:
        print(f"[WORKER] FAILED to process {file_path}: {e}")

    @app.post("/protect")
    async def protect_images(
        background_tasks: BackgroundTasks, 
        user_id: str = Form(...),
        files: List[UploadFile] = File(...)
    ):
        saved_paths = []
        
        for file in files:
            safe_filename = file.filename if file.filename else "fallback.jpg"
            file_ext = safe_filename.split(".")[-1]
            
            temp_name = f"{uuid.uuid4()}.{file_ext}"
            temp_path = os.path.join(STORAGE_DIR, temp_name)
            
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            saved_paths.append(temp_path)
            
            background_tasks.add_task(process_image_background, user_id, temp_path)
            
        return {
            "message": "Images accepted. The Armor is being applied in the background.",
            "files_processing": len(saved_paths)
        }