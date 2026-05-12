import os
import uuid
from typing import Annotated

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import STORAGE_DIR
from services.amor_service import armor_image
from services.db_service import init_db, save_image_metadata
from utils.face import extract_face_encoding
from utils.hashing import compute_phash

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
    print("AMOR Database (PostgreSQL + pgvector) initialized.")


def process_image_background(user_id: str, file_path: str):
    try:
        print(f"[WORKER] Starting protection for: {file_path}")

        image = Image.open(file_path).convert("RGB")
        image_array = np.array(image)
        phash = compute_phash(image_array)

        watermark_id = f"W-{uuid.uuid4().hex[:8]}"
        armored_array, validation = armor_image(image_array, watermark_id)
        armored_path = f"{file_path}.armored.jpg"
        Image.fromarray(armored_array).save(armored_path, format="JPEG", quality=95)

        face_encoding = extract_face_encoding(image_array)
        save_image_metadata(
            user_id=user_id,
            phash=phash,
            watermark_id=watermark_id,
            face_encoding=face_encoding,
            armored_image_path=armored_path,
            validation_passed=validation["passed"],
            compression_quality_tested=validation["compression_quality_tested"],
        )
        print(f"[WORKER] Successfully secured and saved: {file_path}")

    except Exception as e:
        print(f"[WORKER] FAILED to process {file_path}: {e}")


@app.post("/protect")
async def protect_images(
    background_tasks: BackgroundTasks,
    user_id: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
):
    saved_paths = []

    for file in files:
        safe_filename = os.path.basename(file.filename or "fallback.jpg")
        file_ext = safe_filename.rsplit(".", maxsplit=1)[-1] if "." in safe_filename else "jpg"

        temp_name = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(STORAGE_DIR, temp_name)

        with open(temp_path, "wb") as f:
            f.write(await file.read())
        saved_paths.append(temp_path)

        background_tasks.add_task(process_image_background, user_id, temp_path)

    return {
        "message": "Images accepted. The Armor is being applied in the background.",
        "files_processing": len(saved_paths),
    }
