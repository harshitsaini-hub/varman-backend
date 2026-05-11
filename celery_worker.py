import os
import uuid
import face_recognition
import imagehash
from PIL import Image
from celery import Celery

from services.amor_service import apply_adversarial_noise, apply_watermark
from services.db_service import save_image_metadata

celery_app = Celery('amor_worker', broker='redis://localhost:6373/0')

@celery_app.task
def process_image_task(user_id: str, file_path: str):
    noised_path = None
    final_path = None
    
    try:
        print(f"[WORKER] Picked up job for: {file_path}")

        img = Image.open(file_path)
        real_phash = str(imagehash.phash(img))

        noised_path = apply_adversarial_noise(file_path)
        watermark_id = f"W-{uuid.uuid4().hex[:8]}"
        final_path = apply_watermark(noised_path, watermark_id)

        image_data = face_recognition.load_image_file(final_path)
        encodings = face_recognition.face_encodings(image_data)

        if len(encodings) > 0:
            face_vector = encodings[0]

            save_image_metadata(user_id, real_phash, watermark_id, face_vector)
            print("[WORKER] Image successfully armored and secured.")
        else:
            print(f"[WORKER] ERROR: No face detected in {file_path}")

    except Exception as e:
        print(f"[WORKER] FAILED to process {file_path}: {e}")
        
    finally:

        print("[WORKER] Sweeping the floor... deleting temporary files.")
        for path in [file_path, noised_path, final_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Error deleting {path}: {e}")