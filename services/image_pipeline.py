import os
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from core.config import TEMP_STORAGE_PATH
from services import amor_service, db_service, notification_service
from utils.face import extract_face_encoding
from utils.hashing import compute_phash


def process_image_file(
    user_id: str | int,
    temp_file_path: str,
    *,
    db: Any | None = None,
    queue_bloom_rebuild: Callable[[], None] | None = None,
    delete_original: bool = True,
) -> dict[str, Any]:
    """Run the AMOR image protection pipeline for one temporary image file.

    This service is shared by FastAPI background tasks and Celery tasks so the
    API and worker paths cannot drift into different behavior.
    """
    if not os.path.exists(temp_file_path):
        raise FileNotFoundError(f"Temp file missing: {temp_file_path}")

    armored_output_path = None
    owns_db_connection = db is None

    try:
        image = Image.open(temp_file_path).convert("RGB")
        image_array = np.array(image)
        phash = compute_phash(image_array)
        face_vector = extract_face_encoding(image_array)

        watermark_id = str(uuid.uuid4())
        armored_array, validation_report = amor_service.armor_image(image_array, watermark_id)

        if not validation_report["passed"]:
            _handle_validation_failure(str(user_id), watermark_id, validation_report)
            if delete_original:
                safe_delete(temp_file_path)
            return {
                "status": "failed",
                "reason": "armor_validation_failed",
                "watermark_id": watermark_id,
                "details": validation_report,
            }

        armored_output_path = os.path.join(
            TEMP_STORAGE_PATH, f"armored_{user_id}_{uuid.uuid4().hex}.jpg"
        )
        Image.fromarray(armored_array).save(armored_output_path, format="JPEG", quality=95)

        if db is None:
            db = db_service.get_db_connection()

        db_service.save_protected_image(
            db=db,
            user_id=str(user_id),
            phash=phash,
            watermark_id=watermark_id,
            face_vector=face_vector,
            armored_image_path=armored_output_path,
            validation_passed=validation_report["passed"],
            compression_quality_tested=validation_report["compression_quality_tested"],
        )

        if delete_original:
            safe_delete(temp_file_path)

        if queue_bloom_rebuild is not None:
            queue_bloom_rebuild()

        return {
            "status": "success",
            "user_id": str(user_id),
            "watermark_id": watermark_id,
            "phash": phash,
            "face_detected": face_vector is not None,
            "validation": validation_report,
            "armored_path": armored_output_path,
        }
    except Exception:
        if delete_original:
            safe_delete(temp_file_path)
        if armored_output_path:
            safe_delete(armored_output_path)
        raise
    finally:
        if owns_db_connection and db is not None:
            db.close()


def _handle_validation_failure(user_id: str, watermark_id: str, report: dict[str, Any]) -> None:
    notification_service.send_ops_alert(
        subject="AMOR Armor Validation Failed",
        body=(
            f"User {user_id}'s image failed watermark validation after compression simulation.\n"
            f"Watermark ID: {watermark_id}\n"
            f"Recovered prefix: {report['recovered_prefix']}\n"
            f"Tested at JPEG quality: {report['compression_quality_tested']}\n"
            "Image was NOT delivered. Manual review required."
        ),
    )


def safe_delete(path: str | None) -> None:
    """Delete a file if it exists; cleanup must not mask pipeline errors."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
