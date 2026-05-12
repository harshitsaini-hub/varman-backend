import os
import uuid
import numpy as np
import redis
import json
from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger
from PIL import Image

from config import REDIS_URL, TEMP_STORAGE_PATH
from db import get_db_session
from services import amor_service, bloom_service, db_service, notification_service
from utils.face import extract_face_encoding
from utils.hashing import compute_phash

logger = get_task_logger(__name__)
redis_client = redis.from_url(REDIS_URL)

app = Celery("amor", broker=REDIS_URL, backend=REDIS_URL)

# ── Beat Schedule ──────────────────────────────────────────────────────────

app.conf.beat_schedule = {
    "rebuild-bloom-filter-daily": {
        "task": "celery_worker.rebuild_global_bloom",
        "schedule": crontab(hour="0", minute="0"),
    },
}
app.conf.update(timezone="UTC")

# ── Scheduled Task ─────────────────────────────────────────────────────────


@app.task(name="celery_worker.rebuild_global_bloom")
def rebuild_global_bloom():

    db = get_db_session()
    try:
        all_phashes = db_service.get_all_phashes(db)
        bloom_data = bloom_service.build_global_bloom_filter(all_phashes)
        redis_client.set("global_bloom_filter", json.dumps(bloom_data), ex=90000)
        logger.info(
            "[BLOOM] Rebuilt. %s hashes. Salt: %s",
            len(all_phashes),
            bloom_service.get_daily_salt(),
        )
    except Exception as e:
        logger.error(f"[BLOOM] Rebuild failed: {e}")
    finally:
        db.close()


# ── Main Pipeline ──────────────────────────────────────────────────────────


@app.task(
    name="celery_worker.process_image",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
)
def process_image(self, user_id: str, temp_file_path: str):

    db = get_db_session()
    armored_output_path = None

    try:
        # ── Step 1: Load ───────────────────────────────────────────────────
        logger.info(f"[TASK] Starting pipeline. User: {user_id}, File: {temp_file_path}")

        if not os.path.exists(temp_file_path):
            raise FileNotFoundError(f"Temp file missing: {temp_file_path}")

        image = Image.open(temp_file_path).convert("RGB")
        image_array = np.array(image)
        logger.info(f"[TASK] Loaded image. Shape: {image_array.shape}")

        # ── Step 2: pHash (on clean original, before any modification) ─────
        phash = compute_phash(image_array)
        logger.info(f"[TASK] pHash computed: {phash}")

        # ── Step 3: Face Encoding ──────────────────────────────────────────
        face_vector = extract_face_encoding(image_array)
        if face_vector is None:
            # Non-fatal: user may upload a non-face image (artwork, screenshot)
            # We still protect it via pHash + watermark, just no face vector
            logger.warning("[TASK] No face detected. Proceeding without face vector.")

        # ── Step 4 + 5 + 6: Armor + Validate (single call to amor_service) ─
        watermark_id = str(uuid.uuid4())
        armored_array, validation_report = amor_service.armor_image(image_array, watermark_id)

        logger.info(
            f"[TASK] Armor validation result: passed={validation_report['passed']} | "
            f"watermark_id={watermark_id} | "
            f"recovered_prefix={validation_report['recovered_prefix']} | "
            f"compression_tested={validation_report['compression_quality_tested']}q"
        )

        # ── Step 6a: Hard stop if validation failed ────────────────────────
        if not validation_report["passed"]:
            _handle_validation_failure(user_id, watermark_id, validation_report)
            return {
                "status": "failed",
                "reason": "armor_validation_failed",
                "watermark_id": watermark_id,
                "details": validation_report,
            }

        # ── Step 7: Persist armored image to disk ──────────────────────────
        armored_output_path = os.path.join(
            TEMP_STORAGE_PATH, f"armored_{user_id}_{uuid.uuid4().hex}.jpg"
        )
        armored_image = Image.fromarray(armored_array)
        armored_image.save(armored_output_path, format="JPEG", quality=95)
        logger.info(f"[TASK] Armored image saved: {armored_output_path}")

        # ── Step 8: Save to DB ─────────────────────────────────────────────
        db_service.save_protected_image(
            db=db,
            user_id=user_id,
            phash=phash,
            watermark_id=watermark_id,
            face_vector=face_vector,  # None is acceptable here
            armored_image_path=armored_output_path,
            validation_passed=validation_report["passed"],
            compression_quality_tested=validation_report["compression_quality_tested"],
        )
        logger.info(f"[TASK] DB record saved. Watermark ID: {watermark_id}")

        # ── Step 9: Clean up temp (original only, keep armored) ───────────
        _safe_delete(temp_file_path)
        logger.info(f"[TASK] Temp file deleted: {temp_file_path}")

        rebuild_global_bloom.apply_async(countdown=5)  # type: ignore[attr-defined]
        logger.info("[TASK] Bloom rebuild queued.")

        return {
            "status": "success",
            "user_id": user_id,
            "watermark_id": watermark_id,
            "phash": phash,
            "face_detected": face_vector is not None,
            "validation": validation_report,
            "armored_path": armored_output_path,
        }

    except FileNotFoundError as e:
        logger.error(f"[TASK] File error: {e}")
        # Don't retry on missing file — it won't appear on retry
        raise self.retry(max_retries=0) from None

    except Exception as e:
        logger.error(f"[TASK] Pipeline error for user {user_id}: {e}", exc_info=True)
        _safe_delete(temp_file_path)
        raise  # Celery autoretry handles this (max_retries=3)

    finally:
        db.close()


# ── Helpers ────────────────────────────────────────────────────────────────


def _handle_validation_failure(user_id: str, watermark_id: str, report: dict):
    """
    Called when the armored image fails compression validation.
    Logs it. Notifies ops. Does NOT deliver the image to the user.
    Future: queue for re-processing with adjusted epsilon.
    """
    logger.error(
        f"[VALIDATION FAIL] user_id={user_id} watermark_id={watermark_id} "
        f"recovered={report['recovered_prefix']} "
        f"tested_at_quality={report['compression_quality_tested']}"
    )
    # Notify ops channel
    notification_service.send_ops_alert(
        subject="AMOR Armor Validation Failed",
        body=(
            f"User {user_id}'s image failed watermark validation after compression simulation.\n"
            f"Watermark ID: {watermark_id}\n"
            f"Recovered prefix: {report['recovered_prefix']}\n"
            f"Tested at JPEG quality: {report['compression_quality_tested']}\n"
            f"Image was NOT delivered. Manual review required."
        ),
    )


def _safe_delete(path: str):
    """Deletes a file silently. Never crashes the pipeline on cleanup failure."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"[CLEANUP] Failed to delete {path}: {e}")
