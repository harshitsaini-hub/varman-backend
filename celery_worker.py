# celery_worker.py
import logging

import redis
from celery import Celery
from celery.schedules import crontab

from core.config import REDIS_URL
from db import get_db_session
from services import bloom_service, db_service
from services.image_pipeline import process_image_file

logger = logging.getLogger(__name__)
redis_client = redis.from_url(REDIS_URL)

app = Celery("amor", broker=REDIS_URL, backend=REDIS_URL)

app.conf.beat_schedule = {
    "rebuild-bloom-filter-daily": {
        "task": "celery_worker.rebuild_global_bloom",
        "schedule": crontab(hour=0, minute=0),
    },
}
app.conf.timezone = "UTC"


@app.task(name="celery_worker.rebuild_global_bloom")
def rebuild_global_bloom() -> dict[str, int | str]:
    """Rebuild the daily salted Bloom filter and cache it in Redis."""
    db = None
    try:
        db = get_db_session()
        all_phashes = db_service.get_all_phashes(db)
        bloom_b64 = bloom_service.build_global_bloom_filter(all_phashes)
        redis_client.set("global_bloom_filter", bloom_b64, ex=90000)  # 25hr TTL
        logger.info(
            "[BLOOM] Rebuilt. %s hashes. Salt: %s",
            len(all_phashes),
            bloom_service.get_daily_salt(),
        )
        return {"status": "success", "hash_count": len(all_phashes)}
    except Exception as exc:
        logger.exception("[BLOOM] Rebuild failed")
        return {"status": "failed", "reason": str(exc)}
    finally:
        if db is not None:
            db.close()


@app.task(
    name="celery_worker.process_image",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
    dont_autoretry_for=(FileNotFoundError,),
)
def process_image(self, user_id: int, temp_file_path: str) -> dict:
    """Run the shared AMOR image pipeline inside a Celery worker."""
    logger.info("[TASK] Starting pipeline. User: %s, File: %s", user_id, temp_file_path)

    try:
        result = process_image_file(
            user_id=user_id,
            temp_file_path=temp_file_path,
            queue_bloom_rebuild=lambda: rebuild_global_bloom.apply_async(countdown=5),
        )
        logger.info("[TASK] Pipeline finished: %s", result)
        return result
    except FileNotFoundError as exc:
        logger.error("[TASK] File error: %s", exc)
        return {"status": "failed", "reason": "temp_file_missing", "path": temp_file_path}
    except Exception:
        logger.exception("[TASK] Pipeline error for user %s", user_id)
        raise
