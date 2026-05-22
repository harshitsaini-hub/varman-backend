import json
import logging

import redis
from celery import Celery
from celery.schedules import crontab

from config import REDIS_URL
from db import get_db_session
from services import bloom_service, db_service, notification_service
from services.image_pipeline import process_image_file

logger = logging.getLogger(__name__)
redis_client = redis.from_url(REDIS_URL)

app = Celery("amor", broker=REDIS_URL, backend=REDIS_URL)
celery_app = app

app.conf.beat_schedule = {
    "rebuild-bloom-filter-daily": {
        "task": "celery_worker.rebuild_global_bloom",
        "schedule": crontab(hour="0", minute="0"),
    },
}
app.conf.update(timezone="UTC")


@app.task(name="celery_worker.rebuild_global_bloom")
def rebuild_global_bloom():
    db = None
    try:
        db = get_db_session()
        phash_count = db_service.count_all_phashes(db)
        bloom_data = bloom_service.build_global_bloom_filter_from_iterable(
            db_service.iter_all_phashes(db),
            capacity=phash_count,
        )
        redis_client.set("global_bloom_filter", json.dumps(bloom_data), ex=90000)
        logger.info(
            "[BLOOM] Rebuilt. %s hashes. Salt: %s",
            phash_count,
            bloom_service.get_daily_salt(),
        )
    except Exception:
        logger.exception("[BLOOM] Rebuild failed")
    finally:
        if db is not None:
            db.close()


@app.task(
    name="celery_worker.process_image",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
)
def process_image(self, user_id: str, temp_file_path: str):
    try:
        return process_image_file(
            user_id=user_id,
            temp_file_path=temp_file_path,
            queue_bloom_rebuild=lambda: rebuild_global_bloom.apply_async(countdown=5),  # type: ignore[attr-defined]
        )
    except FileNotFoundError as exc:
        logger.error("[TASK] File error: %s", exc)
        raise self.retry(exc=exc, max_retries=0) from exc
    except Exception:
        logger.exception("[TASK] Pipeline error for user %s", user_id)
        raise


@app.task(name="celery_worker.send_radar_alert")
def send_radar_alert_task(
    user_id: str,
    suspect_url: str,
    image_url: str,
    platform: str,
    context: str,
) -> None:
    notification_service.send_radar_alert(
        user_id=user_id,
        suspect_url=suspect_url,
        image_url=image_url,
        platform=platform,
        context=context,
    )
