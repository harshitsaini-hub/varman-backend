import logging

logger = logging.getLogger(__name__)


def send_ops_alert(subject: str, body: str) -> None:
    logger.error("[OPS ALERT] %s\n%s", subject, body)


def send_radar_alert(
    user_id: str,
    suspect_url: str,
    image_url: str,
    platform: str,
    context: str,
) -> None:
    logger.warning(
        "[RADAR ALERT] user=%s platform=%s suspect=%s image=%s context=%s",
        user_id,
        platform,
        suspect_url,
        image_url,
        context,
    )


def queue_radar_alert(
    user_id: str,
    suspect_url: str,
    image_url: str,
    platform: str,
    context: str,
) -> bool:
    """Queue user-facing alerts so HTTP/scraper paths never wait on notification I/O."""
    try:
        from celery_worker import send_radar_alert_task

        send_radar_alert_task.delay(  # type: ignore[attr-defined]
            user_id=user_id,
            suspect_url=suspect_url,
            image_url=image_url,
            platform=platform,
            context=context,
        )
        return True
    except Exception:
        logger.exception(
            "[RADAR ALERT QUEUE FAILED] user=%s platform=%s suspect=%s",
            user_id,
            platform,
            suspect_url,
        )
        return False
