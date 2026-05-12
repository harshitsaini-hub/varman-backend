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
